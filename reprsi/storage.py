from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def append_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def journal(path, repair=False):
    path = Path(path)
    if not path.exists():
        return []
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    rows, offset = [], 0
    for index, line in enumerate(lines):
        if not line.endswith(b"\n"):
            if not repair or index != len(lines) - 1:
                raise ValueError("Incomplete journal record")
            with path.open("r+b") as stream:
                stream.truncate(offset)
            break
        rows.append(json.loads(line))
        offset += len(line)
    return rows


def link_or_copy(source, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".link.tmp")
    temporary.unlink(missing_ok=True)
    try:
        os.link(source, temporary)
    except OSError:
        shutil.copyfile(source, temporary)
    os.replace(temporary, destination)


@contextmanager
def exclusive(directory):
    import fcntl
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("Another process is writing this run directory") from exc
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


class RunState:
    def __init__(self, output, policy, config, inputs, resume=False):
        self.output = Path(output)
        self.states = self.output / "states"
        self.states.mkdir(parents=True, exist_ok=True)
        self.keep_every = int(config.get("keep_every", 0))
        portable = {k: v for k, v in config.items() if not k.endswith("_root")}
        scientific = {k: v for k, v in portable.items() if k not in
                      {"rounds", "max_gpu_seconds", "keep_every", "max_units"}}
        self.signature = digest({"config": scientific, "inputs": inputs})
        pointer = self.output / "progress.json"
        if pointer.exists():
            if not resume:
                raise ValueError("Run already exists; pass --resume")
            self.progress = json.loads(pointer.read_text())
            if self.progress.get("signature") != self.signature:
                raise ValueError("Resume configuration or input fingerprint changed")
        else:
            if resume:
                raise ValueError("No committed run exists to resume")
            if (self.output / "config.json").exists():
                raise ValueError("Incomplete initialization; use a new output directory")
            policy.save(self.output / "initial.pt")
            first = self.states / "step_000000"
            first.mkdir()
            for name in ("teacher.pt", "student.pt"):
                link_or_copy(self.output / "initial.pt", first / name)
            atomic_json(self.output / "config.json", portable)
            atomic_json(self.output / "inputs.json", inputs)
            self.progress = {"completed_rounds": 0, "state": first.name,
                             "signature": self.signature, "format_version": 2}
            atomic_json(pointer, self.progress)
        self.work = self.output / "working"
        if self.work.exists():
            shutil.rmtree(self.work)
        self.work.mkdir()
        self._aliases()

    @property
    def completed(self):
        return self.progress["completed_rounds"]

    @property
    def current(self):
        return self.states / self.progress["state"]

    @property
    def student(self):
        return self.current / "student.pt"

    @property
    def teacher(self):
        return self.current / "teacher.pt"

    def _aliases(self):
        for name in ("teacher.pt", "student.pt"):
            source = self.current / name
            if not source.is_file():
                raise ValueError("Committed checkpoint is missing")
            link_or_copy(source, self.output / name)
        row = self.current / "record.json"
        if row.exists():
            atomic_json(self.output / f"round_{self.completed-1:06d}.json",
                        json.loads(row.read_text()))

    def commit(self, teacher, student, record):
        old = self.current
        final = self.states / f"step_{self.completed+1:06d}"
        temporary = self.states / (final.name + ".pending")
        if temporary.exists():
            shutil.rmtree(temporary)
        temporary.mkdir()
        link_or_copy(teacher, temporary / "teacher.pt")
        link_or_copy(student, temporary / "student.pt")
        atomic_json(temporary / "record.json", record)
        if final.exists():
            shutil.rmtree(final)
        os.replace(temporary, final)
        self.progress = {**self.progress, "completed_rounds": self.completed + 1,
                         "state": final.name}
        atomic_json(self.output / "progress.json", self.progress)
        self._aliases()
        previous = self.completed - 1
        if previous and not (self.keep_every and previous % self.keep_every == 0):
            shutil.rmtree(old)

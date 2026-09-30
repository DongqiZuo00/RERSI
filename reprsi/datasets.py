from contextlib import closing
from io import BytesIO
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from urllib.error import URLError
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen
from zipfile import ZipFile
from .diagnostics import read_jsonl, write_jsonl
from .storage import atomic_json, digest, exclusive, file_digest


def read_records(path):
    path=Path(path)
    if path.suffix==".jsonl":return read_jsonl(path)
    if path.suffix==".parquet":
        try:import pyarrow.parquet as pq
        except ImportError as exc:raise ImportError("Install requirements-benchmarks.txt for Parquet datasets") from exc
        return pq.read_table(path).to_pylist()
    if path.suffix==".json":
        value=json.loads(path.read_text())
        return value if isinstance(value,list) else [value]
    raise ValueError("Expected JSON, JSONL or Parquet dataset")


def source_url(source):
    if source.get("repo_id"):
        if not re.fullmatch(r"[0-9a-f]{40}",source.get("revision","")):
            raise ValueError("Hugging Face datasets require a pinned commit revision")
        return "https://huggingface.co/datasets/"+quote(source["repo_id"],safe="/")+"/resolve/"+source["revision"]+"/"+quote(source["filename"],safe="/")
    url=source["url"]
    if urlparse(url).scheme not in ("http","https"):raise ValueError("Dataset URL must use HTTP or HTTPS")
    return url


def fetch(source,cache):
    expected=source["sha256"]
    if not re.fullmatch(r"[0-9a-f]{64}",expected):raise ValueError("A SHA-256 digest is required")
    suffix=Path(source.get("member") or source.get("filename") or urlparse(source.get("url","")).path).suffix
    cache=Path(cache);cache.mkdir(parents=True,exist_ok=True)
    destination=cache/(expected+suffix)
    if destination.is_file() and file_digest(destination)==expected:return destination
    headers={"User-Agent":"reprsi-datasets"}
    if source.get("token_env"):
        token=os.environ.get(source["token_env"])
        if not token:raise ValueError("Dataset access token environment variable is unset")
        headers["Authorization"]="Bearer "+token
    for attempt in range(3):
        temporary=None
        try:
            with closing(urlopen(Request(source_url(source),headers=headers),timeout=60)) as response:
                if source.get("member"):
                    with ZipFile(BytesIO(response.read())) as archive:
                        stream=BytesIO(archive.read(source["member"]))
                else:stream=response
                with tempfile.NamedTemporaryFile(dir=cache,delete=False) as output:
                    temporary=Path(output.name)
                    for block in iter(lambda:stream.read(1024*1024),b""):output.write(block)
                    output.flush();os.fsync(output.fileno())
            if file_digest(temporary)!=expected:raise ValueError("Dataset checksum mismatch")
            os.replace(temporary,destination)
            return destination
        except (URLError,TimeoutError,ConnectionError,OSError,ValueError):
            if attempt==2:raise
        finally:
            if temporary is not None:temporary.unlink(missing_ok=True)


def download(benchmark,output,manifest=None,cache=None):
    sources=json.loads(Path(manifest or Path(__file__).with_name("dataset_sources.json")).read_text())
    if benchmark not in sources:raise ValueError("Unknown dataset source")
    spec=sources[benchmark];root=Path(output)
    if not spec.get("sources"):raise ValueError("Dataset source manifest is empty")
    if any(not re.fullmatch(r"[A-Za-z0-9_-]+",s.get("split","")) for s in spec["sources"]):
        raise ValueError("Invalid dataset split name")
    signature=digest(spec);cache=Path(cache) if cache else root/"cache"
    with exclusive(root):
        path=root/"manifest.json"
        if path.exists():
            saved=json.loads(path.read_text())
            if saved["signature"]!=signature:raise ValueError("Dataset source manifest changed; use another output directory")
            if all((root/f).is_file() and file_digest(root/f)==sha for f,sha in saved["files"].items()):return saved
        groups={};receipts=[]
        for source in spec["sources"]:
            local=fetch(source,cache);rows=read_records(local)
            if len(rows)!=source["count"]:raise ValueError("Downloaded dataset row count differs")
            split=source["split"]
            if benchmark=="MATH":
                prefix=source.get("subject") or Path(source.get("filename","data")).parent.as_posix()
                rows=[{**row,"id":row.get("id",f"{split}/{prefix}/{i}"),"split":split} for i,row in enumerate(rows)]
            groups.setdefault(split,[]).extend(rows)
            receipts.append({k:v for k,v in source.items() if k!="token_env"})
        files={};counts={}
        for split,rows in groups.items():
            filename="HARP.jsonl" if benchmark=="HARP" else split+".jsonl"
            write_jsonl(root/filename,rows)
            files[filename]=file_digest(root/filename);counts[split]=len(rows)
        report={"benchmark":benchmark,"signature":signature,"files":files,"counts":counts,"sources":receipts}
        atomic_json(path,report)
        return report


def setup_dependency(name,output):
    from .benchmarks import HARP_REVISION
    from .manufactoria import DELTA_REVISION
    repositories={"HARP":("https://github.com/aadityasingh/HARP.git",HARP_REVISION,"src"),
                  "DELTA":("https://github.com/sunblaze-ucb/rl-grok-recipe.git",DELTA_REVISION,"manufactoria")}
    url,revision,subdirectory=repositories[name];root=Path(output).resolve()
    if root.exists():
        current=subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],text=True).strip()
        if current!=revision:raise ValueError("Dependency revision differs")
        subprocess.run(["git","-C",str(root),"diff","--exit-code","HEAD","--",subdirectory],check=True,stdout=subprocess.DEVNULL)
    else:
        root.parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root.parent) as temporary:
            directory=Path(temporary)/"checkout"
            subprocess.run(["git","init","-q",str(directory)],check=True)
            subprocess.run(["git","-C",str(directory),"sparse-checkout","set","--no-cone","/"+subdirectory+"/"],check=True)
            subprocess.run(["git","-C",str(directory),"fetch","--depth","1",url,revision],check=True)
            subprocess.run(["git","-C",str(directory),"checkout","--detach","FETCH_HEAD"],check=True)
            os.replace(directory,root)
    if not (root/subdirectory).is_dir():raise ValueError("Dependency source is missing")
    return {"dependency":name,"revision":revision,"path":str(root)}

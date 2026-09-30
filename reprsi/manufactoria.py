from collections import defaultdict,deque
from dataclasses import dataclass
import itertools
import json
from pathlib import Path
import random
import re
import subprocess
import sys
from .diagnostics import BOUNDARY
from .metrics import seed_for

DELTA_REVISION="8500bec984d4a84a4aa94ca3adc31c004aa6a388"
FAMILIES={"START":"starts_with","APPEND":"append_sequence","EXACT":"exact_sequence",
    "REGEX":"regex_pattern","COMPR":"numerical_comparison","HAS":"contains_substring"}
RELEASED_FAMILIES={key:{key,value} for key,value in FAMILIES.items()}
RELEASED_FAMILIES["HAS"].update({"contains_ordered","contains_count"})


def prepare_released(source,output,split,family,domain):
    from .datasets import read_records
    from .diagnostics import write_jsonl
    from .curricula import task_record
    if family not in RELEASED_FAMILIES or split not in ("train","dev","test"):
        raise ValueError("Invalid Manufactoria partition or family")
    records=read_records(source)
    if not records:raise ValueError("Empty released Manufactoria dataset")
    labels=[r.get("problem_family") or r.get("problem_type") or r.get("pattern_type") or r.get("family") for r in records]
    if any(label not in RELEASED_FAMILIES[family] for label in labels):
        raise ValueError("Released family metadata does not match --family")
    tasks=domain.load_released_tasks(records)
    if len(tasks)!=len(records):raise ValueError("Released task count changed")
    rows=[{**task_record(t),"split":split,"benchmark":"Manufactoria-"+family,"family":family,"problem_family":label}
          for t,label in zip(tasks,labels)]
    if len({r["id"] for r in rows})!=len(rows):raise ValueError("Duplicate released Manufactoria identity")
    write_jsonl(output,rows)
    return {"examples":len(rows),"families":{label:labels.count(label) for label in sorted(set(labels))}}


@dataclass
class FactoryTask:
    prompt:str
    tests:list
    parser:object
    identity:str

    def verify(self,response):
        blocks=re.findall(r"```(?:\w+)?\s*\n(.*?)```",response,flags=re.S)
        program=blocks[-1] if blocks else response.strip()
        try: factory=self.parser(program)
        except (ValueError,KeyError,IndexError,RuntimeError): return 0.0
        if not self.tests: raise ValueError("Empty Manufactoria test suite")
        for test in self.tests:
            result=factory.process_robot(test["input"])
            accepted=result.finished
            if test.get("check_output",False):
                expected=test["expected_output"]
                if any(c in expected for c in ".+*?|()"):
                    try: matches=bool(re.fullmatch(expected,result.final_tape))
                    except re.error: matches=result.final_tape==expected
                else: matches=result.final_tape==expected
                accepted=matches and result.finished
            if bool(accepted)!=bool(test["expected_accepted"]): return 0.0
        return 1.0


class Domain:
    def __init__(self,root):
        root=Path(root)
        revision=subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],text=True).strip()
        if revision!=DELTA_REVISION: raise ValueError("DELTA revision differs from tested adapter revision")
        if subprocess.run(["git","-C",str(root),"diff","--quiet","HEAD","--","manufactoria"]).returncode:
            raise ValueError("Pinned DELTA source has local modifications")
        sys.path.insert(0,str(root/"manufactoria"))
        from manufactoria_problem_generators import GeneratorRegistry,GeneratorConfig
        from hf_file_wrapper import TrainingFileWrapper
        from utils.manufactoria_parser import create_robot_factory,ParseError
        GeneratorConfig._config=GeneratorConfig._get_default_config()
        GeneratorConfig._populate_attributes()
        self.registry=GeneratorRegistry;self.cfg=GeneratorConfig;self.wrapper=TrainingFileWrapper()
        def parser(text):
            try:return create_robot_factory(text)
            except ParseError as exc:raise ValueError(str(exc)) from exc
        self.parser=parser

    def construct(self,item):
        if set(item)!={"family","parameters","composition","presentation"}:
            raise ValueError("Unexpected Manufactoria fields")
        family=item["family"]
        if family not in FAMILIES or item["composition"]!=[] or item["presentation"]!="default":
            raise ValueError("Use one family constructor, empty composition, default presentation")
        p=dict(item["parameters"]);mode=p.pop("color_mode","two_color")
        if mode not in {"two_color","four_color"}:raise ValueError("Invalid color mode")
        chars=list("RB" if mode=="two_color" else "RBYG")
        if family in {"START","APPEND","EXACT","HAS"}:
            if set(p)!={"sequence"}:raise ValueError("Expected sequence parameter")
            lo,hi=self.cfg.get_sequence_length_range(FAMILIES[family])
            if not lo<=len(p["sequence"])<=hi or set(p["sequence"])-set(chars):raise ValueError("Invalid sequence")
        elif family=="COMPR":
            if set(p)!={"threshold","operator"} or mode!="two_color":raise ValueError("Invalid comparison parameters")
            lo,hi=self.cfg.NUMERICAL_THRESHOLDS
            if type(p["threshold"]) is not int or not lo<=p["threshold"]<=hi or p["operator"] not in {">",">=","<","<=","="}:raise ValueError("Invalid comparison bounds")
        else:
            if set(p)!={"pattern_parts"} or len(p["pattern_parts"])!=self.cfg.REGEX_CONCATENATION_COUNT:raise ValueError("Invalid regex parts")
            limits=self.cfg.REGEX_MAX_PATTERN_LENGTH
            lo,hi=limits if isinstance(limits,(list,tuple)) else (1,limits)
            for part in p["pattern_parts"]:
                if not isinstance(part,list) or len(part)!=2:raise ValueError("Invalid regex part")
                word,op=part
                if not lo<=len(word)<=hi or set(word)-set(chars) or op not in self.cfg.REGEX_OPERATORS:raise ValueError("Invalid regex component")
        generated=self.registry.get_generator(FAMILIES[family]).generate_problem(chars,mode=="four_color",params=p)
        formatted=self.wrapper.convert_problem(generated)
        return FactoryTask(formatted["messages"][0]["content"],formatted["ground_truth"],self.parser,json.dumps(item,sort_keys=True))

    def build_curriculum(self,text,count):
        obj=json.loads(text)
        if set(obj)!={"items"} or len(obj["items"])!=count:raise ValueError("Wrong curriculum length")
        return [self.construct(x) for x in obj["items"]]

    @staticmethod
    def curriculum_schema(count):
        string={"type":"string"};integer={"type":"integer"}
        params={"type":"object","properties":{"color_mode":{"enum":["two_color","four_color"],"type":"string"},
            "sequence":string,"threshold":integer,"operator":{"type":"string","enum":[">",">=","<","<=","="]},
            "pattern_parts":{"type":"array","items":{"type":"array","items":string,"minItems":2,"maxItems":2},"minItems":2,"maxItems":2}},"additionalProperties":False}
        item={"type":"object","properties":{"family":{"type":"string","enum":list(FAMILIES)},"parameters":params,
            "composition":{"type":"array","items":string,"maxItems":0},"presentation":{"type":"string","enum":["default"]}},
            "required":["family","parameters","composition","presentation"],"additionalProperties":False}
        return {"type":"object","properties":{"items":{"type":"array","items":item,"minItems":count,"maxItems":count}},"required":["items"],"additionalProperties":False}

    def teacher_prompt(self,count):
        examples=self.example_items()
        return f"""Generate an ordered curriculum of {count} training problems using the supplied constructors.
Return one JSON object with an items array. Each item contains family, parameters, composition,
presentation. Choose the problem content, values and order to improve student learning.
Feedback comes from student progress after the complete curriculum. No diagnostic labels are exposed.
family: START, APPEND, EXACT, REGEX, COMPR, HAS. composition: []. presentation: default.
parameters.color_mode: two_color (R,B) or four_color (R,B,Y,G). COMPR requires two_color.
START/APPEND/EXACT/HAS: sequence with length in the fixed generator range.
COMPR: threshold in {self.cfg.NUMERICAL_THRESHOLDS}, operator >,>=,<,<=,=.
REGEX: {self.cfg.REGEX_CONCATENATION_COUNT} pattern_parts, each [word,operator], word length
{self.cfg.REGEX_MAX_PATTERN_LENGTH}, operator from {self.cfg.REGEX_OPERATORS}.
Examples:\n"""+"\n".join(json.dumps(x,separators=(",",":")) for x in examples)

    @staticmethod
    def example_items():
        return [{"family":f,"parameters":({"threshold":7,"operator":">"} if f=="COMPR" else
            {"pattern_parts":[["RB","+"],["B","*"]]} if f=="REGEX" else {"sequence":"RB"}),
            "composition":[],"presentation":"default"} for f in FAMILIES]

    def fixed_calibration_curriculum(self,count,seed):
        rng=random.Random(seed);items=self.example_items();out=[]
        for i in range(count):
            item=json.loads(json.dumps(items[i%len(items)]))
            if "sequence" in item["parameters"]:item["parameters"]["sequence"]="".join(rng.choices("RB",k=3))
            out.append(self.construct(item))
        return out

    def load_released_tasks(self,records):
        out=[]
        for row in records:
            formatted=row if "messages" in row else self.wrapper.convert_problem(row)
            if not formatted.get("id") or not formatted.get("ground_truth") or not formatted.get("messages"):
                raise ValueError("Released Manufactoria record lacks identity, prompt or tests")
            out.append(FactoryTask(formatted["messages"][0]["content"],formatted["ground_truth"],self.parser,str(formatted["id"])))
        return out


def make_dfa_pool(split,specifications=128,seed=2026):
    from automata.fa.nfa import NFA
    from automata.fa.dfa import DFA
    result=[];used=set()
    for family in ["HAS","REGEX"]:
        rng=random.Random(seed_for(seed,split,family));index=0
        for attempt in range(100000):
            word="".join(rng.choices("RB",k=rng.randint(3,9)))
            if family=="HAS": expression=f"(R|B)*{word}(R|B)*"
            else:
                other="".join(rng.choices("RB",k=rng.randint(1,3)))
                expression=f"({word})+({other})*"
            if expression in used:continue
            bucket=seed_for(0,"diagnostic_specification_partition",expression)%10
            assigned="reward" if bucket<7 else "monitor" if bucket==7 else "calibration"
            if assigned!=split:continue
            dfa=DFA.from_nfa(NFA.from_regex(expression,input_symbols=set("RB"))).to_complete().minify()
            order={dfa.initial_state:0};queue=deque([dfa.initial_state]);transitions={}
            while queue:
                state=queue.popleft();row={}
                for c in "RB":
                    dest=dfa.transitions[state][c]
                    if dest not in order:order[dest]=len(order);queue.append(dest)
                    row[c]=order[dest]
                transitions[str(order[state])]=row
            accepting=sorted(order[s] for s in dfa.final_states)
            def advance(state,text):
                for c in text:state=transitions[str(state)][c]
                return state
            prefixes=defaultdict(list)
            for length in range(15):
                for chars in itertools.product("RB",repeat=length):
                    prefix="".join(chars);state=advance(0,prefix)
                    if len(prefixes[state])<4:prefixes[state].append(prefix)
                eligible=[s for s in sorted(prefixes) if len(prefixes[s])==4]
                if len(eligible)>=4:break
            if len(eligible)<4:continue
            used.add(expression)
            for state in eligible[:4]:
                suffixes={False:[],True:[]};visits=defaultdict(int);queue=deque([(state,"")])
                while queue:
                    endpoint,suffix=queue.popleft();yes=endpoint in accepting
                    if len(suffixes[yes])<2:suffixes[yes].append(suffix)
                    if all(len(v)==2 for v in suffixes.values()):break
                    if visits[endpoint]>=4:continue
                    visits[endpoint]+=1
                    for c in "RB":queue.append((transitions[str(endpoint)][c],suffix+c))
                choices=[x for pair in zip(suffixes[False],suffixes[True]) for x in pair]
                if len(choices)<4:choices=(suffixes[False]+suffixes[True])*4
                for context,prefix in enumerate(prefixes[state]):
                    suffix=choices[context];answer=int(advance(state,suffix) in accepting)
                    prefix_text=f"Regular language: {expression}. Prefix tape: \"{prefix}\""
                    continuation={"kind":"dfa_suffix","suffix":suffix,"transitions":transitions,"accepting":accepting}
                    result.append({"id":f"{split}/{family}/{index}/{state}/{context}","split":split,"family":family,
                        "specification":f"{split}/{family}/{index}","state":str(state),"context":str(context),
                        "prefix":prefix_text,"prompt":prefix_text+BOUNDARY+f'Append suffix "{suffix}". Is the whole tape accepted? Return \\boxed{{1}} for yes and \\boxed{{0}} for no.',
                        "reference_state":str(state),"continuation":continuation,"answer":str(answer)})
            index+=1
            if index==specifications:break
        if index!=specifications:raise RuntimeError("Could not generate enough eligible acceptor specifications")
    return result

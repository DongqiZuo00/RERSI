import ast
from dataclasses import dataclass
import json
import math
import re
import sympy as sp

Z = sp.Symbol("z")
PRIMES = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31}


def final_box(text):
    matches = list(re.finditer(r"\\boxed(?![A-Za-z])\s*", text))
    if not matches:
        return None
    start = matches[-1].end()
    if start >= len(text):
        return None
    if text[start] != "{":
        token = re.match(r"\\[A-Za-z]+|\\[^A-Za-z]|[^\s$}\\]", text[start:])
        return token.group() if token else None
    start += 1
    depth = 1
    escaped = False
    for i in range(start, len(text)):
        if escaped:
            escaped = False
            continue
        if text[i] == "\\":
            escaped = True
            continue
        depth += (text[i] == "{") - (text[i] == "}")
        if depth == 0:
            return text[start:i].strip()
    return None


def _latex_fraction(text):
    text = text.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    while "\\frac" in text:
        p = text.index("\\frac")
        pos, terms = p + 5, []
        for _ in range(2):
            if pos >= len(text) or text[pos] != "{":
                raise ValueError("Use braced fractions")
            begin, depth = pos + 1, 1
            pos += 1
            while pos < len(text) and depth:
                depth += (text[pos] == "{") - (text[pos] == "}")
                pos += 1
            if depth:
                raise ValueError("Unclosed fraction")
            terms.append(_latex_fraction(text[begin:pos-1]))
        text = text[:p] + f"(({terms[0]})/({terms[1]}))" + text[pos:]
    return text


def safe_expression(text):
    if len(text) > 2048:
        raise ValueError("Answer too long")
    text = _latex_fraction(text)
    for a, b in [("\\left", ""), ("\\right", ""), ("\\cdot", "*"),
                 ("\\times", "*"), ("^", "**"), ("{", "("), ("}", ")")]:
        text = text.replace(a, b)
    text = re.sub(r"(?<=\d)z", "*z", text.strip())
    tree = ast.parse(text, mode="eval")
    if sum(1 for _ in ast.walk(tree)) > 256:
        raise ValueError("Answer too complex")

    def visit(n):
        if isinstance(n, ast.Constant) and type(n.value) in (int, float):
            if (type(n.value) is int and n.value.bit_length()>1024) or (type(n.value) is float and not math.isfinite(n.value)):
                raise ValueError("Answer magnitude exceeded")
            return sp.Rational(str(n.value))
        if isinstance(n, ast.Name) and n.id == "z":
            return Z
        if isinstance(n, (ast.Tuple, ast.List)):
            return tuple(visit(x) for x in n.elts)
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.UAdd, ast.USub)):
            x = visit(n.operand)
            return x if isinstance(n.op, ast.UAdd) else -x
        if isinstance(n, ast.BinOp):
            x, y = visit(n.left), visit(n.right)
            if isinstance(n.op, ast.Add): return x + y
            if isinstance(n.op, ast.Sub): return x - y
            if isinstance(n.op, ast.Mult): return x * y
            if isinstance(n.op, ast.Div) and y != 0: return x / y
            if isinstance(n.op, ast.Pow) and y.is_Integer and abs(int(y)) <= 16:
                if x==0 and y<0:raise ValueError("Division by zero")
                return bounded(x**y)
        raise ValueError("Unsupported answer syntax")
    def bounded(value):
        if isinstance(value,tuple):return tuple(bounded(v) for v in value)
        if not isinstance(value,sp.Expr):raise ValueError("Expected an exact expression")
        if value.has(sp.zoo,sp.oo,-sp.oo,sp.nan):raise ValueError("Non-finite expression")
        if value.free_symbols:
            if sp.degree(value,Z)>16:raise ValueError("Answer polynomial degree exceeded")
            coefficients=sp.Poly(value,Z).all_coeffs()
        else:coefficients=[value]
        for coefficient in coefficients:
            if not coefficient.is_Rational:raise ValueError("Expected rational polynomial coefficients")
            if int(coefficient.p).bit_length()>4096 or int(coefficient.q).bit_length()>4096:
                raise ValueError("Exact answer size exceeded")
        return value
    try:return bounded(visit(tree.body))
    except sp.PolynomialError as exc:raise ValueError("Expected a polynomial answer") from exc


def equivalent(x, y):
    if isinstance(x, (tuple, list)) or isinstance(y, (tuple, list)):
        return isinstance(x, (tuple, list)) and isinstance(y, (tuple, list)) and len(x) == len(y) and all(equivalent(a,b) for a,b in zip(x,y))
    return sp.expand(x - y) == 0


@dataclass
class Task:
    prompt: str
    answer: str
    family: str
    identity: str = ""

    def verify(self, response):
        answer = final_box(response)
        if answer is None:
            return 0.0
        try:
            return float(equivalent(safe_expression(answer), safe_expression(self.answer)))
        except (ValueError, TypeError, SyntaxError, ZeroDivisionError, OverflowError, AttributeError, RecursionError):
            return 0.0


def _integer(x, lo, hi):
    if type(x) is not int or not lo <= x <= hi:
        raise ValueError(f"Expected integer in [{lo},{hi}]")
    return x


def parameter(value, family):
    if type(value) is int:
        return sp.Integer(_integer(value, -20, 20)), "scalar"
    if isinstance(value, list) and len(value) == 2:
        return sp.Rational(_integer(value[0], -20, 20), _integer(value[1], 1, 20)), "scalar"
    if isinstance(value, dict) and set(value) == {"poly"} and family == "polynomial":
        c = value["poly"]
        if not isinstance(c, list) or not 1 <= len(c) <= 5:
            raise ValueError("Polynomial degree must be <=4")
        return sum(_integer(v, -9, 9) * Z**i for i,v in enumerate(c)), "poly"
    if isinstance(value, dict) and set(value) == {"matrix"} and family == "linear":
        rows = value["matrix"]
        n = len(rows)
        if n not in (2,3) or any(len(row) != n+1 for row in rows):
            raise ValueError("Expected augmented 2x3 or 3x4 matrix")
        return sp.Matrix([[_integer(x,-9,9) for x in row] for row in rows]), "matrix"
    if isinstance(value, dict) and set(value) == {"vector"} and family == "linear":
        vals = value["vector"]
        if len(vals) not in (2,3): raise ValueError("Linear form needs 2 or 3 coefficients")
        return tuple(sp.Integer(_integer(x,-9,9)) for x in vals), "vector"
    raise ValueError("Invalid parameter type")


def _text(value):
    if isinstance(value, sp.MatrixBase):
        return str(value.tolist())
    if isinstance(value, tuple):
        return "(" + ", ".join(map(str,value)) + ")"
    return str(value)


def construct(item):
    if set(item) != {"family", "parameters", "composition", "presentation"}:
        raise ValueError("Unexpected or missing item fields")
    family = item["family"]
    if family not in {"rational", "polynomial", "modular", "linear"}:
        raise ValueError("Unknown mathematical family")
    if item["presentation"] not in {"symbolic", "verbal"}:
        raise ValueError("Unknown presentation")
    params = item["parameters"]
    if not isinstance(params,dict) or not 1 <= len(params) <= 7:
        raise ValueError("Invalid parameters")
    modulus = params.get("modulus")
    if family == "modular" and modulus not in PRIMES:
        raise ValueError("Prime modulus required")
    env, desc = {}, {}
    for name, value in params.items():
        if name == "modulus" and family == "modular": continue
        if not re.fullmatch(r"[a-f]", name): raise ValueError("Parameter names are a..f")
        env[name] = parameter(value, family)
        desc[name] = _text(env[name][0])
    nodes = item["composition"]
    if not isinstance(nodes,list) or not 1 <= len(nodes) <= 4:
        raise ValueError("A graph has 1..4 operations")
    for i, node in enumerate(nodes):
        if set(node) != {"id","op","args"} or node["id"] != f"n{i}":
            raise ValueError("Nodes must be ordered n0..n3")
        args = node["args"]
        if not isinstance(args,list) or any(a not in env for a in args):
            raise ValueError("Unknown or forward reference")
        values = [env[a][0] for a in args]
        types = [env[a][1] for a in args]
        ds = [desc[a] for a in args]
        op, outtype = node["op"], "scalar"
        if op in {"add","sub","mul","div"}:
            if len(values)!=2: raise ValueError("Binary operation requires two arguments")
            if family == "modular":
                if types != ["scalar","scalar"] or op=="div" or not all(x.is_Integer for x in values): raise ValueError("Modular type error")
            elif family == "rational":
                if types != ["scalar","scalar"]: raise ValueError("Rational type error")
            elif family == "polynomial":
                if any(t not in {"poly","scalar"} for t in types) or op=="div": raise ValueError("Polynomial type error")
                outtype = "poly" if "poly" in types else "scalar"
            else: raise ValueError("Unsupported linear-system operation")
            x,y = values
            if op=="div" and y==0: raise ValueError("Division by zero")
            out = {"add":lambda:x+y,"sub":lambda:x-y,"mul":lambda:x*y,"div":lambda:x/y}[op]()
            symbol = {"add":"+","sub":"-","mul":"*","div":"/"}[op]
            expression = f"({ds[0]}) {symbol} ({ds[1]})"
        elif op=="diff" and family=="polynomial" and types==["poly"]:
            out, outtype, expression = sp.diff(values[0],Z), "poly", f"differentiate ({ds[0]}) with respect to z"
        elif op=="eval" and family=="polynomial" and types==["poly","scalar"]:
            out, expression = values[0].subs(Z,values[1]), f"evaluate ({ds[0]}) at z={ds[1]}"
        elif op=="pow" and family=="modular" and types==["scalar","scalar"]:
            x,y = values
            if not x.is_Integer or not y.is_Integer or not 0<=y<=8: raise ValueError("Invalid modular exponent")
            out, expression = sp.Integer(pow(int(x),int(y),modulus)), f"({ds[0]})**({ds[1]})"
        elif family=="linear":
            if op in {"swap","row_add","scale"}:
                expected = {"swap":["matrix","scalar","scalar"],"row_add":["matrix","scalar","scalar","scalar"],"scale":["matrix","scalar","scalar"]}[op]
                if types != expected: raise ValueError("Row-operation type mismatch")
                out = values[0].copy()
                row = values[1]
                if not row.is_Integer or not 0<=row<out.rows: raise ValueError("Invalid row")
                row = int(row)
                if op in {"swap","row_add"}:
                    col = values[2]
                    if not col.is_Integer or not 0<=col<out.rows: raise ValueError("Invalid row")
                    col=int(col)
                    if row==col: raise ValueError("Distinct row indices required")
                    if op=="swap": out.row_swap(row,col)
                    else: out[row,:] = out[row,:] + values[3]*out[col,:]
                else:
                    if values[2]==0: raise ValueError("Nonzero row scale required")
                    out[row,:] = values[2]*out[row,:]
                outtype,expression = "matrix", f"{op}({', '.join(ds)})"
            elif op=="solve" and types==["matrix"]:
                matrix=values[0]
                if matrix[:,:-1].det()==0: raise ValueError("Unique solution required")
                out=tuple(matrix[:,:-1].inv()*matrix[:,-1])
                outtype,expression="solution",f"solve augmented system {ds[0]} (variable order x0,x1[,x2])"
            elif op=="linform" and types==["solution","vector"]:
                if len(values[0])!=len(values[1]): raise ValueError("Linear form dimension mismatch")
                out=sum(a*b for a,b in zip(*values))
                expression=f"dot({ds[0]}, {ds[1]})"
            else: raise ValueError("Unknown linear operation or wrong argument types")
        else: raise ValueError("Unknown operation or wrong argument types")
        if family=="modular": out=sp.Integer(int(out)%modulus)
        if outtype=="poly":
            out=sp.expand(out)
            if sp.degree(out,Z)>8: raise ValueError("Polynomial degree exceeded")
        if len(_text(out))>8192: raise ValueError("Exact expression size exceeded")
        env[node["id"]]=(out,outtype)
        desc[node["id"]]=expression
    if outtype=="matrix":
        raise ValueError("Final linear node must solve or evaluate a linear form")
    expression=desc[nodes[-1]["id"]]
    if family=="modular": expression+=f" modulo {modulus} (least nonnegative residue)"
    lead="Compute: " if item["presentation"]=="symbolic" else "Carry out the following exact computation: "
    prompt=lead+expression+". Return the final answer in \\boxed{...}. Use fractions, expressions in z, or an ordered tuple as appropriate."
    if family=="linear":
        prompt+=" Row indices start at 0; row_add(A,i,j,c) replaces row i by row i+c*row j; scale(A,i,c) scales row i; swap exchanges two rows."
    return Task(prompt,_text(out),family,json.dumps(item,sort_keys=True,separators=(",",":")))


def build_curriculum(text, count):
    obj=json.loads(text)
    if set(obj)!={"items"} or not isinstance(obj["items"],list) or len(obj["items"])!=count:
        raise ValueError(f"Expected exactly {count} curriculum items")
    return [construct(item) for item in obj["items"]]

import json

EXAMPLES = [
 {"family":"rational","parameters":{"a":[1,2],"b":[1,3],"c":[6,1]},"composition":[{"id":"n0","op":"add","args":["a","b"]},{"id":"n1","op":"mul","args":["n0","c"]}],"presentation":"symbolic"},
 {"family":"polynomial","parameters":{"a":{"poly":[1,2,1]}},"composition":[{"id":"n0","op":"diff","args":["a"]}],"presentation":"symbolic"},
 {"family":"modular","parameters":{"a":5,"b":3,"modulus":7},"composition":[{"id":"n0","op":"pow","args":["a","b"]}],"presentation":"symbolic"},
 {"family":"linear","parameters":{"a":{"matrix":[[1,2,3],[0,1,1]]}},"composition":[{"id":"n0","op":"solve","args":["a"]}],"presentation":"symbolic"}
]


def curriculum_schema(count):
    integer={"type":"integer"}
    array={"type":"array","items":integer,"minItems":1,"maxItems":5}
    def obj(properties,required=None):
        return {"type":"object","properties":properties,"required":list(properties) if required is None else required,"additionalProperties":False}
    parameter={"anyOf":[integer,{"type":"array","items":integer,"minItems":2,"maxItems":2},
      obj({"poly":array}),obj({"vector":array}),
      obj({"matrix":{"type":"array","items":array,"minItems":2,"maxItems":3}})]}
    properties={k:parameter for k in "abcdef"}; properties["modulus"]=integer
    node=obj({"id":{"type":"string","enum":[f"n{i}" for i in range(4)]},
      "op":{"type":"string","enum":["add","sub","mul","div","diff","eval","pow","swap","row_add","scale","solve","linform"]},
      "args":{"type":"array","minItems":1,"maxItems":4,"items":{"type":"string","enum":list("abcdef")+[f"n{i}" for i in range(4)]}}})
    item=obj({"family":{"type":"string","enum":["rational","polynomial","modular","linear"]},
      "parameters":obj(properties,[]),"composition":{"type":"array","items":node,"minItems":1,"maxItems":4},
      "presentation":{"type":"string","enum":["symbolic","verbal"]}})
    return obj({"items":{"type":"array","items":item,"minItems":count,"maxItems":count}})


def teacher_prompt(count):
    return f"""Generate an ordered curriculum of {count} training problems using the supplied constructors.
Return one JSON object with an items array. For every item, provide its family, parameters,
composition, and presentation fields using the listed schemas. Choose the problem content,
parameter values, composition, and order to improve the student's learning. The task
interface constructs each problem and verifies student answers. Your curriculum receives
feedback from the student's progress after training on the complete curriculum.
Schema: exactly {count} items, no extra keys. family: rational|polynomial|modular|linear.
parameters: names a..f; scalars integers [-20,20] or [numerator,denominator] with numerator
[-20,20] and denominator [1,20]; polynomials {{"poly":[coefficients low to high]}}
have degree <=4 and coefficients [-9,9]; linear {{"matrix":[augmented rows]}} is 2x3 or
3x4 with integer entries [-9,9], {{"vector":[coefficients]}} has length 2 or 3.
modular requires parameters.modulus prime in [2,31].
composition: 1..4 nodes {{id,op,args}}, ids n0,n1,... in order; args refer to parameters or
earlier nodes. Last node is output. rational: add/sub/mul/div (nonzero divisor).
polynomial: add/sub/mul/diff/eval(poly,scalar), resulting degree <=8.
modular: add/sub/mul/pow; integer operands, exponent 0..8, reduce after every node.
linear: swap(matrix,row,row), row_add(matrix,destination,source,multiplier),
scale(matrix,row,nonzero_scalar), solve(full_rank_matrix), linform(solution,vector).
Row indices are zero-based scalar parameters. Final linear output is solution or linform.
presentation: symbolic|verbal. Examples, one per constructor:
"""+"\n".join(json.dumps(x,separators=(",",":")) for x in EXAMPLES)

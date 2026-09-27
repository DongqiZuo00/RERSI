# Validation record

This record describes executed software checks. It is not a benchmark result.

## Executed

- `python -m unittest discover -s tests -v`: **10 tests passed**.
- CPU integration run: two recursive rounds, three candidates per round, two paired student trials per candidate, real tiny causal Transformer training and teacher updates.
- State-balanced cohesion, scale invariance, population-standard-deviation reward normalization, invalid penalties, tie-breaking and pass@k arithmetic.
- Complete model/optimizer restoration; BF16 model weights backed by persistent FP32 master weights and FP32 AdamW moments.
- Common student origin and optimizer state across trials; paired seeds across candidates; selection of the first replicate; all-invalid groups skip the teacher update and retain the student.
- Hugging Face generation on a randomly initialized GPT-2-sized test model with actual JSON grammar constraints; replayed token masks; finite policy gradients.
- Production four-family curriculum schema validated by replaying all four constructor examples through the grammar.
- Full mathematical diagnostic generation: 4096 reward / 1024 monitoring / 256 calibration examples, with duplicate and split-overlap checks.
- Full Manufactoria DFA diagnostic generation: 4096 / 1024 / 256 examples, with duplicate and split-overlap checks.
- Semantic patching calibration executed on two tiny-model checkpoints and both decoder blocks.
- Six official DELTA family constructors exercised; a valid append program received full-pass 1, an incorrect program received 0, and malformed programs received 0.
- Official pinned HARP checker exercised with equivalent fractions, incorrect answers and missing boxed answers.
- Original 4780-item HARP release partitioned into 3442 / 382 / 956 items, preserving all 4780 unique identities.
- The specified Gemma checkpoint's configuration was retrieved at the paper revision and reports `gemma4` with 42 decoder layers.

## Local test versions

Python 3.12; PyTorch 2.14.0+cpu; Transformers 5.17.0; NumPy 2.5.3;
SymPy 1.14.0; LM Format Enforcer 0.11.3; automata-lib 9.2.0.

The LM Format Enforcer Transformers helper is incompatible with a moved tokenizer
import in Transformers 5. This package uses its library-neutral TokenEnforcer API;
the constrained generation/replay test exercises that bridge.

## Not executed or not supplied

- No Gemma weight download, GPU training, 100-round run, five-seed benchmark sweep,
  or measurement of Gemma peak GPU/host memory.
- No reproduction of the paper's reported accuracy, compute ratios, uncertainty
  intervals or mechanistic figures.
- The original diagnostic manifests, calibration trajectories, full run artifacts,
  exact HARP seasonal-year tie ordering and original DELTA commit were not supplied.

## Package anonymity

The distributed archive contains source, configuration, tests and documentation.
It excludes personal identifiers, institution/author metadata, credentials, private
paths, Git history, the input manuscript, dependencies, caches, checkpoints and
execution logs. External public model and benchmark references remain explicit.

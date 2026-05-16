# Examples

| Example                   | What it demonstrates                                    | Writes artifact? |
| ------------------------- | ------------------------------------------------------- | ---------------- |
| `run_two_clients.py`      | One-round federated quantum execution                   | No               |
| `run_multi_round.py`      | Multi-round coordination                                | Yes              |
| `run_parameter_update.py` | Heuristic parameter update + loss tracking              | Yes              |
| `run_gradient_update.py`  | Finite-difference gradient update                       | Yes              |
| `run_from_manifest.py`    | Runs a supported experiment from a JSON manifest        | Yes              |

## Run

```bash
python examples/run_two_clients.py
python examples/run_multi_round.py
python examples/run_parameter_update.py
python examples/run_gradient_update.py
python examples/run_from_manifest.py examples/manifests/gradient_update.json
```

Artifact-producing examples write timestamped JSON files under `runs/`.

## What each example does

**`run_two_clients.py`** — the smallest demo. Two clients run one coordination round and the result is printed. No artifact is saved.

**`run_multi_round.py`** — runs three coordination rounds and saves a JSON artifact. The artifact includes environment metadata and the per-round aggregated results.

**`run_parameter_update.py`** — runs a heuristic parameter update loop. Each round applies the rule `next_theta = theta - learning_rate * aggregated_result` and tracks a simple squared loss. Saves a JSON artifact.

**`run_gradient_update.py`** — runs a finite-difference gradient update loop. Each round estimates a gradient using central finite differences and updates `next_theta = theta - learning_rate * gradient`. Saves a JSON artifact.

**`run_from_manifest.py`** — loads a JSON manifest file, validates it, and runs the specified experiment. Currently only `gradient_update` manifests are supported. The manifest defines clients, rounds, initial theta, learning rate, target, and epsilon. Saves a JSON artifact that includes both the normalized manifest config and the run result.

```bash
python examples/run_from_manifest.py examples/manifests/gradient_update.json
```

The examples form a progression from the simplest possible execution toward a minimal gradient-based optimization trace. They are demos, not a training framework.

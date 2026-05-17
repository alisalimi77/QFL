# Examples

| Example                          | What it demonstrates                                                  | Writes artifact? |
| -------------------------------- | --------------------------------------------------------------------- | ---------------- |
| `run_two_clients.py`             | One-round federated quantum execution                                 | No               |
| `run_multi_round.py`             | Multi-round coordination                                              | Yes              |
| `run_parameter_update.py`        | Heuristic parameter update + loss tracking                            | Yes              |
| `run_gradient_update.py`         | Finite-difference gradient update                                     | Yes              |
| `run_from_manifest.py`           | Runs a supported experiment from a JSON manifest                      | Yes              |
| `compare_artifacts.py`           | Compares saved JSON artifacts from multiple runs                      | No               |
| `run_custom_backend.py`          | Backend injection with a deterministic `ConstantBackend`              | No               |
| `run_clean_vs_noisy_backend.py`  | Clean vs. deterministic noisy backend comparison                      | Yes              |

For a guided sequence through these examples, see [../docs/walkthrough.md](../docs/walkthrough.md).

## Run

```bash
python examples/run_two_clients.py
python examples/run_multi_round.py
python examples/run_parameter_update.py
python examples/run_gradient_update.py
python examples/run_from_manifest.py examples/manifests/gradient_update.json
python examples/run_custom_backend.py
python examples/run_clean_vs_noisy_backend.py
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

**`run_custom_backend.py`** — demonstrates backend injection without artifacts. Two clients use `ConstantBackend` with fixed values (0.2 and 0.6). The coordinator aggregates them to 0.4. No artifact is saved.

```bash
python examples/run_custom_backend.py
```

**`run_clean_vs_noisy_backend.py`** — runs two coordinated rounds side by side: one with `PennyLaneBackend` (clean) and one with `NoisyBackend` wrapping `PennyLaneBackend`. Prints both aggregated results and their difference. Saves a JSON artifact.

```bash
python examples/run_clean_vs_noisy_backend.py
```

The examples form a progression from the simplest possible execution toward a minimal gradient-based optimization trace. They are demos, not a training framework.

## Compare artifacts

First generate some artifacts:

```bash
python examples/run_from_manifest.py examples/manifests/gradient_update.json
python examples/run_from_manifest.py examples/manifests/gradient_update_more_rounds.json
```

Then compare them:

```bash
python examples/compare_artifacts.py runs/<artifact1>.json runs/<artifact2>.json
```

`compare_artifacts.py` is dependency-free. It prints run_id, manifest name, manifest file, backend name, experiment, rounds, final_theta, and final_loss for each artifact in a plain text table. It is not a dashboard or experiment tracking system.

## Manifest examples

| Manifest                             | Name                      | Purpose                                      |
| ------------------------------------ | ------------------------- | -------------------------------------------- |
| `gradient_update.json`               | `default-gradient-update` | Default finite-difference gradient update    |
| `gradient_update_low_lr.json`        | `low-learning-rate`       | Same experiment with a smaller learning rate |
| `gradient_update_target_half.json`   | `target-half`             | Same experiment with a non-zero target       |
| `gradient_update_more_rounds.json`   | `more-rounds`             | Same experiment with more update rounds      |

All manifests use `"manifest_version": "0.1"` and `"experiment": "gradient_update"`. Artifact-producing runs save timestamped JSON artifacts under `runs/`.

```bash
python examples/run_from_manifest.py examples/manifests/gradient_update.json
python examples/run_from_manifest.py examples/manifests/gradient_update_low_lr.json
python examples/run_from_manifest.py examples/manifests/gradient_update_target_half.json
python examples/run_from_manifest.py examples/manifests/gradient_update_more_rounds.json
```

# Examples

`run_two_clients.py` is the smallest one-round demo. It creates two local quantum clients, runs one coordination round, and prints a readable report.

`run_multi_round.py` runs repeated coordination rounds and saves a timestamped JSON artifact under `runs/`.

`run_parameter_update.py` demonstrates a minimal repeated parameter update loop with simple objective/loss tracking and a reproducibility artifact.

`run_gradient_update.py` demonstrates a minimal finite-difference gradient update loop with objective/loss tracking and a reproducibility artifact.

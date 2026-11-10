# Contributing

Use Python 3.12 and install the project in a virtual environment with
`python -m pip install -e .`. Run `python -m unittest discover -s tests -v`
before submitting a change. The tests do not require trained models or thesis
outputs.

For simulation or controller changes, include a regression test and state which
settings changed. Keep baseline and RL comparisons matched in reference,
disturbance, and physical constraints. Do not commit generated runs, weights,
machine-local paths, private data, or third-party datasheets. Never load
checkpoints from an untrusted source.

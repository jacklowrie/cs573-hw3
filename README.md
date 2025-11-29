# cs573 Homework 3

## Structure
```
.
├── data
│   └── datahw3.npz
├── main.py
├── notebook.ipynb
├── pyproject.toml
├── README.md
├── src
│   ├── __init__.py
│   ├── bagging.py
│   ├── boosting.py
│   ├── runner.py
│   └── utils.py
└── uv.lock
```
Implementation is in `src`. `main.py` contains a way to run from the commandline, 
while `notebook.ipynb` contains some visualization and a record of one run on my machine.

`bagging`/`boosting` contains ensemble classifiers, while `runner` is a wrapper to run both.

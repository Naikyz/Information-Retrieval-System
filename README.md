# Experimental IR System for NLP Class

### Init dev env

You can create a virtual env with python 3.9 using `conda` or `venv`

```
conda create -n ir-system python=3.9
conda activate ir-system
pip install -r requirements.txt
```

### Run local server

Run `python main.py`

options:
- `--boolean "QUERY"`: Executes a boolean query. example: `python main.py --boolean "Hello AND document"`
- `--vector "QUERY"`: Executes a vector space model query. example: `python main.py --vector "hello"`

⚠️ ONE OF BOTH OPTIONS IS MANDATORY ⚠️
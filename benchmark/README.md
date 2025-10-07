This is the source code of our benchmark.

`evaluation/` contains the evaluation/parsing trigger code of all statistic-based log parsers.

`logparsers/` contains the implementation of all statistic-based log parsers.

`old_benchmark/` contains some old code of previous benchamrking.

`LogPPT/` contains the implementation of LogPPT.

`UniParser` contains the implementaion of UniParser.

## HybridParser evaluation (using datasets/full)

We added `evaluation/Hybrid_eval.py` to run the Loghub benchmark against the in-repo HybridParser.

- Input datasets: `../../../../datasets/{2k|full}/<Dataset>/<Dataset>_{2k|full}.log_structured.csv`
- Output directory: `../result/result_Hybrid_{2k|full}`

How to run (examples):

1) Single dataset smoke test (Apache, full):

	- Make sure you have at least one trained checkpoint folder like `results/**/fold_*_Apache` containing `model.pt` and `config.json`.
	- Optionally set `HP_CHECKPOINT_DIR` to override auto-discovery.

	PowerShell:

	```powershell
	$env:HP_DATASET = 'Apache'
	# optional overrides
	# $env:HP_CHECKPOINT_DIR = 'C:\path\to\results\...\fold_0_Apache'
	# $env:HP_DEVICE = 'cpu'  # or 'cuda'
	# $env:HP_MIN_MATCH_PROB = '0.5'
	python -X utf8 evaluation/Hybrid_eval.py -full
	```

2) All datasets (full):

	```powershell
	python -X utf8 evaluation/Hybrid_eval.py -full
	```

Notes:
- The adapter auto-discovers the most recent `fold_*_<Dataset>` under `results/actual` first, then `results/experiments`.
- Results are aggregated into `../result/Hybrid_full_*.csv` via the shared postprocess step.
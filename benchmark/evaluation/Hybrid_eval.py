"""
Evaluate HybridParser within the Loghub 2.0 benchmark harness, aligned with Drain's evaluator.

Key behaviors:
- Uses datasets from the repo's datasets/{2k|full} folders (not loghub-2.0/*_dataset).
- Conforms to the LogParser interface expected by evaluator_main.evaluator.
- Automatically selects a trained model checkpoint matching the dataset name
  by searching under results/actual first, then results/experiments. You can override
  via env HP_CHECKPOINT_DIR.

Outputs:
- Writes <output_dir>/<Dataset>_<type>.log_structured.csv (EventId, EventTemplate produced by HybridParser)
- Appends scores to <output_dir>/summary_....csv via the shared evaluator utils.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure we can import evaluation.* regardless of CWD
_BENCH_ROOT = Path(__file__).resolve().parents[1]
if str(_BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_BENCH_ROOT))
from evaluation.utils.common import common_args
from evaluation.utils.evaluator_main import evaluator, prepare_results
from evaluation.utils.postprocess import post_average
from old_benchmark.Drain_benchmark import benchmark_settings
from old_benchmark.Hybrid_benchmark import hybrid_settings
from logparser.Hybrid import LogParser


DATASETS_2K = [
    "Proxifier",
    "Linux",
    "Apache",
    "Zookeeper",
    "Hadoop",
    "HealthApp",
    "OpenStack",
    "HPC",
    "Mac",
    "OpenSSH",
    "Spark",
    "Thunderbird",
    "BGL",
    "HDFS",
]

DATASETS_FULL = DATASETS_2K.copy()


CURR = Path(__file__).resolve()


if __name__ == "__main__":
    args = common_args()
    data_type = "full" if args.full_data else "2k"

    # Resolve base locations (two candidates)
    bench_root = Path(__file__).resolve().parents[1]  # .../benchmark
    lh_root = bench_root.parent  # .../loghub-2.0
    lh_input_base = lh_root / f"{data_type}_dataset"
    repo_datasets = lh_root.parent  # .../datasets
    repo_input_base = repo_datasets / ("full" if data_type == "full" else "2k")

    output_dir = str(bench_root / f"result/result_Hybrid_{data_type}")
    os.makedirs(output_dir, exist_ok=True)

    result_file = prepare_results(
        output_dir=str(output_dir),
        otc=args.oracle_template_correction,
        complex=args.complex,
        frequent=args.frequent,
    )

    datasets = DATASETS_FULL if args.full_data else DATASETS_2K
    # Optional filter via env var to run a single dataset (smoke test)
    only = os.environ.get("HP_DATASET", "").strip()
    if only:
        datasets = [d for d in datasets if d.lower() == only.lower()]
        if not datasets:
            raise SystemExit(f"HP_DATASET={only} did not match any known dataset.")

    for dataset in datasets:
        setting = benchmark_settings[dataset]
        log_file = setting['log_file'].replace("_2k", f"_{data_type}")

        # Decide input_dir per dataset by checking which base contains the oracle CSV
        log_base = os.path.dirname(log_file)
        log_name = os.path.basename(log_file)
        candidate1 = lh_input_base / log_base / f"{log_name}_structured.csv"
        candidate2 = repo_input_base / log_base / f"{log_name}_structured.csv"
        if candidate1.exists():
            input_dir_ds = str(lh_input_base)
        elif candidate2.exists():
            input_dir_ds = str(repo_input_base)
        else:
            # fallback to lh even if missing; evaluator will report no output
            input_dir_ds = str(lh_input_base if lh_input_base.exists() else repo_input_base)
        indir = os.path.join(input_dir_ds, log_base)

        expected_out = os.path.join(output_dir, f"{dataset}_{data_type}.log_structured.csv")
        if os.path.exists(expected_out):
            parser = None
            print("parsing result exist.")
        else:
            parser = LogParser

        # Run evaluator for one dataset
        print(setting['log_format'])
        # Merge Drain settings with Hybrid-specific hardcoded settings
        hset = hybrid_settings.get(dataset, {})
        evaluator(
            dataset=dataset,
            input_dir=input_dir_ds,
            output_dir=output_dir,
            log_file=log_file,
            LogParser=parser,
            param_dict={
                'log_format': setting['log_format'],
                'indir': indir,
                'outdir': output_dir,
                'rex': setting['regex'],
                'depth': setting['depth'],
                'st': setting['st'],
                # Hybrid-specific extras (ignored by Drain, used by our adapter)
                'dataset': dataset,
                'checkpoint_dir': hset.get('checkpoint_dir') or os.environ.get('HP_CHECKPOINT_DIR', ''),
                'device': hset.get('device', os.environ.get('HP_DEVICE', 'cpu')),
                'min_match_prob': float(hset.get('min_match_prob', os.environ.get('HP_MIN_MATCH_PROB', '0.5'))),
            },
            otc=args.oracle_template_correction,
            complex=args.complex,
            frequent=args.frequent,
            result_file=result_file,
        )
    metric_file = os.path.join(output_dir, result_file)
    post_average(metric_file, f"Hybrid_{data_type}_complex={args.complex}_frequent={args.frequent}", args.complex, args.frequent)

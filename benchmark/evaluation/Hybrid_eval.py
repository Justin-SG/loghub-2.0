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
#    "Proxifier",
#    "Linux",
#    "Apache",
#    "Zookeeper",
#    "Hadoop",
#    "HealthApp",
#    "OpenStack",
#    "HPC",
#    "Mac",
#    "OpenSSH",
#    "Spark",
    "Thunderbird",
    "BGL",
    "HDFS",
]

DATASETS_FULL = DATASETS_2K.copy()


CURR = Path(__file__).resolve()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # Common args
    parser.add_argument('-otc', '--oracle_template_correction', help="Set this if you want to use corrected oracle templates", default=False, action='store_true')
    parser.add_argument('-full', '--full_data', help="Set this if you want to test on full dataset", default=False, action='store_true')
    parser.add_argument('--complex', type=int, help="Set this if you want to test on complex dataset", default=0)
    parser.add_argument('--frequent', type=int, help="Set this if you want to test on frequent dataset", default=0)
    parser.add_argument('--shot', type=int, help="Set this if you want to test on complex dataset", default=0)
    parser.add_argument('--example_size', type=int, help="Set this if you want to test on frequent dataset", default=0)    

    # Hybrid Specific Args
    parser.add_argument('--no_grouper', action='store_true', help='Disable log grouper (Cache only if no PI)')
    parser.add_argument('--use_param_identifier', action='store_true', help='Enable Parameter Identifier')
    import platform
    default_workers = 1 if platform.system() == "Windows" else 8
    parser.add_argument("--workers", type=int, default=default_workers, help="Number of processes to use")
    parser.add_argument('--run_name', type=str, default='', help='Suffix for output directory')
    parser.add_argument('--checkpoint_dir_grouper', type=str, default=None, help='Override grouper checkpoint dir')
    parser.add_argument('--checkpoint_dir_param', type=str, default=None, help='Override param identifier checkpoint dir')
    parser.add_argument('--device', type=str, default=None, help='Force device (cpu/cuda)')
    parser.add_argument('--preload_ground_truth', action='store_true', help='Preload ground truth templates into cache')

    args = parser.parse_args()

    data_type = "full" if args.full_data else "2k"

    # Resolve base locations (two candidates)
    bench_root = Path(__file__).resolve().parents[1]  # .../benchmark
    lh_root = bench_root.parent  # .../loghub-2.0
    lh_input_base = lh_root / f"{data_type}_dataset"
    repo_datasets = lh_root.parent  # .../datasets
    repo_input_base = repo_datasets / ("full" if data_type == "full" else "2k")

    # Output directory logic
    base_output = bench_root / f"result/result_Hybrid_{data_type}"
    if args.run_name:
        output_dir = str(base_output / args.run_name)
    else:
        output_dir = str(base_output)
    
    os.makedirs(output_dir, exist_ok=True)

    result_file = prepare_results(
        output_dir=str(output_dir),
        otc=args.oracle_template_correction,
        complex=args.complex,
        frequent=args.frequent,
    )

    datasets = DATASETS_FULL if args.full_data else DATASETS_2K

    # proj_root logic removed as per user request (strict config only)

    for dataset in datasets:
        setting = benchmark_settings[dataset]
        log_file = setting['log_file'].replace("_2k", f"_{data_type}")

        # Decide input_dir per dataset by checking which base contains the oracle CSV
        log_base = os.path.dirname(log_file)
        log_name = os.path.basename(log_file)
        candidate1 = lh_input_base / log_base / f"{log_name}_structured.csv"
        candidate2 = repo_input_base / log_base / f"{log_name}_structured.csv"
        candidate3 = lh_input_base / log_base / log_name # Check for the log file itself in lh
        candidate4 = repo_input_base / log_base / log_name # Check for the log file itself in repo

        if candidate1.exists():
            input_dir_ds = str(lh_input_base)
        elif candidate2.exists():
            input_dir_ds = str(repo_input_base)
        elif candidate3.exists():
             input_dir_ds = str(lh_input_base)
        elif candidate4.exists():
             input_dir_ds = str(repo_input_base)
        else:
            # fallback to lh even if missing; evaluator will report no output
            input_dir_ds = str(lh_input_base if lh_input_base.exists() else repo_input_base)
        indir = os.path.join(input_dir_ds, log_base)

        expected_out = os.path.join(output_dir, f"{dataset}_{data_type}.log_structured.csv")
        # FORCE RE-RUN: Removed check for existing file
        parser = LogParser

        # Run evaluator for one dataset
        print(f"Dataset: {dataset} | Format: {setting['log_format']}")
        
        # Merge Drain settings with Hybrid-specific settings
        hset = hybrid_settings.get(dataset, {})
        
        # Determine Configuration Priorities: CLI > Config > Default
        
        # 1. Use Grouper
        if args.no_grouper:
            use_grouper = False
        else:
            use_grouper = hset.get("use_grouper", True)

        # 2. Use Param Identifier
        # CLI flag enables it. If not set, fall back to config (default False)
        if args.use_param_identifier:
            use_param_identifier = True
        else:
            use_param_identifier = hset.get("use_param_identifier", False)

        # 3. Preload GT
        if args.preload_ground_truth:
            preload_gt = True
        else:
            preload_gt = hset.get("preload_ground_truth", False)

        # Validation: Cache Only mode (no grouper or PI) usually requires preloaded GT to function meaningfully
        if not use_grouper and not use_param_identifier and not preload_gt:
             raise ValueError(f"Dataset {dataset}: Both 'use_grouper' and 'use_param_identifier' are False, and 'preload_ground_truth' is also False. The cache needs either a grouper, a param identifier, or pre-loaded templates to function.")


        # 4. Checkpoints
        # Grouper Checkpoint
        grouper_ckpt = args.checkpoint_dir_grouper
        if not grouper_ckpt:
            grouper_ckpt = hset.get('checkpoint_dir')
        
        if grouper_ckpt:
             # Strict check if supplied
             if not os.path.exists(grouper_ckpt):
                 raise FileNotFoundError(f"Dataset {dataset}: Log Grouper checkpoint path does not exist: {grouper_ckpt}")
        elif use_grouper:
             # Required but not supplied
             raise ValueError(f"Dataset {dataset}: use_grouper=True but no 'checkpoint_dir' supplied via CLI or settings.")

        # Param ID Checkpoint
        param_id_ckpt = args.checkpoint_dir_param
        if not param_id_ckpt:
             param_id_ckpt = hset.get("param_id_checkpoint_dir")
        
        if param_id_ckpt:
             # Strict check if supplied
             if not os.path.exists(param_id_ckpt):
                  raise FileNotFoundError(f"Dataset {dataset}: Param Identifier checkpoint path does not exist: {param_id_ckpt}")
        elif use_param_identifier:
            # Required but not supplied
             raise ValueError(f"Dataset {dataset}: use_param_identifier=True but no 'param_id_checkpoint_dir' supplied via CLI or settings.")

        print(f"Running dataset={dataset} | use_grouper={use_grouper} | use_param_identifier={use_param_identifier} | preload_gt={preload_gt}")
        print(f"Grouper checkpoint: {grouper_ckpt}")
        print(f"Param ID checkpoint: {param_id_ckpt}")
        
        import time
        start_time = time.time()
        
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
                'checkpoint_dir': grouper_ckpt,
                'device': args.device if args.device else hset.get('device', 'cpu'),
                'min_match_prob': float(hset.get('min_match_prob', 0.5)),
                'use_grouper': use_grouper,
                'use_param_identifier': use_param_identifier,
                'param_id_checkpoint': param_id_ckpt if use_param_identifier else None,
                'preload_ground_truth': preload_gt,
            },

            otc=args.oracle_template_correction,
            complex=args.complex,
            frequent=args.frequent,
            result_file=result_file,
        )
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"Finished {dataset}. Time taken: {elapsed:.2f} seconds.")
        
        # Append to a central summary file
        summary_csv = os.path.join(output_dir, "Hybrid_detailed_summary.csv")
        file_exists = os.path.exists(summary_csv)
        
        metric_file_path = os.path.join(output_dir, result_file)
        
        pa_acc = "N/A"
        f1_score = "N/A"
        try:
             import pandas as pd
             if os.path.exists(metric_file_path):
                 m_df = pd.read_csv(metric_file_path)
                 row = m_df[m_df['Dataset'] == dataset]
                 if not row.empty:
                     pa_acc = row.iloc[-1].get('PA', 'N/A')
                     f1_score = row.iloc[-1].get('FGA', 'N/A')
        except Exception:
             pass

        with open(summary_csv, "a", newline="") as f:
            if not file_exists:
                f.write("Timestamp,Dataset,Mode,UseGrouper,UseParamID,PreloadGT,TimeSeconds,ParsingAccuracy,F1_Measure\n")
            
            import datetime
            ts = datetime.datetime.now().isoformat()
            # Mode string for clarity:
            mode_str = "Custom"
            if not use_grouper and not use_param_identifier: mode_str = "CacheOnly"
            elif not use_grouper and use_param_identifier: mode_str = "CachePlusPI"
            elif use_grouper and not use_param_identifier: mode_str = "CachePlusGrouper"
            elif use_grouper and use_param_identifier: mode_str = "FullPipeline"
            
            f.write(f"{ts},{dataset},{mode_str},{use_grouper},{use_param_identifier},{preload_gt},{elapsed:.4f},{pa_acc},{f1_score}\n")
        
    metric_file = os.path.join(output_dir, result_file)
    post_average(metric_file, f"Hybrid_{data_type}_complex={args.complex}_frequent={args.frequent}", args.complex, args.frequent)

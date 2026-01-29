# Pipeline Evaluation Reproduction Commands

This document contains the commands to reproduce the 4 configurations for the HybridParser evaluation.

## Prerequisites
- Ensure you are in the `datasets/loghub-2.0/benchmark/evaluation` directory.
- Ensure the python environment is active and has necessary dependencies (torch, etc.).

## Runs

### 1. Nur Cache (Cache Only)
**Hypothesis H1 - Effizienz**: Evaluates the baseline speed of the Trie cache without any ML models. Requires ground truth preloading to simulate a warm cache.

```bash
python Hybrid_eval.py --no_grouper --preload_ground_truth --run_name Run1_CacheOnly
```
*   `use_grouper`: False
*   `use_param_identifier`: False
*   `preload_ground_truth`: True (Required)

### 2. Cache + Log Grouper
**Hypothesis H2 - Grouping Benchmark**: Evaluates the grouping accuracy using the Log Grouper model.

```bash
python Hybrid_eval.py --use_grouper --run_name Run2_CacheGrouper
```
*   `use_grouper`: True
*   `use_param_identifier`: False (Default)

### 3. Cache + Parameter Identifier
**Hypothesis H3 - Parsing Benchmark**: Evaluates the parsing accuracy with the Parameter Identifier, isolated from the Log Grouper (using cache-only grouping logic).

```bash
python Hybrid_eval.py --no_grouper --use_param_identifier --preload_ground_truth --run_name Run3_CachePI
```
*   `use_grouper`: False
*   `use_param_identifier`: True
*   `preload_ground_truth`: True (Required)

### 4. Cache + Log Grouper + Parameter Identifier
**Hypothesis H4 - Robustheit / Gesamtsystem**: Evaluates the full pipeline with all components enabled.

```bash
python Hybrid_eval.py --use_param_identifier --run_name Run4_FullPipeline
```
*   `use_grouper`: True
*   `use_param_identifier`: True

## Output
Results will be generated in `datasets/loghub-2.0/benchmark/result/result_Hybrid_2k/<RunName>`.
A summary CSV `Hybrid_detailed_summary.csv` will be located in each run folder.

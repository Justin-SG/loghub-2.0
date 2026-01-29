from __future__ import annotations

import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import regex as re


# Resolve repo root so we can import the project HybridParser
CURR = Path(__file__).resolve()
# .../datasets/loghub-2.0/benchmark/logparser/Hybrid -> repo root is five levels up
REPO_ROOT = CURR.parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from HybridParser.HybridParser import HybridParser  # type: ignore



def _find_checkpoints_for_dataset(repo_root: Path, dataset: str) -> List[Path]:
    cand_roots = [repo_root / "results" / "actual", repo_root / "results" / "experiments"]
    found: List[Tuple[float, Path]] = []
    for root in cand_roots:
        if not root.exists():
            continue
        for p in root.rglob(f"fold_*_{dataset}"):
            if not p.is_dir():
                continue
            model = p / "model.pt"
            cfg = p / "config.json"
            if model.exists() and cfg.exists():
                try:
                    mt = model.stat().st_mtime
                except Exception:
                    mt = 0.0
                found.append((mt, p))
    found.sort(key=lambda t: t[0], reverse=True)
    return [p for _, p in found]


# Optional hardcoded mapping of datasets to checkpoint directories.
# Fill in absolute paths to folders that contain model.pt and config.json.
# If a mapping is empty or invalid, the adapter will fall back to param/env/auto-discovery.
DATASET_CHECKPOINTS: dict[str, str] = {
    # Example (Windows path):
    # "Apache": r"C:\\Users\\schoe\\Desktop\\master\\test_code\\master\\results\\actual\\...\\fold_?_Apache",
    "Apache": "",
    "BGL": "",
    "HDFS": "",
    "Hadoop": "",
    "HealthApp": "",
    "HPC": "",
    "Linux": "",
    "Mac": "",
    "OpenSSH": "",
    "OpenStack": "",
    "Proxifier": "",
    "Spark": "",
    "Thunderbird": "",
    "Zookeeper": "",
}


@dataclass
class Params:
    log_format: str
    indir: str
    outdir: str
    rex: List[str]
    depth: int
    st: float
    # Hybrid extras
    dataset: str
    checkpoint_dir: Optional[str] = None
    device: str = "cpu"
    min_match_prob: float = 0.5
    param_id_checkpoint: Optional[str] = None
    use_grouper: bool = True
    use_param_identifier: bool = False
    preload_ground_truth: bool = False



class LogParser:
    """Drain-compatible adapter for HybridParser.

    Matches the constructor and parse(log_file_basename) contract used by evaluator_main.
    """

    def __init__(self, **kwargs):
        self.params = Params(
            dataset=kwargs.pop("dataset"),
            # map possible kwargs to Params fields if they are passed directly
            **{k: v for k, v in kwargs.items() if k in Params.__annotations__}
        )
        self._parser: Optional[HybridParser] = None



    @staticmethod
    def _format_to_regex(log_format: str) -> re.Pattern:
        """
        Function to generate regular expression to split log messages.
        Adapted from Drain's implementation to support regex characters in log_format.
        """
        splitters = re.split(r'(<[^<>]+>)', log_format)
        regex = ''
        for k in range(len(splitters)):
            if k % 2 == 0:
                # Do not escape splitters as they might contain regex syntax (like \[ or ()?)
                splitter = splitters[k]
                splitter = re.sub(' +', r'\\s+', splitter)
                regex += splitter
            else:
                header = splitters[k].strip('<').strip('>')
                regex += '(?P<%s>.*?)' % header
        return re.compile(r"^" + regex + r"$")

    def _resolve_checkpoint(self) -> str:
        # 1) Explicit param override (from Hybrid_benchmark mapping via Hybrid_eval)
        if self.params.checkpoint_dir:
            return self.params.checkpoint_dir
        # 2) Environment variable
        env_ckpt = os.environ.get("HP_CHECKPOINT_DIR", "").strip()
        if env_ckpt:
            return env_ckpt
        # 3) Adapter internal mapping (optional convenience)
        mapped = DATASET_CHECKPOINTS.get(self.params.dataset)
        if mapped:
            p = Path(mapped)
            if (p / "model.pt").exists() and (p / "config.json").exists():
                return str(p)
        # 4) Auto-discovery
        candidates = _find_checkpoints_for_dataset(REPO_ROOT, self.params.dataset)
        if not candidates:
            raise FileNotFoundError(
                f"No checkpoint found for dataset '{self.params.dataset}'. Set HP_CHECKPOINT_DIR or pass checkpoint_dir."
            )
        return str(candidates[0])

    def parse(self, log_file_basename: str) -> None:
        indir = Path(self.params.indir)
        outdir = Path(self.params.outdir)
        in_log = indir / log_file_basename
        out_csv = outdir / f"{log_file_basename}_structured.csv"

        if not in_log.exists():
            raise FileNotFoundError(f"Input log not found: {in_log}")
        outdir.mkdir(parents=True, exist_ok=True)

        ckpt_dir = self._resolve_checkpoint()
        device = self.params.device or os.environ.get("HP_DEVICE", "cpu")
        min_prob = float(os.environ.get("HP_MIN_MATCH_PROB", self.params.min_match_prob))

        # Resolve param_id_checkpoint: prefer env var, then params
        pid_ckpt = os.environ.get("HP_PARAM_ID_CHECKPOINT", self.params.param_id_checkpoint)
        
        # Resolve use_grouper: prefer env var (as string "true"/"false"), then params
        use_grouper_env = os.environ.get("HP_USE_GROUPER")
        if use_grouper_env is not None:
             use_grouper = use_grouper_env.lower() in ("true", "1", "yes")
        else:
             use_grouper = self.params.use_grouper

        hp = HybridParser(
            checkpoint_path=ckpt_dir,
            device=device,
            min_match_prob=min_prob,
            param_id_checkpoint=pid_ckpt,
            use_grouper=use_grouper,
            use_param_identifier=self.params.use_param_identifier,
            rex=self.params.rex
        )

        self._parser = hp

        # --- OPTIONAL: Pre-load Ground Truth Templates for Cache Benchmark ---
        # If use_grouper=False and we are in a benchmark context, we might want to "cheat" and load GT.
        # Check params or env var? 
        # User asked for "implement ground truth preloading". 
        # I'll check a new flag in params: 'preload_ground_truth' (or inferred from config?)
        # Let's add 'preload_gt' to the Params or just check kwargs.
        # Actually Hybrid_eval can pass it.
        
        if self.params.preload_ground_truth:
            print("Preloading ground truth templates...")
            # We need to find the ground truth file. 
            # Ideally the benchmark runner knows where it is.
            # In Hybrid_eval.py:
            # candidate1 = lh_input_base / log_base / f"{log_name}_structured.csv" (Oracle)
            # The 'indir' param passed to LogParser is usually the directory CONTAINING the log file.
            # The structured file is usually named "<LogName>_structured.csv".
            
            # The log_file_basename is passed to `parse`.
            # Usually: `Apache_2k.log` -> `Apache_2k.log_structured.csv`
            gt_path = indir / f"{log_file_basename}_structured.csv"
            
            if not gt_path.exists():
                # Try replacing .log with .log_structured.csv?
                # Sometimes file is `Apache_2k.log`, oracle is `Apache_2k.log_structured.csv`
                # Sometimes `Apache.log` -> `Apache_structured.csv` ?
                pass
            
            if gt_path.exists():
                # import pandas as pd # Removed to avoid shadowing global pd
                try:
                    df = pd.read_csv(gt_path)

                    if "EventTemplate" in df.columns:
                        unique_templates = df["EventTemplate"].dropna().unique().tolist()
                        hp.bootstrap_cache(unique_templates)
                        print(f"Bootstrapped cache with {len(unique_templates)} templates.")
                    else:
                        print(f"Warning: 'EventTemplate' column not found in {gt_path}")
                except Exception as e:
                    print(f"Failed to preload GT: {e}")
            else:
                 print(f"Ground truth file not found at {gt_path}, skipping preload.")


        line_re = self._format_to_regex(self.params.log_format)

        line_ids: List[int] = []
        contents: List[str] = []
        with in_log.open("r", encoding="utf-8", errors="ignore") as fh:
            for idx, line in enumerate(fh, start=1):
                s = line.rstrip("\n")
                m = line_re.match(s)
                content = m.group("Content") if (m and "Content" in m.groupdict()) else s
                line_ids.append(idx)
                contents.append(content)

                
                
                




        event_ids: List[int] = []
        event_templates: List[str] = []
        total_lines = len(contents)
        for i, text in enumerate(contents):
            if not text:
                event_ids.append(-1)
                event_templates.append("")
                continue
            gid, tpl = hp.group(text)
            event_ids.append(int(gid))
            event_templates.append(str(tpl))
            
            if (i + 1) % 5000 == 0:
                print(f"Parsed {i+1}/{total_lines} lines...")

        # Post-process: Map all lines to their FINAL refined template for this group
        # This fixes the "template snapshot" issue where early lines keep unrefined strings.
        #final_templates = {gid: tpl for gid, tpl in hp.grouper.templates()}
        #for i in range(len(event_ids)):
        #    gid = event_ids[i]
        #    if gid != -1 and gid in final_templates:
        #        event_templates[i] = final_templates[gid]


        out_df = pd.DataFrame(
            {
                "LineId": line_ids,
                "Content": contents,
                "EventId": event_ids,
                "EventTemplate": event_templates,
            }
        )
        out_df.to_csv(out_csv, index=False, quoting=csv.QUOTE_MINIMAL)

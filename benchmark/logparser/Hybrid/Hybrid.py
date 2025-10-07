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


class LogParser:
    """Drain-compatible adapter for HybridParser.

    Matches the constructor and parse(log_file_basename) contract used by evaluator_main.
    """

    def __init__(self, **kwargs):
        self.params = Params(dataset=kwargs.pop("dataset"), **kwargs)  # type: ignore[arg-type]
        self._parser: Optional[HybridParser] = None

    @staticmethod
    def _format_to_regex(log_format: str) -> re.Pattern:
        pat = re.escape(log_format)
        pat = re.sub(r"\\<([A-Za-z0-9_]+)\\>", r"(?P<\1>.*?)", pat)
        pat = re.sub(r"\\\s+", r"\\s+", pat)
        return re.compile(r"^" + pat + r"$")

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

        hp = HybridParser(checkpoint_path=ckpt_dir, device=device, min_match_prob=min_prob)
        self._parser = hp

        line_re = self._format_to_regex(self.params.log_format)

        line_ids: List[int] = []
        contents: List[str] = []
        with in_log.open("r", encoding="utf-8", errors="ignore") as fh:
            for idx, line in enumerate(fh, start=1):
                s = line.rstrip("\n")
                m = line_re.match(s)
                content = m.group("Content") if (m and "Content" in m.groupdict()) else s
                for rx in (self.params.rex or []):
                    content = re.sub(rx, "<*>", content)
                line_ids.append(idx)
                contents.append(content)

        event_ids: List[int] = []
        event_templates: List[str] = []
        for text in contents:
            if not text:
                event_ids.append(-1)
                event_templates.append("")
                continue
            gid, tpl = hp.group(text)
            event_ids.append(int(gid))
            event_templates.append(str(tpl))

        out_df = pd.DataFrame(
            {
                "LineId": line_ids,
                "Content": contents,
                "EventId": event_ids,
                "EventTemplate": event_templates,
            }
        )
        out_df.to_csv(out_csv, index=False, quoting=csv.QUOTE_MINIMAL)

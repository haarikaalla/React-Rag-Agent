"""
Agent Logger
Lightweight in-session trace logger. Writes JSONL for later analysis.
"""

from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Dict, Any


LOG_DIR = Path("logs")


class AgentLogger:
    def __init__(self):
        LOG_DIR.mkdir(exist_ok=True)
        ts = int(time.time())
        self._path = LOG_DIR / f"session_{ts}.jsonl"
        self.records = []

    def log(self, query: str, result: Dict[str, Any], latency: float) -> None:
        record = {
            "ts":       time.strftime("%Y-%m-%dT%H:%M:%S"),
            "query":    query,
            "answer":   result.get("answer", ""),
            "steps":    result.get("steps", 0),
            "sources":  result.get("num_sources", 0),
            "tokens":   result.get("total_tokens", 0),
            "latency":  round(latency, 3),
        }
        self.records.append(record)
        with self._path.open("a") as f:
            f.write(json.dumps(record) + "\n")

    def summary(self) -> Dict[str, Any]:
        if not self.records:
            return {}
        latencies = [r["latency"] for r in self.records]
        tokens    = [r["tokens"] for r in self.records]
        return {
            "total_queries":   len(self.records),
            "avg_latency_s":   round(sum(latencies) / len(latencies), 2),
            "total_tokens":    sum(tokens),
            "avg_steps":       round(
                sum(r["steps"] for r in self.records) / len(self.records), 1
            ),
        }

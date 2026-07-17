# Archive

This folder contains **earlier iterations** of the analysis scripts that are **not
part of the final benchmark pipeline** documented in [`../REPRODUCTION.md`](../REPRODUCTION.md).

They are retained for historical reference only and are not maintained. The
final, supported scripts live in [`../scripts/`](../scripts/).

| Archived script | Superseded by / reason for archiving |
|---|---|
| `scripts/automl_performance_evaluation.py` | Older, simpler duplicate of [`../scripts/compare_algorithms.py`](../scripts/compare_algorithms.py). |
| `scripts/rank_extraction_benchmark.py` | Used a hardcoded execution-time data dict and a hardcoded `D:\` path; not wired into the final ranking workflow. |
| `scripts/critical_difference.py` | Read the stale `AutoML_results.csv` (pre-leakage-fix results) and used a hardcoded `D:\` path. The current ranking uses `compare_algorithms.py` on `performance_master_table.csv`. |
| `scripts/rank_performance.py` | Used a hardcoded `data = {...}` dict of performance numbers instead of reading the master table. |

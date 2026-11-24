{
  "model": "pibs scheduling model (steps == kernel launches; no wall-clock measurement)",
  "batches": [1, 2, 4, 8, 16, 32, 64],
  "rows": [
    {"batch": 1, "scheduler": "sequential", "steps": 1, "launches": 1, "mean_m": 3, "padded_total": 16, "wasted": 13, "wasted_pct": 81.25, "efficiency": 0.1875},
    {"batch": 1, "scheduler": "interleaved", "steps": 3, "launches": 3, "mean_m": 1, "padded_total": 48, "wasted": 45, "wasted_pct": 93.75, "efficiency": 0.0625},
    {"batch": 1, "scheduler": "bucketed", "steps": 1, "launches": 1, "mean_m": 3, "padded_total": 16, "wasted": 13, "wasted_pct": 81.25, "efficiency": 0.1875},
    {"batch": 2, "scheduler": "sequential", "steps": 2, "launches": 2, "mean_m": 3, "padded_total": 32, "wasted": 26, "wasted_pct": 81.25, "efficiency": 0.1875},
    {"batch": 2, "scheduler": "interleaved", "steps": 3, "launches": 3, "mean_m": 2, "padded_total": 48, "wasted": 42, "wasted_pct": 87.5, "efficiency": 0.125},
    {"batch": 2, "scheduler": "bucketed", "steps": 1, "launches": 1, "mean_m": 6, "padded_total": 16, "wasted": 10, "wasted_pct": 62.5, "efficiency": 0.375},
    {"batch": 4, "scheduler": "sequential", "steps": 4, "launches": 4, "mean_m": 3, "padded_total": 64, "wasted": 52, "wasted_pct": 81.25, "efficiency": 0.1875},
    {"batch": 4, "scheduler": "interleaved", "steps": 3, "launches": 3, "mean_m": 4, "padded_total": 48, "wasted": 36, "wasted_pct": 75, "efficiency": 0.25},
    {"batch": 4, "scheduler": "bucketed", "steps": 1, "launches": 1, "mean_m": 12, "padded_total": 16, "wasted": 4, "wasted_pct": 25, "efficiency": 0.75},
    {"batch": 8, "scheduler": "sequential", "steps": 8, "launches": 8, "mean_m": 3, "padded_total": 128, "wasted": 104, "wasted_pct": 81.25, "efficiency": 0.1875},
    {"batch": 8, "scheduler": "interleaved", "steps": 3, "launches": 3, "mean_m": 8, "padded_total": 48, "wasted": 24, "wasted_pct": 50, "efficiency": 0.5},
    {"batch": 8, "scheduler": "bucketed", "steps": 2, "launches": 2, "mean_m": 12, "padded_total": 32, "wasted": 8, "wasted_pct": 25, "efficiency": 0.75},
    {"batch": 16, "scheduler": "sequential", "steps": 16, "launches": 16, "mean_m": 3, "padded_total": 256, "wasted": 208, "wasted_pct": 81.25, "efficiency": 0.1875},
    {"batch": 16, "scheduler": "interleaved", "steps": 3, "launches": 3, "mean_m": 16, "padded_total": 48, "wasted": 0, "wasted_pct": 0, "efficiency": 1},
    {"batch": 16, "scheduler": "bucketed", "steps": 3, "launches": 3, "mean_m": 16, "padded_total": 48, "wasted": 0, "wasted_pct": 0, "efficiency": 1},
    {"batch": 32, "scheduler": "sequential", "steps": 32, "launches": 32, "mean_m": 3, "padded_total": 512, "wasted": 416, "wasted_pct": 81.25, "efficiency": 0.1875},
    {"batch": 32, "scheduler": "interleaved", "steps": 3, "launches": 3, "mean_m": 32, "padded_total": 96, "wasted": 0, "wasted_pct": 0, "efficiency": 1},
    {"batch": 32, "scheduler": "bucketed", "steps": 3, "launches": 3, "mean_m": 32, "padded_total": 96, "wasted": 0, "wasted_pct": 0, "efficiency": 1},
    {"batch": 64, "scheduler": "sequential", "steps": 64, "launches": 64, "mean_m": 3, "padded_total": 1024, "wasted": 832, "wasted_pct": 81.25, "efficiency": 0.1875},
    {"batch": 64, "scheduler": "interleaved", "steps": 3, "launches": 3, "mean_m": 64, "padded_total": 192, "wasted": 0, "wasted_pct": 0, "efficiency": 1},
    {"batch": 64, "scheduler": "bucketed", "steps": 3, "launches": 3, "mean_m": 64, "padded_total": 192, "wasted": 0, "wasted_pct": 0, "efficiency": 1}
  ],
  "breakeven": {
    "interleaved": 4,
    "bucketed": 2
  }
}

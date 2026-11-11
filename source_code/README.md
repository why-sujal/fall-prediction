# Early-Warning Fall Prediction on UMAFall

PRML individual course project: *Early-Warning Fall Prediction for Humanoid/Assistive Robotics: a cost-sensitive, time-to-impact trade-off analysis.*

The question: how many seconds **before impact** can a wearable-sensor model raise a fall alarm, and what does that extra lead time cost in missed falls and false alarms?

## Contents

| File | Description |
|---|---|
| `code.ipynb` | Full pipeline with saved outputs (Jupyter notebook, executed). |
| `code.py` | The same code as a plain Python script. |
| `requirements.txt` | Python dependencies. |
| `README.md` | This file. |

Sections of the notebook:
0. Setup and config; 1. Parsing and data audit; 2. Causal windows and 28 features; 3. Time-to-impact labelling; 4. Single-feature baselines; 5. Main experiment (lead times 0 to 1.0 s); 6. Model comparison; 7. Feature importance; 8. Cost-ratio sweep; 9. Sensor positions; 10. Error analysis; 11. Summary table; 12. Extension (strict metric and temporal persistence); 13. Live streaming demo.

## Data (not included)

UMAFall dataset (746 trials, 19 subjects), 78 MB: https://doi.org/10.6084/m9.figshare.4214283 (file `UMAFall_Dataset_corrected_version.zip`).

Put the 746 `UMAFall_Subject_*.csv` files anywhere under `./data/` (for example `data/UMAFall/`). The code finds them automatically. A different location can be given with `UMAFALL_DIR=/path/to/csvs`.

## Folder layout expected when running

```
project_folder/
    data/UMAFall/*.csv
    code.ipynb   code.py   requirements.txt
    results/                (created automatically)
```

Run everything from `project_folder` (the folder that contains `data/`).

## How to run

```
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab                        # open code.ipynb -> Kernel -> Restart and Run All Cells
```

Or as a script:

```
python code.py                     # full run
FAST=1 python code.py              # quick dry run (about 5 min, fewer lead times and folds)
REBUILD=1 python code.py           # ignore the parsed-data cache
```

Runtime: the first run parses all CSVs (a few minutes) and caches them in `results/trials_cache.pkl`. A full run then takes roughly 7 to 10 minutes on a laptop.

## Outputs

Written to `results/`:
- figures `fig01` to `fig09` (signals, peak overlap, lead-time trade-off, operating curves, feature importance, cost sweep, sensor positions, error analysis, persistence extension);
- tables `table_main_results.csv`, `table_model_comparison.csv`, `table_cost_sweep.csv`, `table_positions.csv`, `table_final_summary.csv`, `table_extension_strict_smoothing.csv`;
- the parsed-data cache.

## Method in brief

- Signals resampled to a common 20 Hz grid; impact time = peak chest acceleration magnitude (SMV).
- Causal 1 s windows with a 2 s context; 28 features (free-fall dip, dynamics, tilt, gyroscope, repetitiveness). Nothing after the decision time is used.
- For each lead time tau: positives are windows ending 0.3 s around (impact - tau); negatives are all daily-activity windows and fall windows more than 2 s before impact; windows in between are dropped.
- Models: random forest (main), logistic regression, histogram gradient boosting.
- Validation: stratified group 5-fold CV grouped by subject (seed 42); the alarm threshold minimises C_FN x FN + C_FP x FP (default 10:1) and is chosen by nested CV inside the training subjects only.
- Metrics: ROC-AUC, PR-AUC, window recall and precision, trial-level falls caught, median lead time, false alarms per minute.
- Extension: strict event-level metric and causal score smoothing (persistence).
- Live demo: `demo_trial("Subject_07_Fall_backwardFall_4")` replays one trial window by window, e.g. `demo_trial("Subject_04_ADL_Jogging_2", tau=0.3, delta=0.2)`. Run it after the notebook cells above it (needs variables in memory).

## Reproducibility

Fixed seed 42; folds are deterministic. Results can differ slightly across scikit-learn versions.

## Limitations

Simulated falls by young volunteers, 19 subjects, single seed, 20 Hz sampling, impact defined by the SMV peak, and a still-high false-alarm rate (not deployable as is).

"""Summarize archived diagnostics without promoting or tuning any model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .artifacts import write_json_atomic
from .round5_derived import bind
from .round7_archived_diagnostics import paired_mean_se
from .round7_data import PROJECT


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def corrector_report(path):
    source = read(path)
    daily = [d for f in source["folds"] for d in f["daily"]]
    dates = np.asarray([d["date_index"] for d in daily])
    indices = dates - dates.min()
    panels = {}
    for name in ("sealed_s0", "score_only_corrector", "all_families_corrector"):
        series = np.full(int(indices.max()) + 1, np.nan)
        series[indices] = [d[name] for d in daily]
        panels[name] = series

    def paired(left, right):
        result = paired_mean_se(left, right)
        result["mean_delta"] = result.pop("mean_ema_minus_raw")
        return result

    return {
        "source": bind(path),
        "folds": len(source["folds"]),
        "scored_dates": len(dates),
        "mean_composite_ic": {k: float(np.nanmean(v)) for k, v in panels.items()},
        "all_families_minus_s0": paired(
            panels["sealed_s0"], panels["all_families_corrector"]
        ),
        "all_families_minus_score_only": paired(
            panels["score_only_corrector"], panels["all_families_corrector"]
        ),
        "qualifications": [
            "IC of the composite against mean traded-horizon rank, not the headline equal-mean per-head IC",
            "matched chronological OOF controls; equal-date fit weights; no hyperparameter or checkpoint tuning",
            "trees consume original stored family fields with NaN missingness and bounded age; no later-window imputation or scaling",
            "no causal attribution to neural architecture and zero promotion weight",
        ],
    }


def report(audit_root):
    arms, all_fits = {}, []
    for arm in ("S0", "fundamentals"):
        fits = [
            read(audit_root / "archived_diagnostics" / f"{arm}_F{f}_{s}.json")
            for f in range(1, 15)
            for s in (11, 29, 47)
        ]
        all_fits.extend(fits)
        # Average matched seed ICs per date before estimating time-series error.
        # Do not treat three correlated seed observations as independent days.
        raw, ema = {}, {}
        for fit in fits:
            panel = fit["panels"]["evaluation"]
            for day, a, b in zip(
                panel["date_indices"],
                panel["raw_patience_daily_ic"],
                panel["final_ema_daily_ic"],
                strict=True,
            ):
                raw.setdefault(day, []).append(a)
                ema.setdefault(day, []).append(b)
        indices = range(min(raw), max(raw) + 1)

        def equal_seed_series(values):
            result = []
            for day in indices:
                row = np.asarray(values.get(day, [None]), float)
                result.append(
                    float(row.mean())
                    if len(row) == 3 and np.isfinite(row).all()
                    else np.nan
                )
            return np.asarray(result)

        a, b = equal_seed_series(raw), equal_seed_series(ema)
        arms[arm] = {
            "fits": len(fits),
            "mean_fit_raw_ic": float(
                np.mean([f["panels"]["fit"]["raw_patience_mean_ic"] for f in fits])
            ),
            "mean_selection_raw_ic": float(
                np.mean(
                    [f["panels"]["selection"]["raw_patience_mean_ic"] for f in fits]
                )
            ),
            "evaluation_mean_seed_raw_ic": float(np.nanmean(a)),
            "evaluation_mean_seed_ema_ic": float(np.nanmean(b)),
            "paired_evaluation_mean_seed_ic": paired_mean_se(a, b),
            "median_selection_paired_se": float(
                np.median(
                    [
                        f["panels"]["selection"]["paired"]["newey_west_mean_se"]
                        for f in fits
                    ]
                )
            ),
            "mean_shared_head_alignment": float(
                np.mean(
                    [f["head_gradients"]["D1_D2_vs_D3_D5_D10_cosine"] for f in fits]
                )
            ),
            "negative_shared_head_alignment_fits": sum(
                f["head_gradients"]["D1_D2_vs_D3_D5_D10_cosine"] < 0 for f in fits
            ),
            "median_sam_gap": float(
                np.median([f["head_gradients"]["sam_gap"] for f in fits])
            ),
        }
    result = {
        "schema": "BRAZIL_RV_ROUND7_DIAGNOSTIC_REPORT_V1",
        "status": "complete",
        "archived_fit_count": len(all_fits),
        "arms": arms,
        "corrector": corrector_report(
            audit_root / "residual_corrector_mature/result.json"
        ),
        "per_fit": all_fits,
        "source_files": [
            bind(p)
            for p in sorted((audit_root / "archived_diagnostics").glob("*.json"))
        ],
        "method_code": [
            bind(PROJECT / "research/src/brazil_rv/v2" / n)
            for n in (
                "round7_archived_diagnostics.py",
                "round7_corrector.py",
                "round7_diagnostics_report.py",
            )
        ],
        "qualifications": [
            "archived raw-patience versus final EMA changes checkpoint time and weighting; early EMA retains transferred initialization",
            "fit versus selection uses the same common D3/D5/D10 rule but different historical dates; gap alone does not prove underfitting",
            "archived per-epoch weights unavailable; only the two saved states are evaluated",
            "paired uncertainty is Newey-West mean SE with ten session lags, preserving missing-date gaps",
            "mean-seed IC is not the IC of a rank-ensembled score panel",
            "CPU FP32 compares both checkpoints equally; it need not exactly match archived BF16 score exports",
            "gradient/SAM and Jacobian diagnostics use the fixed final fit-date cross-section in evaluation mode",
            "repaired-store S0 diagnostics follow the authorized GPU anchor before screening",
        ],
        "new_neural_fits": False,
        "forward_capture": False,
        "heldout_access": False,
    }
    write_json_atomic(PROJECT / "docs/v2_round7_archived_diagnostics.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-root", type=Path, required=True)
    result = report(parser.parse_args().audit_root)
    print(
        json.dumps(
            {
                "fits": result["archived_fit_count"],
                "arms": result["arms"],
                "corrector": result["corrector"],
            }
        )
    )


if __name__ == "__main__":
    main()

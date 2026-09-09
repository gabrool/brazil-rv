"""Bounded fourteen-fold CPU checkpoint; no cloud or holdout entry point."""

from __future__ import annotations

import argparse
import gc
import shutil
import time
from pathlib import Path

import numpy as np

from . import research_rounds as rr
from .artifacts import sha256_file, write_json_atomic
from .baselines import build_store_baselines
from .config import PROJECT_ROOT, FULL_PROTOCOL, _expected_protocol_payload
from .contract import DEVELOPMENT_END, DEVELOPMENT_FOLDS
from .evaluate import enforce_registered_book_bounds
from .execution_policy import load_selected_policy
from .store import peak_rss_bytes

REGISTRATION = PROJECT_ROOT / "research/preregistrations/v2_research_checkpoint.md"
CELLS = ("a_slow", "b_intraday", "c_lending", "b_intraday_legacy12")
SCHEMA = "BRAZIL_RV_V2_RESEARCH_CHECKPOINT_V1"


def freeze(
    source_design: Path,
    output: Path,
    *,
    num_threads: int,
    reuse_controls_from: Path | None = None,
) -> None:
    source = rr._read_json(source_design)
    code = rr._git_identity()
    _, dates = rr._read_store_header(Path(source["store"]["root"]))
    if dates[-1] > np.datetime64(DEVELOPMENT_END):
        raise PermissionError("checkpoint refuses a store extending beyond development")
    *_, fold_table = rr._fold_indices(dates)
    registered = rr._read_json(REGISTRATION.with_suffix(".json"))
    expected = _expected_protocol_payload(FULL_PROTOCOL)
    if any(registered.get(key) != value for key, value in expected.items()):
        raise ValueError("registered checkpoint JSON differs from executable protocol")
    for row, fold in zip(
        registered["fold_table"],
        rr.development_folds(dates.astype("datetime64[D]").astype(object)),
        strict=True,
    ):
        if row["name"] != fold.name:
            raise ValueError("registered fold order differs from the source calendar")
        for key, values in (
            ("fit", fold.fit_dates),
            ("selection", fold.selection_dates),
            ("evaluation", fold.evaluation_dates),
            ("purge_before", fold.purge_before_dates),
            ("purge_after", fold.purge_after_dates),
        ):
            actual = [values[0].isoformat(), values[-1].isoformat()]
            if key not in ("purge_before", "purge_after"):
                actual.append(len(values))
            if row[key] != actual:
                raise ValueError(
                    f"registered {fold.name}/{key} differs from source calendar"
                )
    design = {
        "schema": SCHEMA,
        "implementation": code,
        "created_at_utc": rr._utc_now(),
        "source_design": {
            "path": str(source_design.resolve()),
            "sha256": sha256_file(source_design),
        },
        **{
            key: source[key]
            for key in ("store", "cdi", "bova11", "lending_archive", "execution_policy")
        },
        "cdi": registered["sources"]["cdi"],
        "registration": {
            "path": str(REGISTRATION),
            "sha256": sha256_file(REGISTRATION),
            "protocol_json_sha256": sha256_file(REGISTRATION.with_suffix(".json")),
        },
        "protocol": _expected_protocol_payload(FULL_PROTOCOL),
        "folds": fold_table,
        "gbdt_num_threads": num_threads,
        "gbdt_cells": list(CELLS),
        "tree_shap": "separate_post_fit_diagnostic_up_to_4096_active_rows_per_fold_evenly_spaced",
        "economics_label": "development_grade_close_proxy_with_observed_imputed_and_placeholder_borrow",
        "official_validation_accessed": False,
        "test_accessed": False,
    }
    output.mkdir(parents=True, exist_ok=False)
    if reuse_controls_from is not None:
        previous = rr._read_json(reuse_controls_from / "frozen_design.json")
        for key in (
            "schema",
            "store",
            "cdi",
            "bova11",
            "lending_archive",
            "execution_policy",
            "registration",
            "protocol",
            "folds",
        ):
            if previous[key] != design[key]:
                raise ValueError(f"control reuse source differs on {key}")
        markers = {}
        for name in rr._BASELINE_SIGNAL_NAMES:
            for fold in DEVELOPMENT_FOLDS:
                relative = Path("baselines") / name / fold
                source_root = reuse_controls_from / relative
                if not _completed(source_root):
                    raise ValueError(
                        f"control reuse requires an accepted cell: {relative}"
                    )
                marker = rr._read_json(source_root / "accepted.json")
                if (
                    marker["name"],
                    marker["fold"],
                    marker["engineering_acceptance"],
                ) != (name, fold, "passed"):
                    raise ValueError(
                        "control reuse cell identity or acceptance differs"
                    )
                markers[relative.as_posix()] = sha256_file(
                    source_root / "accepted.json"
                )
        for relative in markers:
            shutil.copytree(reuse_controls_from / relative, output / relative)
            if rr.inventory(reuse_controls_from / relative) != rr.inventory(
                output / relative
            ):
                raise ValueError("copied control differs byte-for-byte from its source")
        design["reused_controls"] = {
            "root": str(reuse_controls_from.resolve()),
            "frozen_design_sha256": sha256_file(
                reuse_controls_from / "frozen_design.json"
            ),
            "accepted_marker_sha256": markers,
            "reason": "unchanged_accepted_controls_after_moving_TreeSHAP_out_of_fitting",
            "recomputed": False,
        }
    write_json_atomic(output / "frozen_design.json", design)


def _context_arguments(context, source_hashes):
    return dict(
        store=context.store,
        cdi=context.cdi,
        bova11_close_by_index=context.bova11.close_by_session,
        bova11_binding=context.bova11_binding,
        lending_borrow=context.lending_borrow,
        source_hashes=source_hashes,
    )


def _finish_cell(root: Path, evaluation, *, name: str, fold: str) -> None:
    report = evaluation.result.report
    enforce_registered_book_bounds(report)
    point = report["primary_ic"]["estimate"]
    if name == "inverse_volatility_20" and (point is None or abs(point) >= 0.02):
        raise RuntimeError(f"null control stop: {name}/{fold}, primary_ic={point}")
    if name in rr._BASELINE_SIGNAL_NAMES and (point is None or abs(point) >= 0.10):
        raise RuntimeError(f"control magnitude stop: {name}/{fold}, primary_ic={point}")
    series = rr._daily_series(evaluation)
    # Calendar availability alone is not evidence that the held shorts had observed
    # rates. Attribute only sessions with positive short exposure and no imputed or
    # placeholder short notional; retain other dates as NaN to preserve time gaps.
    headline = [
        row
        for row in report["economics"]["daily_table"]
        if row["scenario"] == "borrow_balance"
    ]
    dates = [day.isoformat() for day in evaluation.result.dates]
    by_date = {row["date"]: row for row in headline}
    observed = np.asarray(
        [
            by_date[day]["held_short_imputed_notional_at_open"] == 0
            and by_date[day]["held_short_placeholder_notional_at_open"] == 0
            and by_date[day]["short_proceeds_interest_base_brl"] > 0
            for day in dates
        ]
    )
    series["observed_rate_sessions_net_excess_bps"] = np.where(
        observed, series["headline_net_excess_bps"], np.nan
    )
    observed_era = (
        ~evaluation.inputs.borrow_rate_imputed
        & ~evaluation.inputs.borrow_rate_placeholder
    ).any(axis=1)
    series["observed_rate_era_net_excess_bps"] = np.where(
        observed_era, series["headline_net_excess_bps"], np.nan
    )
    write_json_atomic(
        root / "daily_readouts.json",
        {
            "dates": dates,
            "series": {
                key: [float(v) if np.isfinite(v) else None for v in values]
                for key, values in series.items()
            },
            "observed_rate_session_mask": observed.tolist(),
        },
    )
    # This completion record is written last; incomplete cells cannot be resumed.
    write_json_atomic(
        root / "accepted.json",
        {
            "name": name,
            "fold": fold,
            "engineering_acceptance": "passed",
            "evaluation_sha256": sha256_file(root / "evaluation.json"),
            "score_manifest_sha256": sha256_file(root / "score_manifest.json"),
            "daily_readouts_sha256": sha256_file(root / "daily_readouts.json"),
        },
    )
    print(f"accepted {name}/{fold}: primary_ic={point}", flush=True)


def _completed(root: Path) -> bool:
    marker = root / "accepted.json"
    if not marker.exists():
        return False
    record = rr._read_json(marker)
    for filename, key in (
        ("evaluation.json", "evaluation_sha256"),
        ("score_manifest.json", "score_manifest_sha256"),
        ("daily_readouts.json", "daily_readouts_sha256"),
    ):
        if sha256_file(root / filename) != record[key]:
            raise ValueError(f"completed checkpoint cell changed: {root / filename}")
    return True


def _summary(output: Path) -> dict:
    candidates = (*rr._BASELINE_SIGNAL_NAMES, *CELLS)
    result = {}
    for name in candidates:
        family = "gbdt" if name in CELLS else "baselines"
        records = {
            fold: rr._read_json(output / family / name / fold / "daily_readouts.json")[
                "series"
            ]
            for fold in DEVELOPMENT_FOLDS
        }
        panels = {
            fold: {
                key: np.asarray(
                    [np.nan if v is None else v for v in values], dtype=np.float64
                )
                for key, values in row.items()
            }
            for fold, row in records.items()
        }
        labels = tuple(panels["F1"])
        result[name] = {
            "fourteen_folds": {
                key: rr._folded_bootstrap(
                    tuple(panels[f][key] for f in DEVELOPMENT_FOLDS)
                )
                for key in labels
            },
            "round3_evaluation_windows": {
                key: rr._folded_bootstrap(
                    tuple(panels[f][key] for f in DEVELOPMENT_FOLDS[-3:])
                )
                for key in labels
            },
            "folds": {
                fold: {
                    key: rr._folded_bootstrap((values,)) for key, values in row.items()
                }
                for fold, row in panels.items()
            },
        }
    return result


def run(output: Path) -> None:
    started = time.perf_counter()
    design = rr._read_json(output / "frozen_design.json")
    if design["schema"] != SCHEMA or design["implementation"] != rr._git_identity():
        raise ValueError("CPU checkpoint implementation differs from its frozen commit")
    if sha256_file(REGISTRATION) != design["registration"]["sha256"]:
        raise ValueError("checkpoint registration changed after freeze")
    if (
        sha256_file(REGISTRATION.with_suffix(".json"))
        != design["registration"]["protocol_json_sha256"]
    ):
        raise ValueError("checkpoint JSON changed after freeze")
    policy, binding = load_selected_policy(
        Path(design["execution_policy"]["root"]),
        expected_result_sha256=design["execution_policy"]["result_sha256"],
    )
    if binding != design["execution_policy"]:
        raise ValueError("selected execution policy binding changed")
    context = rr._open_ledger_replay(design)
    store = context.store
    _, dates = rr._read_store_header(Path(design["store"]["root"]))
    fit, selection, evaluation, windows, _ = rr._fold_indices(dates)
    sources = {
        "v2_store_manifest": design["store"]["manifest_sha256"],
        "bova11_manifest": context.bova11.manifest_sha256,
        "bova11_data": context.bova11.data_sha256,
        "lending_archive_manifest": context.lending_borrow.manifest_sha256,
        "lending_archive_balances": context.lending_borrow.balance_sha256,
        "lending_archive_rates": context.lending_borrow.rate_sha256,
        "cdi_development_extension": design["cdi"]["development_extension"]["sha256"],
        "cdi_experiment52_reference": design["cdi"]["experiment52_reference"]["sha256"],
        "checkpoint_registration": design["registration"]["sha256"],
    }
    kwargs = _context_arguments(context, sources)
    current = None
    try:
        for fold, indices in evaluation.items():
            if all(
                _completed(output / "baselines" / name / fold)
                for name in rr._BASELINE_SIGNAL_NAMES
            ):
                continue
            start = max(0, int(indices[0]) - 253)
            panels = build_store_baselines(
                store, np.arange(start, int(indices[-1]) + 1)
            )
            for name, panel in panels.items():
                current = f"baselines/{name}/{fold}"
                root = output / current
                if _completed(root):
                    continue
                if (root / "score_manifest.json").exists():
                    raise RuntimeError(
                        f"incomplete cell requires diagnosis and a fresh root: {current}"
                    )
                local = indices - start
                rr._persist_scores(
                    root,
                    {
                        "scores": panel.scores[local],
                        "score_mask": panel.score_mask[local],
                    },
                    {
                        **rr._source_tier_labels(store.manifest),
                        "engine": "naive_baseline",
                        "name": name,
                        "fold": fold,
                        "evaluation_date_indices": indices.tolist(),
                    },
                )
                evaluated = rr._evaluate(
                    **kwargs,
                    indices=indices,
                    scores=panel.scores[local],
                    score_mask=panel.score_mask[local],
                    fold=fold,
                    output=root / "evaluation.json",
                    execution_policy=policy,
                )
                _finish_cell(root, evaluated, name=name, fold=fold)
                del evaluated
            del panels
            gc.collect()
        for name in CELLS:
            for fold in DEVELOPMENT_FOLDS:
                current = f"gbdt/{name}/{fold}"
                root = output / current
                if _completed(root):
                    continue
                if (root / "score_manifest.json").exists():
                    raise RuntimeError(
                        f"incomplete cell requires diagnosis and a fresh root: {current}"
                    )
                reports, _ = rr._run_gbdt_candidate(
                    **kwargs,
                    rung=name,
                    fit={fold: fit[fold]},
                    selection={fold: selection[fold]},
                    evaluation={fold: evaluation[fold]},
                    fit_target_window={fold: windows[fold]},
                    pretrain=None,
                    decay_half_life=None,
                    root=output / "gbdt" / name,
                    num_threads=design["gbdt_num_threads"],
                    execution_policy=policy,
                )
                _finish_cell(root, reports[fold], name=name, fold=fold)
                del reports
                gc.collect()
        write_json_atomic(
            output / "checkpoint_cpu_result.json",
            {
                "schema": SCHEMA,
                "status": "cpu_rebaseline_complete",
                "finished_at_utc": rr._utc_now(),
                "candidates": _summary(output),
                "access": context.access,
                "paid_compute_launched": False,
                "official_validation_accessed": False,
                "test_accessed": False,
            },
        )
    except BaseException as error:
        write_json_atomic(
            output / "stop.json",
            {
                "cell": current,
                "error": str(error),
                "at_utc": rr._utc_now(),
                "status": "stopped_for_diagnosis",
            },
        )
        raise
    finally:
        store.close()
        write_json_atomic(
            output / "cpu_run_resources.json",
            {
                "elapsed_seconds": time.perf_counter() - started,
                "process_peak_rss_bytes": peak_rss_bytes(),
                "fit_threads": design["gbdt_num_threads"],
                "scope": "this_CPU_process_including_store_views_training_and_evaluation",
            },
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "run"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-design", type=Path)
    parser.add_argument("--num-threads", type=int, default=8)
    parser.add_argument("--reuse-controls-from", type=Path)
    args = parser.parse_args(argv)
    if args.action == "freeze":
        if args.source_design is None:
            parser.error("freeze requires --source-design")
        freeze(
            args.source_design,
            args.root,
            num_threads=args.num_threads,
            reuse_controls_from=args.reuse_controls_from,
        )
    else:
        run(args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

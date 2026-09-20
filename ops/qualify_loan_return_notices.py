"""Read completed books for exposure ranking and numerical-change attribution."""

import json
from pathlib import Path
import shutil

import numpy as np

from brazil_rv.v2.artifacts import sha256_file, write_json_atomic

PROJECT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def main():
    path = PROJECT / "docs/v2_economic_data_scaling_run.json"
    pointer = read(path)
    binding = pointer["loan_return_notice_audit"]
    assert sha256_file(Path(binding["path"])) == binding["sha256"]
    run = read(binding["path"])
    root = Path(binding["path"]).parent
    out = root.parent / "readout_qualification"
    out.mkdir(exist_ok=False)
    shutil.copyfile(__file__, out / "executed.py")
    store = Path(read(PROJECT / "docs/v2_data_inputs.json")["store"]["root"])
    calendar = np.load(store / "date_index.npy").astype("datetime64[D]")
    isins = np.load(store / "isin_index.npy")
    raw = np.load(store / "raw_close.npy", mmap_mode="r")
    before_root = root.parent / "qualified"
    rankings, contrasts, numerical, cases = [], [], [], []
    for case in run["cases"]:
        label = case["case"]
        scenario = label.rsplit("_", 1)[-1]
        book = read(root / label / "book.json")
        base_label = label.rsplit("_", 1)[0] + "_base"
        base = read(root / base_label / "book.json")
        daily = np.array(book["daily"]["net_excess_bps"])
        delta = daily - np.array(base["daily"]["net_excess_bps"])
        contrasts.append(
            dict(
                case=label,
                versus=base_label,
                mean_net_excess_delta_bps=float(delta.mean()),
                sum_net_excess_delta_bps=float(delta.sum()),
                terminal_nav=book["summary"]["terminal_nav"]
                if "terminal_nav" in book["summary"]
                else None,
            )
        )
        previous = before_root / label / "book.json"
        if previous.exists():
            old = read(previous)
            difference = daily - np.array(old["daily"]["net_excess_bps"])
            numerical.append(
                dict(
                    case=label,
                    previous_book=str(previous),
                    previous_sha256=sha256_file(previous),
                    mean_net_excess_delta_bps=float(difference.mean()),
                    sum_net_excess_delta_bps=float(difference.sum()),
                    interpretation="combined full-return, mandatory-close and strict-face implementation effects; not alpha",
                )
            )
        if "_10000000_" in label and scenario != "base":
            with np.load(root / label / "account.npz") as state:
                rows = np.searchsorted(
                    calendar, np.array(book["state_dates"], dtype="datetime64[D]")
                )
                shares, marks = state["signed_shares"], state["mark_price"]
                missing_short = (shares < 0) & ~np.isfinite(raw[rows])
                values = np.where(
                    missing_short, np.abs(shares * np.nan_to_num(marks)), 0
                )
                maximum = values.max(0)
                for name in np.flatnonzero(maximum > 0):
                    observed = np.flatnonzero(missing_short[:, name])
                    rankings.append(
                        dict(
                            case=label,
                            security_index=int(name),
                            isin=str(isins[name]),
                            maximum_missing_quote_short_value_brl=float(maximum[name]),
                            missing_quote_held_sessions=len(observed),
                            first_date=str(calendar[rows[observed[0]]]),
                            last_date=str(calendar[rows[observed[-1]]]),
                            disposition="ranked held inventory candidate; not an inferred loan reference, source event or executable liquidation",
                        )
                    )
        cases.append({k: v for k, v in case.items() if k != "summary"})
    rankings.sort(key=lambda r: -r["maximum_missing_quote_short_value_brl"])
    report = {
        "status": "completed_book_readout_qualified_stage_A_still_incomplete",
        "source": binding,
        "cases": cases,
        "same_intentions_max_nav_error_brl": max(
            c["identical_intentions_nav_error"] for c in cases
        ),
        "adaptive_max_nav_difference_bps_by_capital": {
            str(cap): max(
                c["adaptive_nav_difference_bps"]
                for c in cases
                if f"_{cap}_" in c["case"]
            )
            for cap in run["plan"]["capital"]
        },
        "adaptive_max_target_difference": max(
            c["account_errors"]["decision"] for c in cases
        ),
        "label_qualification": "The executed adaptive_solve_within_primal_tolerance label compares TWO DIFFERENT adaptive paths. It is not a constraint-feasibility test. Every individual QP passed its existing residual check. Current recipe calls this adaptive_target_difference_le_2e_6; no book rerun for this metadata correction.",
        "uncertainty": "Old fixed minimum charges can differ for genuinely small free-optimum trades between independently accumulated/re-solved books. Report their measured total path differences; not daily alpha or bit-exact adaptive equality.",
        "ranked_missing_quote_exposure": rankings,
        "scenario_contrasts": contrasts,
        "implementation_attribution": numerical,
        "remaining": "Source-bound NATU/ALSC/SOMA held-event disposition; pre-custody disposal and remaining allocation/timing/invoice/grouping/minimum/clearing bounds; final real-model matched economics and refits.",
        "timing": {
            "adaptive_audit_seconds": run["seconds"],
            "scope": "24x128-session/all933 books, independent adaptive and identical-intention account replays; not a GPU fit ETA",
        },
    }
    digest = write_json_atomic(out / "manifest.json", report)
    pointer["loan_return_notice_qualification"] = {
        "path": str(out / "manifest.json"),
        "sha256": digest,
    }
    write_json_atomic(path, pointer)
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "status",
                    "same_intentions_max_nav_error_brl",
                    "adaptive_max_nav_difference_bps_by_capital",
                    "adaptive_max_target_difference",
                )
            }
        )
    )


if __name__ == "__main__":
    main()

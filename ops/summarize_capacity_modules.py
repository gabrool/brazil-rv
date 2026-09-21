"""Summarize saved fit/module probes for the conditional capacity disposition."""

from collections import defaultdict
import json
import math
from pathlib import Path
from statistics import median

from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.data_repair import binding, bound_json

PROJECT = Path(__file__).resolve().parents[1]


def numeric_fields(value, prefix=""):
    """Keep module/metric names separate; probe dates are not independent fits."""
    if isinstance(value, dict):
        for key, child in value.items():
            yield from numeric_fields(child, f"{prefix}.{key}" if prefix else key)
    elif isinstance(value, list):
        for child in value:
            yield from numeric_fields(child, prefix)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        yield prefix, value


def main():
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text())
    keys = ["stage_c_refit_results"] + [
        f"stage_d_{wave}_results" for wave in ("width", "depth", "lstm")
    ]
    # All named comparisons must be complete before this combined disposition input.
    results = {key: bound_json(run[key]) for key in keys}
    groups, evidence = defaultdict(list), []
    for key, result in results.items():
        for fit in bound_json(result["fit_diagnostics"]):
            if "/F/" not in fit["key"]:
                continue
            manifest = bound_json(fit["underlying_fit"])
            assert manifest["status"] == "completed"
            detail = bound_json(fit["diagnostics"]) if fit["diagnostics"] else {}
            probes = [
                p for p in detail.get("module_probes", []) if p["state"] == "selected"
            ]
            assert len(probes) <= 1
            arm = fit["key"].split("/")[0]
            groups[(key, arm)].append((fit, manifest, probes))
            evidence.append(
                dict(key=fit["key"], result=run[key], diagnostic=fit["diagnostics"])
            )
    summaries = []
    for (key, arm), fits in groups.items():
        fields = defaultdict(list)
        missing, nonfinite = defaultdict(int), defaultdict(int)
        for fit, manifest, probes in fits:
            for name in (
                "selected_clean_fit_ic",
                "terminal_clean_fit_ic",
                "selection_ic",
                "ema_selection_ic",
                "epochs",
                "complete_fit_seconds",
                "median_epoch_after_first_seconds",
                "peak_cuda_bytes",
            ):
                value = fit[name]
                if value is None:
                    missing[name] += 1
                else:
                    fields["fit." + name].append(value)
            fields["fit.padded_name_count"].append(
                manifest["contract"]["padded_name_count"]
            )
            for probe in probes:
                for name, value in numeric_fields(probe):
                    if math.isfinite(value):
                        fields["probe." + name].append(value)
                    else:
                        nonfinite[name] += 1
        summaries.append(
            dict(
                result_key=key,
                arm=arm,
                fits=len(fits),
                fits_with_selected_probe=sum(bool(p) for _, _, p in fits),
                missing=dict(missing),
                nonfinite_probe_values=dict(nonfinite),
                fields={
                    name: dict(
                        count=len(values),
                        min=min(values),
                        median=median(values),
                        max=max(values),
                    )
                    for name, values in sorted(fields.items())
                },
            )
        )
    out = Path(run["root"]) / "capacity_disposition"
    out.mkdir(exist_ok=True)
    target = out / "module_diagnostics.json"
    assert not target.exists()
    write_json_atomic(
        target,
        dict(
            status="saved_diagnostics_summarized",
            results={key: run[key] for key in keys},
            groups=summaries,
            evidence=evidence,
            recipe=binding(Path(__file__)),
            limits="Only saved F-fit diagnostics from the complete C/width/depth/LSTM comparisons. Selected module probes and clean-fit IC belong to selected.pt, not the separately selected EMA checkpoint used for exports where enabled. No forward, gradient, optimizer, source or portfolio computation reruns. Multiple probe dates/horizons and correlated fits are not independent observations. Finite activity, attention entropy, update magnitudes and fit-selection gaps do not prove a module is adequately sized or identify a causal capacity bottleneck; economic results and explicit interpretation govern the conditional branch. Missing probes remain missing, and zero values are not automatically defects.",
        ),
    )
    run["stage_d_module_diagnostics"] = binding(target)
    write_json_atomic(pointer, run)
    print(json.dumps(dict(groups=len(summaries), fits=len(evidence))), flush=True)


if __name__ == "__main__":
    main()

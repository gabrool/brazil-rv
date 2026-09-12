"""Fit-only scaling and explicit common/per-security characteristic inputs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import collate_v2_daily
from .round5_magnitude import FitClip
from .round7 import cross_market_partition


@dataclass(frozen=True)
class RobustScaler:
    center: np.ndarray
    scale: np.ndarray
    fit_date_indices: tuple[int, ...]

    @classmethod
    def fit(cls, values, valid, fit_date_indices):
        """Caller supplies only fit rows; stock fields use valid stock-days."""
        centers, scales = [], []
        for f in range(values.shape[-1]):
            sample = values[..., f][valid[..., f] & np.isfinite(values[..., f])]
            if sample.size:
                low, center, high = np.quantile(sample, (0.25, 0.5, 0.75))
                scale = high - low
            else:
                center, scale = 0.0, 1.0
            centers.append(center)
            scales.append(scale if scale > 0 else 1.0)
        return cls(
            np.asarray(centers, np.float32),
            np.asarray(scales, np.float32),
            tuple(int(i) for i in fit_date_indices),
        )

    def transform(self, values, valid):
        return np.where(
            valid, np.clip((values - self.center) / self.scale, -5.0, 5.0), 0.0
        ).astype(np.float32)

    def payload(self):
        return {
            "center": self.center.tolist(),
            "scale": self.scale.tolist(),
            "fit_date_indices": list(self.fit_date_indices),
            "clip": [-5.0, 5.0],
            "missing_fit_rule": "center zero, unit scale; never fit on later data",
        }

    @classmethod
    def from_payload(cls, payload):
        return cls(
            np.asarray(payload["center"], np.float32),
            np.asarray(payload["scale"], np.float32),
            tuple(payload["fit_date_indices"]),
        )


def common_snapshot(values, valid, ages):
    """One observation per date/field, independent of which stock carries it."""
    known = valid.any(axis=-2)
    minimum = np.where(valid, values, np.inf).min(axis=-2)
    maximum = np.where(valid, values, -np.inf).max(axis=-2)
    age_min = np.where(valid, ages, np.inf).min(axis=-2)
    age_max = np.where(valid, ages, -np.inf).max(axis=-2)
    if np.any(known & ((minimum != maximum) | (age_min != age_max))):
        raise ValueError("registered common field varies by security")
    return np.where(known, minimum, 0.0), known, np.where(known, age_min, -1.0)


@dataclass
class Round7Preprocessing:
    native: RobustScaler | None
    common: RobustScaler | None
    magnitude: FitClip | None
    common_columns: tuple[int, ...] = ()
    per_name_columns: tuple[int, ...] = ()

    @classmethod
    def fit(cls, dataset, *, split_common):
        store, rows = dataset.store, dataset.date_indices
        native = common = magnitude = None
        common_columns = per_name_columns = ()
        active = store.read("active", rows)
        if "fundamentals_native" in dataset.enabled_sidecars:
            native = RobustScaler.fit(
                store.read("sidecar_fundamentals_native_values", rows),
                store.read("sidecar_fundamentals_native_valid", rows)
                & active[..., None],
                rows,
            )
        if "magnitudes" in dataset.enabled_sidecars:
            fitted = FitClip.fit(
                store.read("sidecar_magnitudes_values", rows),
                store.read("sidecar_magnitudes_valid", rows),
                active,
                np.arange(len(rows)),
            )
            magnitude = FitClip(fitted.lower, fitted.upper, tuple(int(i) for i in rows))
        if split_common:
            names = store.manifest["feature_names"]["sidecar_cross_market"]
            shared, per_name = cross_market_partition(names)
            common_columns = tuple(names.index(n) for n in shared)
            per_name_columns = tuple(names.index(n) for n in per_name)
            values, masks = [], []
            # Avoid a full [date, 933, 65] temporary and duplicate-date weighting.
            for start in range(0, len(rows), 128):
                chunk = rows[start : start + 128]
                panel = [
                    store.read("sidecar_cross_market_" + suffix, chunk)[
                        ..., common_columns
                    ]
                    for suffix in ("values", "valid", "age_sessions")
                ]
                v, m, _ = common_snapshot(*panel)
                values.append(
                    np.concatenate(
                        (v, store.read("common_state_diagnostic_values", chunk)),
                        axis=-1,
                    )
                )
                masks.append(
                    np.concatenate(
                        (m, store.read("common_state_diagnostic_valid", chunk)), axis=-1
                    )
                )
            common = RobustScaler.fit(
                np.concatenate(values), np.concatenate(masks), rows
            )
        return cls(native, common, magnitude, common_columns, per_name_columns)

    def payload(self):
        return {
            **{
                name: getattr(self, name).payload()
                if getattr(self, name) is not None
                else None
                for name in ("native", "common", "magnitude")
            },
            "common_columns": list(self.common_columns),
            "per_name_columns": list(self.per_name_columns),
        }

    @classmethod
    def from_payload(cls, payload):
        return cls(
            RobustScaler.from_payload(payload["native"]) if payload["native"] else None,
            RobustScaler.from_payload(payload["common"]) if payload["common"] else None,
            FitClip.from_payload(payload["magnitude"])
            if payload["magnitude"]
            else None,
            tuple(payload["common_columns"]),
            tuple(payload["per_name_columns"]),
        )

    def collate(self, samples, *, fixed_name_count):
        prepared = []
        for sample in samples:
            sample = dict(sample)
            if self.native is not None:
                sample["sidecar_fundamentals_native_values"] = self.native.transform(
                    sample["sidecar_fundamentals_native_values"],
                    sample["sidecar_fundamentals_native_valid"],
                )
            if self.common is not None:
                keys = tuple(
                    "sidecar_cross_market_" + suffix
                    for suffix in ("values", "valid", "age_sessions")
                )
                values, valid, ages = common_snapshot(
                    *(sample[key][..., self.common_columns] for key in keys)
                )
                diagnostic_mask = sample["common_state_feature_mask"]
                valid = np.concatenate((valid, diagnostic_mask))
                values = np.concatenate((values, sample["common_state_features"]))
                sample["common_state_features"] = self.common.transform(values, valid)
                sample["common_state_feature_mask"] = valid
                sample["common_state_age_sessions"] = np.concatenate(
                    (ages, np.where(diagnostic_mask, 1.0, -1.0))
                ).astype(np.float32)
                for key in keys:
                    sample[key] = sample[key][..., self.per_name_columns]
            prepared.append(sample)
        return collate_v2_daily(prepared, fixed_name_count=fixed_name_count)

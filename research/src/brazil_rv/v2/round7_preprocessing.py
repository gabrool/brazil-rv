"""Causal, mask-preserving conditioning shared by scalar model consumers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import collate_v2_daily


def cross_market_partition(names):
    common = tuple(
        n
        for n in names
        if n.startswith("shock_")
        or n
        in {
            "foreign_flow_1",
            "foreign_flow_5",
            "foreign_flow_month_reset",
            "foreign_flow_methodology_change",
            "ewz_minus_bova11_1",
        }
    )
    return common, tuple(n for n in names if n not in common)


@dataclass(frozen=True)
class RobustScaler:
    center: np.ndarray
    scale: np.ndarray
    fit_date_indices: tuple[int, ...]
    support: tuple[int, ...]
    varying: tuple[bool, ...]
    passthrough: tuple[bool, ...]
    inherited: tuple[bool, ...]

    @classmethod
    def fit_columns(cls, samples, fit_date_indices, *, passthrough=(), parent=None):
        """Fit only admitted rows; retain a parent's established coordinates.

        An absent/constant parent field has learned no scale. It may acquire one
        from the child's fit window, never from selection or later test rows.
        """
        passthrough = tuple(passthrough) or (False,) * len(samples)
        centers, scales, support, varying, inherited = [], [], [], [], []
        for f, sample in enumerate(samples):
            sample = np.asarray(sample, np.float64)
            sample = sample[np.isfinite(sample)]
            changes = bool(sample.size and sample.min() != sample.max())
            inherit = parent is not None and parent.varying[f]
            if inherit:
                center, scale = parent.center[f], parent.scale[f]
            elif sample.size:
                low, center, high = np.quantile(sample, (0.25, 0.5, 0.75))
                scale = high - low
                if scale == 0 and changes:
                    deviations = np.abs(sample - center)
                    scale = np.median(deviations[deviations > 0])
                if scale == 0:
                    scale = 1.0
            else:
                center, scale = 0.0, 1.0
            centers.append(center)
            scales.append(scale)
            support.append(int(sample.size))
            varying.append(changes or inherit)
            inherited.append(inherit)
        dates = set(int(i) for i in fit_date_indices)
        if parent is not None and any(inherited):
            dates.update(parent.fit_date_indices)
        return cls(
            np.asarray(centers),
            np.asarray(scales),
            tuple(sorted(dates)),
            tuple(support),
            tuple(varying),
            passthrough,
            tuple(inherited),
        )

    @classmethod
    def fit(cls, values, valid, fit_date_indices, **kwargs):
        return cls.fit_columns(
            [values[..., f][valid[..., f]] for f in range(values.shape[-1])],
            fit_date_indices,
            **kwargs,
        )

    def transform(self, values, valid):
        # Smooth tails preserve ordering and magnitude. Float64 arithmetic avoids
        # overflow/cancellation before the bounded-range result becomes float32.
        transformed = np.arcsinh(
            (np.asarray(values, np.float64) - self.center) / self.scale
        )
        transformed = np.where(self.passthrough, values, transformed)
        return np.where(valid, transformed, 0.0).astype(np.float32)

    def payload(self):
        return {
            "center": self.center.tolist(),
            "scale": self.scale.tolist(),
            "fit_date_indices": list(self.fit_date_indices),
            "support": list(self.support),
            "varying": list(self.varying),
            "passthrough": list(self.passthrough),
            "inherited": list(self.inherited),
            "transform": "asinh((value - fit_median) / fit_IQR); no clipping",
            "zero_IQR": "median nonzero absolute deviation, else unit scale",
        }

    @classmethod
    def from_payload(cls, payload):
        return cls(
            np.asarray(payload["center"]),
            np.asarray(payload["scale"]),
            *(
                tuple(payload[k])
                for k in (
                    "fit_date_indices",
                    "support",
                    "varying",
                    "passthrough",
                    "inherited",
                )
            ),
        )


def common_snapshot(values, valid, ages):
    """One observation per date/field; age is independent of value availability."""
    known = valid.any(axis=-2)
    minimum = np.where(valid, values, np.inf).min(axis=-2)
    maximum = np.where(valid, values, -np.inf).max(axis=-2)
    age_known = ages >= 0
    age_min = np.where(age_known, ages, np.inf).min(axis=-2)
    age_max = np.where(age_known, ages, -np.inf).max(axis=-2)
    if np.any(known & (minimum != maximum)) or np.any(
        age_known.any(axis=-2) & (age_min != age_max)
    ):
        raise ValueError("registered common field varies by security")
    return (
        np.where(known, minimum, 0.0),
        known,
        np.where(age_known.any(axis=-2), age_min, -1.0),
    )


@dataclass
class Round7Preprocessing:
    families: dict[str, RobustScaler]
    diagnostic: RobustScaler | None = None
    common_columns: tuple[int, ...] = ()
    per_name_columns: tuple[int, ...] = ()
    feature_names: dict[str, tuple[str, ...]] | None = None

    @classmethod
    def fit(cls, dataset, *, split_common, parent=None):
        store, rows = dataset.store, dataset.date_indices
        families, names_by_family = {}, {}
        common_columns = per_name_columns = ()
        specifications = {
            (s["family"], s["name"]): s
            for s in store.manifest["metadata"]["feature_schema"]["specifications"]
        }
        for family in dataset.enabled_sidecars:
            prefix = "sidecar_" + family
            names = tuple(store.manifest["feature_names"][prefix])
            names_by_family[family] = names
            shared_indices = ()
            if family == "cross_market":
                shared, per_name = cross_market_partition(names)
                shared_indices = tuple(names.index(n) for n in shared)
                if split_common:
                    common_columns = shared_indices
                    per_name_columns = tuple(names.index(n) for n in per_name)
            columns = [[] for _ in names]
            for start in range(0, len(rows), 128):
                chunk = rows[start : start + 128]
                values = store.read(prefix + "_values", chunk)
                valid = (
                    store.read(prefix + "_valid", chunk)
                    & store.read("active", chunk)[..., None]
                )
                if shared_indices:
                    common_values, common_valid, _ = common_snapshot(
                        values[..., shared_indices],
                        valid[..., shared_indices],
                        store.read(prefix + "_age_sessions", chunk)[
                            ..., shared_indices
                        ],
                    )
                for f in range(len(names)):
                    if f in shared_indices:
                        j = shared_indices.index(f)
                        columns[f].append(common_values[:, j][common_valid[:, j]])
                    else:
                        columns[f].append(values[..., f][valid[..., f]])
            inherited = None if parent is None else parent.families.get(family)
            if inherited is not None and parent.feature_names[family] != names:
                raise ValueError("parent scaler field order differs")
            families[family] = RobustScaler.fit_columns(
                [np.concatenate(column) for column in columns],
                rows,
                passthrough=tuple(
                    specifications[prefix, n]["transform"]
                    in {"binary", "bounded_fraction", "signed_identity", "age_sessions"}
                    for n in names
                ),
                parent=inherited,
            )
        diagnostic = None
        if split_common:
            diagnostic = RobustScaler.fit(
                store.read("common_state_diagnostic_values", rows),
                store.read("common_state_diagnostic_valid", rows),
                rows,
                parent=None if parent is None else parent.diagnostic,
            )
        return cls(
            families, diagnostic, common_columns, per_name_columns, names_by_family
        )

    def payload(self):
        return {
            "families": {n: s.payload() for n, s in self.families.items()},
            "diagnostic": None
            if self.diagnostic is None
            else self.diagnostic.payload(),
            "common_columns": list(self.common_columns),
            "per_name_columns": list(self.per_name_columns),
            "feature_names": {
                n: list(v) for n, v in (self.feature_names or {}).items()
            },
        }

    @classmethod
    def from_payload(cls, payload):
        return cls(
            {n: RobustScaler.from_payload(p) for n, p in payload["families"].items()},
            RobustScaler.from_payload(payload["diagnostic"])
            if payload["diagnostic"]
            else None,
            tuple(payload["common_columns"]),
            tuple(payload["per_name_columns"]),
            {n: tuple(v) for n, v in payload["feature_names"].items()},
        )

    def transform_sample(self, sample):
        sample = dict(sample)
        for family, scaler in self.families.items():
            prefix = "sidecar_" + family
            sample[prefix + "_values"] = scaler.transform(
                sample[prefix + "_values"], sample[prefix + "_valid"]
            )
        if self.common_columns:
            keys = tuple(
                "sidecar_cross_market_" + s for s in ("values", "valid", "age_sessions")
            )
            values, valid, ages = common_snapshot(
                *(sample[k][..., self.common_columns] for k in keys)
            )
            diagnostic_mask = sample["common_state_feature_mask"]
            sample["common_state_features"] = np.concatenate(
                (
                    values,
                    self.diagnostic.transform(
                        sample["common_state_features"], diagnostic_mask
                    ),
                )
            )
            sample["common_state_feature_mask"] = np.concatenate(
                (valid, diagnostic_mask)
            )
            sample["common_state_age_sessions"] = np.concatenate(
                (ages, np.where(diagnostic_mask, 1.0, -1.0))
            ).astype(np.float32)
            for key in keys:
                sample[key] = sample[key][..., self.per_name_columns]
        return sample

    def collate(self, samples, *, fixed_name_count):
        return collate_v2_daily(
            [self.transform_sample(s) for s in samples],
            fixed_name_count=fixed_name_count,
        )

"""Compact, JSON-safe payloads for the local spatial viewer."""

from __future__ import annotations

from typing import Any

import anndata as ad
import numpy as np
import pandas as pd


SAMPLE_COLUMNS = ("sample_id", "sample", "Sample", "image_id", "imageid", "ImageNumber")
UNCERTAINTY_PALETTES = {
    "Normalized entropy": "viridis",
    "Gini impurity": "plasma",
    "Margin uncertainty": "magma",
    "Spatial discordance": "cividis",
}
MAX_VIEWER_MARKERS = 200


def spatial_payload(
    adata: ad.AnnData,
    *,
    title: str,
    label_column: str | None = None,
    max_points: int = 30_000,
) -> dict[str, Any] | None:
    """Return a deterministic, bounded payload when valid spatial coordinates exist."""
    if "spatial" not in adata.obsm:
        return None
    coordinates = np.asarray(adata.obsm["spatial"])
    if coordinates.ndim != 2 or coordinates.shape[1] < 2 or len(coordinates) != adata.n_obs:
        return None
    valid = np.isfinite(coordinates[:, :2]).all(axis=1)
    candidates = np.flatnonzero(valid)
    if not len(candidates):
        return None
    if len(candidates) > max_points:
        positions = np.linspace(0, len(candidates) - 1, max_points, dtype=int)
        selected = candidates[positions]
    else:
        selected = candidates

    labels = None
    if label_column and label_column in adata.obs:
        labels = adata.obs[label_column].astype(str).iloc[selected].tolist()

    marker_names = list(map(str, adata.var_names[:MAX_VIEWER_MARKERS]))
    marker_values = None
    if marker_names:
        expression = adata.layers["exprs"] if "exprs" in adata.layers else adata.X
        selected_expression = expression[selected, : len(marker_names)]
        if hasattr(selected_expression, "toarray"):
            selected_expression = selected_expression.toarray()
        selected_expression = np.asarray(selected_expression, dtype=float)
        marker_values = []
        for marker_index in range(len(marker_names)):
            marker_values.append([
                None if not np.isfinite(value) else round(float(value), 4)
                for value in selected_expression[:, marker_index]
            ])

    uncertainty_metrics = {}
    metric_names = {
        "entropy": "Normalized entropy",
        "gini_impurity": "Gini impurity",
        "margin_uncertainty": "Margin uncertainty",
        "spatial_discordance": "Spatial discordance",
    }
    for column, label in metric_names.items():
        if column not in adata.obs:
            continue
        values = pd.to_numeric(adata.obs[column], errors="coerce")
        uncertainty_metrics[label] = [
            None if pd.isna(value) else round(float(value), 5)
            for value in values.iloc[selected]
        ]

    sample_column = next((column for column in SAMPLE_COLUMNS if column in adata.obs), None)
    sample_counts = None
    if sample_column:
        valid_samples = adata.obs[sample_column].astype(str).iloc[candidates]
        sample_counts = {
            str(name): int(count) for name, count in valid_samples.value_counts().items()
        }
    samples = (
        adata.obs[sample_column].astype(str).iloc[selected].tolist()
        if sample_column
        else None
    )
    probability_source = str(
        adata.uns.get("cytogater_uncertainty_probability_source", "not available")
    )
    return {
        "title": title,
        "x": coordinates[selected, 0].astype(float).round(4).tolist(),
        "y": coordinates[selected, 1].astype(float).round(4).tolist(),
        "labels": labels,
        "markerNames": marker_names,
        "markerValues": marker_values,
        "markerAssay": "exprs" if "exprs" in adata.layers else "X",
        "markersTruncated": int(adata.n_vars) > MAX_VIEWER_MARKERS,
        "uncertaintyMetrics": uncertainty_metrics or None,
        "uncertaintyPalettes": (
            {name: UNCERTAINTY_PALETTES[name] for name in uncertainty_metrics}
            if uncertainty_metrics
            else None
        ),
        "samples": samples,
        "sampleCounts": sample_counts,
        "totalCells": int(adata.n_obs),
        "displayedCells": int(len(selected)),
        "labelColumn": label_column,
        "uncertaintyNote": (
            "Calculated by cytoGater from a complete probability matrix. "
            f"Probability source: {probability_source}."
        ),
    }

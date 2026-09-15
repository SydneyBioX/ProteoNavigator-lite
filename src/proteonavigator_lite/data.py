"""Load user inputs into the single AnnData/lineage contract used by the agent."""

from __future__ import annotations

import ast
import hashlib
import shutil
import subprocess
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


SUPPORTED_DATA = {".h5ad", ".csv", ".rds"}


def _identifier(path: Path) -> str:
    token = f"{path.name}:{path.stat().st_size}:{path.stat().st_mtime_ns}"
    return hashlib.sha256(token.encode()).hexdigest()[:12]


def ensure_exprs_layer(adata: ad.AnnData) -> str:
    """Ensure the cytoGater-compatible ``exprs`` layer exists."""
    if "exprs" not in adata.layers:
        adata.layers["exprs"] = adata.X.copy()
    return "exprs"


def load_dataset(path: str | Path, work_dir: Path) -> tuple[str, ad.AnnData, str]:
    """Load .h5ad, expression CSV, or SpatialExperiment RDS as AnnData."""
    source = Path(path).resolve()
    suffix = source.suffix.lower()
    if suffix not in SUPPORTED_DATA:
        raise ValueError(f"Unsupported dataset format {suffix!r}; use .h5ad, .csv, or .rds")
    if suffix == ".h5ad":
        adata = ad.read_h5ad(source)
    elif suffix == ".csv":
        frame = pd.read_csv(source, index_col=0)
        if frame.empty:
            raise ValueError("The expression CSV is empty")
        numeric = frame.apply(pd.to_numeric, errors="raise")
        adata = ad.AnnData(
            X=numeric.to_numpy(dtype=np.float32),
            obs=pd.DataFrame(index=numeric.index.astype(str)),
            var=pd.DataFrame(index=numeric.columns.astype(str)),
        )
    else:
        adata = _load_rds(source, work_dir)
    adata.obs_names = adata.obs_names.astype(str)
    adata.var_names = adata.var_names.astype(str)
    adata.obs_names_make_unique()
    adata.var_names_make_unique()
    assay = ensure_exprs_layer(adata)
    return _identifier(source), adata, assay


def _load_rds(path: Path, work_dir: Path) -> ad.AnnData:
    """Convert a SpatialExperiment/SingleCellExperiment RDS through Rscript."""
    rscript = shutil.which("Rscript")
    if not rscript:
        raise RuntimeError("RDS input requires Rscript; use the matching .h5ad or CSV instead")
    output = work_dir / f"rds_{_identifier(path)}"
    output.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).parents[2] / "scripts" / "convert_rds.R"
    completed = subprocess.run(
        [rscript, str(script), str(path), str(output)],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"RDS conversion failed: {completed.stderr[-2000:]}")
    expression = pd.read_csv(output / "expression.csv", index_col=0)
    obs = pd.read_csv(output / "obs.csv", index_col=0)
    obs.index = obs.index.astype(str)
    expression.index = expression.index.astype(str)
    obs = obs.reindex(expression.index)
    adata = ad.AnnData(X=expression.to_numpy(dtype=np.float32), obs=obs)
    adata.var_names = expression.columns.astype(str)
    spatial = output / "spatial.csv"
    if spatial.exists():
        coords = pd.read_csv(spatial, index_col=0).reindex(expression.index)
        adata.obsm["spatial"] = coords.to_numpy(dtype=float)
    return adata


def _parse_marker_list(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("["):
        parsed = ast.literal_eval(text)
        return [str(item).strip() for item in parsed]
    return [item.strip() for item in text.replace(";", ",").split(",") if item.strip()]


def load_lineage_table(path: str | Path) -> pd.DataFrame:
    """Load wide R-style or long Python-style lineage CSV."""
    frame = pd.read_csv(path)
    if "Populations" in frame.columns:
        markers = [column for column in frame.columns if column != "Populations"]
        return pd.DataFrame(
            {
                "cell_type": frame["Populations"].astype(str),
                "pos_markers": frame[markers].eq(1).apply(
                    lambda row: row.index[row].astype(str).tolist(), axis=1
                ),
                "neg_markers": frame[markers].eq(-1).apply(
                    lambda row: row.index[row].astype(str).tolist(), axis=1
                ),
            }
        )
    required = {"cell_type", "pos_markers", "neg_markers"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            "Lineage CSV needs either 'Populations' plus marker columns, or "
            f"{sorted(required)}; missing {sorted(missing)}"
        )
    output = frame[list(required)].copy()
    output["cell_type"] = output["cell_type"].astype(str)
    for column in ("pos_markers", "neg_markers"):
        output[column] = output[column].map(_parse_marker_list)
    return output[["cell_type", "pos_markers", "neg_markers"]]


def lineage_from_definitions(definitions: list[dict]) -> pd.DataFrame:
    """Convert validated model output to cytoGater lineage format."""
    return pd.DataFrame(
        {
            "cell_type": [item["cell_type"] for item in definitions],
            "pos_markers": [item.get("positive_markers", []) for item in definitions],
            "neg_markers": [item.get("negative_markers", []) for item in definitions],
        }
    )


def validate_lineage(lineage: pd.DataFrame, markers: list[str]) -> dict:
    """Validate biological definitions against the actual panel."""
    panel = set(markers)
    errors: list[str] = []
    warnings: list[str] = []
    if lineage.empty:
        errors.append("The lineage table has no cell types")
    duplicated = lineage["cell_type"][lineage["cell_type"].duplicated()].tolist()
    if duplicated:
        errors.append(f"Duplicate cell types: {', '.join(duplicated)}")
    signatures: dict[tuple, str] = {}
    for row in lineage.itertuples(index=False):
        positive = [str(marker) for marker in row.pos_markers]
        negative = [str(marker) for marker in row.neg_markers]
        missing_pos = [marker for marker in positive if marker not in panel]
        missing_neg = [marker for marker in negative if marker not in panel]
        usable_positive = [marker for marker in positive if marker in panel]
        if not usable_positive:
            errors.append(f"{row.cell_type}: no positive marker is present in the panel")
        if missing_pos:
            errors.append(f"{row.cell_type}: missing positive marker(s): {', '.join(missing_pos)}")
        if missing_neg:
            warnings.append(f"{row.cell_type}: missing negative marker(s): {', '.join(missing_neg)}")
        signature = (tuple(sorted(positive)), tuple(sorted(negative)))
        if signature in signatures:
            errors.append(f"{row.cell_type} has the same definition as {signatures[signature]}")
        signatures[signature] = str(row.cell_type)
    return {"valid": not errors, "errors": errors, "warnings": warnings}


def lineage_revision(lineage: pd.DataFrame) -> str:
    """Return a stable short revision identifier for a lineage table."""
    payload = lineage.to_json(orient="records")
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


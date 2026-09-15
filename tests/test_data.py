from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from proteonavigator_lite.data import (
    load_dataset,
    load_lineage_table,
    validate_lineage,
)
from proteonavigator_lite.models import DatasetRecord, SessionWorkspace
from proteonavigator_lite.services import WorkspaceService


def test_load_expression_csv_creates_exprs_layer(tmp_path: Path):
    path = tmp_path / "expression.csv"
    pd.DataFrame(
        {"CD3": [1.0, 2.0], "CD20": [0.0, 4.0]}, index=["c1", "c2"]
    ).to_csv(path)

    _, result, assay = load_dataset(path, tmp_path)

    assert result.shape == (2, 2)
    assert assay == "exprs"
    np.testing.assert_array_equal(result.layers["exprs"], result.X)


def test_load_h5ad_preserves_and_normalizes_data(tmp_path: Path):
    path = tmp_path / "sample.h5ad"
    ad.AnnData(
        X=np.ones((3, 2)),
        obs=pd.DataFrame(index=["a", "b", "c"]),
        var=pd.DataFrame(index=["CD3", "CD20"]),
    ).write_h5ad(path)

    _, result, assay = load_dataset(path, tmp_path)

    assert assay == "exprs"
    assert list(result.var_names) == ["CD3", "CD20"]


def test_load_wide_lineage_table(tmp_path: Path):
    path = tmp_path / "lineage.csv"
    pd.DataFrame(
        {
            "Populations": ["T cell", "B cell"],
            "CD3": [1, -1],
            "CD20": [-1, 1],
        }
    ).to_csv(path, index=False)

    result = load_lineage_table(path)

    assert result.loc[0, "pos_markers"] == ["CD3"]
    assert result.loc[0, "neg_markers"] == ["CD20"]


def test_lineage_validation_blocks_missing_positive_markers():
    lineage = pd.DataFrame(
        {
            "cell_type": ["T cell"],
            "pos_markers": [["CD3", "NOT_IN_PANEL"]],
            "neg_markers": [["CD20"]],
        }
    )

    report = validate_lineage(lineage, ["CD3", "CD20"])

    assert not report["valid"]
    assert "missing positive" in report["errors"][0]


def test_chainlit_bin_upload_uses_original_h5ad_name(tmp_path: Path):
    upload_path = tmp_path / "chainlit-upload.bin"
    ad.AnnData(
        X=np.ones((2, 1)),
        obs=pd.DataFrame(index=["a", "b"]),
        var=pd.DataFrame(index=["CD3"]),
    ).write_h5ad(upload_path)
    service = WorkspaceService(SessionWorkspace(root=tmp_path / "session"))
    service.workspace.root.mkdir()

    summary = service.ingest_dataset(upload_path, "real_sample.h5ad")

    assert summary["name"] == "real_sample.h5ad"
    assert summary["n_cells"] == 2
    assert service.workspace.dataset.source_path.suffix == ".h5ad"


def test_single_cell_type_lineage_edit_preserves_other_definitions(tmp_path: Path):
    adata = ad.AnnData(
        X=np.ones((2, 3)),
        var=pd.DataFrame(index=["CD20", "CD3", "CD68"]),
    )
    workspace = SessionWorkspace(root=tmp_path)
    workspace.dataset = DatasetRecord("data", "data.h5ad", tmp_path / "data.h5ad", adata, "X")
    service = WorkspaceService(workspace)
    original = service.save_lineage(
        pd.DataFrame(
            {
                "cell_type": ["Bcell", "Tcell"],
                "pos_markers": [["CD20"], ["CD3"]],
                "neg_markers": [["CD3"], ["CD20"]],
            }
        )
    )

    updated = service.update_lineage_cell_type(
        "Bcell", ["CD20"], ["CD3", "CD68"]
    )

    assert updated["revision"] != original["revision"]
    definitions = {item["cell_type"]: item for item in updated["definitions"]}
    assert definitions["Bcell"]["neg_markers"] == ["CD3", "CD68"]
    assert definitions["Tcell"]["pos_markers"] == ["CD3"]


def test_revised_lineage_requires_fresh_gating_branch(tmp_path: Path):
    adata = ad.AnnData(
        X=np.ones((2, 2)), var=pd.DataFrame(index=["CD20", "CD3"])
    )
    workspace = SessionWorkspace(root=tmp_path)
    workspace.dataset = DatasetRecord("data", "data.h5ad", tmp_path / "data.h5ad", adata, "X")
    service = WorkspaceService(workspace)
    original = service.save_lineage(
        pd.DataFrame(
            {
                "cell_type": ["Bcell", "Tcell"],
                "pos_markers": [["CD20"], ["CD3"]],
                "neg_markers": [["CD3"], ["CD20"]],
            }
        )
    )
    previous = tmp_path / "annotated_oldplan.h5ad"
    previous.touch()
    workspace.last_output = previous
    workspace.last_result = {"lineage_revision": original["revision"]}
    service.update_lineage_cell_type("Bcell", ["CD20"], ["CD3"])
    # Force a genuinely new revision for the branch check.
    service.update_lineage_cell_type("Bcell", ["CD20", "CD3"], [])

    with pytest.raises(ValueError, match="Start a new analysis"):
        service.create_plan([{"kind": "select_core_cells"}])

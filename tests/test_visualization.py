import anndata as ad
import numpy as np
import pandas as pd

from proteonavigator_lite.visualization import spatial_payload


def test_spatial_payload_contains_labels_uncertainty_and_samples():
    adata = ad.AnnData(
        X=np.ones((4, 1)),
        obs=pd.DataFrame(
            {
                "sample_id": ["a", "a", "b", "b"],
                "annotation_label": ["T", "B", "T", "B"],
                "entropy": [0.1, np.nan, 0.2, 0.8],
                "gini_impurity": [0.2, 0.3, 0.4, 0.5],
                "margin_uncertainty": [0.3, 0.4, 0.5, 0.6],
                "spatial_discordance": [0.4, 0.5, 0.6, 0.7],
            },
            index=["c1", "c2", "c3", "c4"],
        ),
    )
    adata.var_names = ["CD3"]
    adata.layers["exprs"] = np.array([[0.1], [0.2], [0.3], [0.4]])
    adata.obsm["spatial"] = np.array([[0, 1], [2, 3], [4, 5], [6, 7]])

    payload = spatial_payload(
        adata, title="result", label_column="annotation_label", max_points=3
    )

    assert payload is not None
    assert payload["totalCells"] == 4
    assert payload["displayedCells"] == 3
    assert len(payload["x"]) == len(payload["labels"]) == 3
    assert set(payload["samples"]) == {"a", "b"}
    assert payload["sampleCounts"] == {"a": 2, "b": 2}
    assert payload["markerNames"] == ["CD3"]
    assert payload["markerValues"] == [[0.1, 0.2, 0.4]]
    assert payload["markerAssay"] == "exprs"
    entropy = payload["uncertaintyMetrics"]["Normalized entropy"]
    assert any(value is None for value in entropy)
    assert payload["uncertaintyPalettes"] == {
        "Normalized entropy": "viridis",
        "Gini impurity": "plasma",
        "Margin uncertainty": "magma",
        "Spatial discordance": "cividis",
    }
    assert len(set(payload["uncertaintyPalettes"].values())) == 4


def test_spatial_payload_is_absent_without_coordinates():
    adata = ad.AnnData(X=np.ones((2, 1)))
    assert spatial_payload(adata, title="input") is None

from pathlib import Path

import anndata as ad
import joblib
import numpy as np
import pandas as pd
import pytest

from proteonavigator_lite.backend import (
    CytoGaterBackend,
    _prepare_prediction_targets,
    _prepare_training_labels,
)
from proteonavigator_lite.models import AnnotationPlan, PipelineStage


def _toy_data(n_per_class=100):
    rng = np.random.default_rng(9)
    values = np.column_stack(
        [
            np.r_[rng.normal(4, 0.2, n_per_class), rng.normal(0, 0.2, n_per_class)],
            np.r_[rng.normal(0, 0.2, n_per_class), rng.normal(4, 0.2, n_per_class)],
        ]
    )
    adata = ad.AnnData(
        X=values,
        obs=pd.DataFrame(index=[f"c{i}" for i in range(2 * n_per_class)]),
        var=pd.DataFrame(index=["CD3", "CD20"]),
    )
    adata.layers["exprs"] = adata.X.copy()
    lineage = pd.DataFrame(
        {
            "cell_type": ["T cell", "B cell"],
            "pos_markers": [["CD3"], ["CD20"]],
            "neg_markers": [["CD20"], ["CD3"]],
        }
    )
    return adata, lineage


def test_predicted_labels_can_be_promoted_to_training_core_labels():
    adata = ad.AnnData(
        X=np.ones((4, 1)),
        obs=pd.DataFrame(
            {
                "annotation_label": pd.Categorical(
                    ["Tumor", "Unassigned", "Unknown", "Stroma"]
                )
            }
        ),
        var=pd.DataFrame(index=["marker"]),
    )
    column = _prepare_training_labels(
        adata,
        {
            "training_label_column": "annotation_label",
            "exclude_training_labels": ["Unknown", "Unassigned"],
        },
    )

    assert adata.obs[column].tolist() == ["Tumor", "Unknown", "Unknown", "Stroma"]
    assert adata.uns["cytogater_training_label_source"] == "annotation_label"


def test_prediction_targets_only_selected_label_values():
    adata = ad.AnnData(
        X=np.ones((4, 1)),
        obs=pd.DataFrame(
            {"annotation_label": ["Tumor", "Unassigned", "Unknown", "Stroma"]}
        ),
        var=pd.DataFrame(index=["marker"]),
    )
    column, sentinel = _prepare_prediction_targets(
        adata,
        {
            "prediction_label_column": "annotation_label",
            "target_labels": ["Unassigned", "Unknown"],
        },
    )

    assert adata.obs[column].tolist() == ["Tumor", sentinel, sentinel, "Stroma"]
    assert adata.uns["cytogater_prediction_target_cells"] == 2


def test_backend_runs_soft_gating(tmp_path: Path):
    rng = np.random.default_rng(4)
    values = np.column_stack(
        [np.r_[rng.normal(3, 0.3, 50), rng.normal(0, 0.3, 50)],
         np.r_[rng.normal(0, 0.3, 50), rng.normal(3, 0.3, 50)]]
    )
    adata = ad.AnnData(
        X=values,
        obs=pd.DataFrame(index=[f"c{i}" for i in range(100)]),
        var=pd.DataFrame(index=["CD3", "CD20"]),
    )
    adata.layers["exprs"] = adata.X.copy()
    lineage = pd.DataFrame(
        {
            "cell_type": ["T cell", "B cell"],
            "pos_markers": [["CD3"], ["CD20"]],
            "neg_markers": [["CD20"], ["CD3"]],
        }
    )
    plan = AnnotationPlan(
        plan_id="test-plan",
        dataset_id="test-data",
        lineage_revision="test-lineage",
        workflow="soft_gating",
        stages=[PipelineStage(kind="soft_gating", parameters={"unknown_thresh": 0.2})],
        approved=True,
    )

    result = CytoGaterBackend().run(adata, lineage, plan, tmp_path)

    assert Path(result["output_path"]).exists()
    assert sum(result["label_counts"].values()) == adata.n_obs


def test_backend_writes_spatial_annotation_and_uncertainty(tmp_path: Path):
    adata, lineage = _toy_data(30)
    adata.obsm["spatial"] = np.column_stack(
        [np.arange(adata.n_obs), np.arange(adata.n_obs) % 7]
    )
    plan = AnnotationPlan(
        plan_id="spatial-plan",
        dataset_id="test-data",
        lineage_revision="test-lineage",
        workflow="soft_gating -> calculate_uncertainty",
        stages=[
            PipelineStage(kind="soft_gating", parameters={"unknown_thresh": 0.2}),
            PipelineStage(kind="calculate_uncertainty", parameters={"k_spatial": 5}),
        ],
        approved=True,
    )

    result = CytoGaterBackend().run(adata, lineage, plan, tmp_path)
    annotated = ad.read_h5ad(result["output_path"])

    assert Path(result["spatial_plot_path"]).exists()
    assert "annotation_label" in annotated.obs
    assert {"entropy", "gini_impurity", "margin_uncertainty", "spatial_discordance"}.issubset(
        annotated.obs.columns
    )
    assert "annotation_uncertainty" not in annotated.obs


def test_uncertainty_can_continue_from_completed_gating_output(tmp_path: Path):
    adata, lineage = _toy_data(30)
    adata.obs["sample_id"] = "sample-1"
    adata.obsm["spatial"] = np.column_stack(
        [np.arange(adata.n_obs), np.arange(adata.n_obs) % 7]
    )
    gating_plan = AnnotationPlan(
        plan_id="gating-step",
        dataset_id="test-data",
        lineage_revision="test-lineage",
        workflow="soft_gating",
        stages=[PipelineStage(kind="soft_gating", parameters={"unknown_thresh": 0.2})],
        approved=True,
    )
    gating_output = CytoGaterBackend().run(adata, lineage, gating_plan, tmp_path / "step1")
    uncertainty_plan = AnnotationPlan(
        plan_id="uncertainty-step",
        dataset_id="test-data",
        lineage_revision="test-lineage",
        workflow="calculate_uncertainty",
        stages=[PipelineStage(kind="calculate_uncertainty", parameters={"k_spatial": 5})],
        input_path=gating_output["output_path"],
        approved=True,
    )

    result = CytoGaterBackend().run(adata, lineage, uncertainty_plan, tmp_path / "step2")
    annotated = ad.read_h5ad(result["output_path"])

    assert result["ordered_stages"] == ["calculate_uncertainty"]
    assert {"entropy", "gini_impurity", "margin_uncertainty", "spatial_discordance"}.issubset(
        annotated.obs.columns
    )


def test_core_cells_can_be_selected_as_a_separate_continuation(tmp_path: Path):
    adata, lineage = _toy_data(30)
    gating_plan = AnnotationPlan(
        plan_id="gating-before-core",
        dataset_id="test-data",
        lineage_revision="test-lineage",
        workflow="soft_gating",
        stages=[PipelineStage(kind="soft_gating", parameters={"unknown_thresh": 0.2})],
        approved=True,
    )
    gating_output = CytoGaterBackend().run(adata, lineage, gating_plan, tmp_path / "gating")
    core_plan = AnnotationPlan(
        plan_id="core-step",
        dataset_id="test-data",
        lineage_revision="test-lineage",
        workflow="select_core_cells",
        stages=[
            PipelineStage(
                kind="select_core_cells",
                parameters={"cutoff_strategy": "quantile", "probability": 0.8},
            )
        ],
        input_path=gating_output["output_path"],
        approved=True,
    )

    result = CytoGaterBackend().run(adata, lineage, core_plan, tmp_path / "core")
    annotated = ad.read_h5ad(result["output_path"])

    assert result["ordered_stages"] == ["select_core_cells"]
    assert result["final_label_column"] == "cutoff_label"
    assert "cutoff_label" in annotated.obs
    assert "Unknown" in set(annotated.obs["annotation_label"].astype(str))


def test_tree_gating_saves_models_for_later_plotting(tmp_path: Path):
    adata, lineage = _toy_data(30)
    plan = AnnotationPlan(
        plan_id="tree-models",
        dataset_id="test-data",
        lineage_revision="test-lineage",
        workflow="tree_gating",
        stages=[
            PipelineStage(
                kind="tree_gating",
                parameters={"max_depth": 2, "min_cells": 10, "min_score": 0.0},
            )
        ],
        approved=True,
    )

    result = CytoGaterBackend().run(adata, lineage, plan, tmp_path)

    assert Path(result["tree_path"]).exists()
    assert result["tree_cell_types"] == ["B cell", "T cell"]


def test_backend_runs_ordered_soft_gating_then_random_forest(tmp_path: Path):
    adata, lineage = _toy_data()
    plan = AnnotationPlan(
        plan_id="rf-plan",
        dataset_id="test-data",
        lineage_revision="test-lineage",
        workflow="soft_gating -> random_forest",
        stages=[
            PipelineStage(kind="soft_gating", parameters={"unknown_thresh": 0.2}),
            PipelineStage(
                kind="random_forest",
                parameters={
                    "reference_quantile": 0.8,
                    "num_trees": 10,
                    "cv_folds": 2,
                    "repeats": 1,
                    "agreement_thresh": 0.0,
                    "num_threads": 1,
                    "seed": 1,
                },
            ),
        ],
        approved=True,
    )

    result = CytoGaterBackend().run(adata, lineage, plan, tmp_path)

    assert result["ordered_stages"] == ["soft_gating", "random_forest"]
    assert result["final_label_column"] == "rf_label_filled"
    assert sum(result["label_counts"].values()) == adata.n_obs


def test_backend_runs_ordered_soft_gating_then_weighted_knn(tmp_path: Path):
    adata, lineage = _toy_data(40)
    plan = AnnotationPlan(
        plan_id="knn-plan",
        dataset_id="test-data",
        lineage_revision="test-lineage",
        workflow="soft_gating -> weighted_knn",
        stages=[
            PipelineStage(kind="soft_gating", parameters={"unknown_thresh": 0.2}),
            PipelineStage(
                kind="weighted_knn",
                parameters={
                    "reference_quantile": 0.8,
                    "cv_folds": 2,
                    "repeats": 1,
                    "agreement_thresh": 0.0,
                    "k": 3,
                    "seed": 1,
                },
            ),
        ],
        approved=True,
    )

    result = CytoGaterBackend().run(adata, lineage, plan, tmp_path)

    assert result["final_label_column"] == "knn_label_filled"
    assert sum(result["label_counts"].values()) == adata.n_obs


def test_backend_runs_ordered_soft_gating_then_neural_network(tmp_path: Path):
    adata, lineage = _toy_data(30)
    plan = AnnotationPlan(
        plan_id="nn-plan",
        dataset_id="test-data",
        lineage_revision="test-lineage",
        workflow="soft_gating -> neural_network",
        stages=[
            PipelineStage(kind="soft_gating", parameters={"unknown_thresh": 0.2}),
            PipelineStage(
                kind="neural_network",
                parameters={
                    "reference_quantile": 0.7,
                    "hidden_dims": [8],
                    "epochs": 1,
                    "batch_size": 16,
                    "seed": 1,
                },
            ),
        ],
        approved=True,
    )

    result = CytoGaterBackend().run(adata, lineage, plan, tmp_path)

    assert result["final_label_column"] == "nn_label_filled"
    assert sum(result["label_counts"].values()) == adata.n_obs


def test_backend_runs_hierarchical_knn_then_rf_rescue(tmp_path: Path):
    adata, lineage = _toy_data(30)
    plan = AnnotationPlan(
        plan_id="hier-rf-plan",
        dataset_id="test-data",
        lineage_revision="test-lineage",
        workflow="soft_gating -> hierarchical_knn -> rf_rescue",
        stages=[
            PipelineStage(kind="soft_gating", parameters={"unknown_thresh": 0.2}),
            PipelineStage(
                kind="hierarchical_knn",
                parameters={
                    "reference_quantile": 0.7,
                    "reference_cv_folds": 2,
                    "reference_repeats": 1,
                    "reference_agreement_thresh": 0.0,
                    "prediction_threshold": 0.0,
                    "prediction_agreement": 0.0,
                    "prediction_repeats": 1,
                    "distance_methods": ["pearson"],
                    "k": 3,
                    "seed": 1,
                },
            ),
            PipelineStage(
                kind="rf_rescue",
                parameters={
                    "num_trees": 10,
                    "cv_folds": 2,
                    "repeats": 1,
                    "agreement_thresh": 0.0,
                    "num_threads": 1,
                    "seed": 1,
                },
            ),
        ],
        approved=True,
    )

    result = CytoGaterBackend().run(adata, lineage, plan, tmp_path)

    assert result["ordered_stages"] == [
        "soft_gating",
        "hierarchical_knn",
        "rf_rescue",
    ]
    assert result["final_label_column"] == "final_label"
    assert sum(result["label_counts"].values()) == adata.n_obs


def test_random_forest_training_and_repeated_prediction_are_separate(tmp_path: Path):
    adata, lineage = _toy_data(40)
    initial = AnnotationPlan(
        plan_id="core-for-rf",
        dataset_id="test-data",
        lineage_revision="test-lineage",
        workflow="soft_gating -> select_core_cells",
        stages=[
            PipelineStage(kind="soft_gating", parameters={"unknown_thresh": 0.2}),
            PipelineStage(
                kind="select_core_cells",
                parameters={"cutoff_strategy": "quantile", "probability": 0.7},
            ),
        ],
        approved=True,
    )
    core = CytoGaterBackend().run(adata, lineage, initial, tmp_path / "core")
    training = AnnotationPlan(
        plan_id="train-rf",
        dataset_id="test-data",
        lineage_revision="test-lineage",
        workflow="train_random_forest",
        stages=[
            PipelineStage(
                kind="train_random_forest",
                parameters={
                    "num_trees": 10,
                    "cv_folds": 2,
                    "repeats": 1,
                    "agreement_thresh": 0.0,
                    "num_threads": 1,
                    "seed": 1,
                },
            )
        ],
        input_path=core["output_path"],
        approved=True,
    )
    trained = CytoGaterBackend().run(adata, lineage, training, tmp_path / "training")
    high_plan = AnnotationPlan(
        plan_id="predict-rf-high",
        dataset_id="test-data",
        lineage_revision="test-lineage",
        workflow="predict_random_forest",
        stages=[
            PipelineStage(
                kind="predict_random_forest",
                parameters={"prediction_threshold": 0.95},
            )
        ],
        input_path=trained["output_path"],
        approved=True,
    )
    high = CytoGaterBackend().run(adata, lineage, high_plan, tmp_path / "high")
    low_plan = high_plan.model_copy(
        update={
            "plan_id": "predict-rf-low",
            "input_path": high["output_path"],
            "stages": [
                PipelineStage(
                    kind="predict_random_forest",
                    parameters={"prediction_threshold": 0.1},
                )
            ],
        }
    )
    low = CytoGaterBackend().run(adata, lineage, low_plan, tmp_path / "low")

    assert trained["final_label_column"] == "cleaned_core_label"
    assert Path(trained["model_artifacts"]["random_forest"]).exists()
    assert high["final_label_column"] == "rf_label_filled"
    assert low["label_counts"].get("Unassigned", 0) <= high["label_counts"].get(
        "Unassigned", 0
    )
    assert high["prediction_summary"] is not None
    assert high["confidence_summary"]["column"] == "rf_confidence"
    assert low["prediction_summary"]["unassigned"] <= high["prediction_summary"][
        "unassigned"
    ]
    assert trained["core_counts_path"] and trained["core_plot_path"]


@pytest.mark.parametrize(
    ("method", "train_parameters", "predict_parameters", "expected_column"),
    [
        (
            "weighted_knn",
            {"cv_folds": 2, "repeats": 1, "agreement_thresh": 0.0, "k": 3, "seed": 1},
            {"prediction_threshold": 0.0, "k": 3, "method": "pearson"},
            "knn_label_filled",
        ),
        (
            "neural_network",
            {"hidden_dims": [8], "epochs": 1, "batch_size": 16, "seed": 1},
            {"prediction_threshold": 0.0},
            "nn_label_filled",
        ),
        (
            "hierarchical_knn",
            {
                "reference_cv_folds": 2,
                "reference_repeats": 1,
                "reference_agreement_thresh": 0.0,
                "top_n": 2,
                "k": 3,
                "seed": 1,
            },
            {
                "prediction_threshold": 0.0,
                "prediction_agreement": 0.0,
                "prediction_repeats": 1,
                "distance_methods": ["pearson"],
                "k": 3,
            },
            "hierarchical_label",
        ),
    ],
)
def test_other_model_families_support_separate_training_and_prediction(
    tmp_path: Path, method, train_parameters, predict_parameters, expected_column
):
    adata, lineage = _toy_data(30)
    initial = AnnotationPlan(
        plan_id=f"core-{method}",
        dataset_id="test-data",
        lineage_revision="test-lineage",
        workflow="soft_gating -> select_core_cells",
        stages=[
            PipelineStage(kind="soft_gating", parameters={"unknown_thresh": 0.2}),
            PipelineStage(
                kind="select_core_cells",
                parameters={"cutoff_strategy": "quantile", "probability": 0.7},
            ),
        ],
        approved=True,
    )
    core = CytoGaterBackend().run(adata, lineage, initial, tmp_path / "core")
    training = AnnotationPlan(
        plan_id=f"train-{method}",
        dataset_id="test-data",
        lineage_revision="test-lineage",
        workflow=f"train_{method}",
        stages=[PipelineStage(kind=f"train_{method}", parameters=train_parameters)],
        input_path=core["output_path"],
        approved=True,
    )
    trained = CytoGaterBackend().run(adata, lineage, training, tmp_path / "train")
    prediction = AnnotationPlan(
        plan_id=f"predict-{method}",
        dataset_id="test-data",
        lineage_revision="test-lineage",
        workflow=f"predict_{method}",
        stages=[PipelineStage(kind=f"predict_{method}", parameters=predict_parameters)],
        input_path=trained["output_path"],
        approved=True,
    )

    result = CytoGaterBackend().run(adata, lineage, prediction, tmp_path / "predict")

    assert Path(trained["model_artifacts"][method]).exists()
    assert result["final_label_column"] == expected_column
    assert sum(result["label_counts"].values()) == adata.n_obs
    assert result["core_counts_path"] is None
    assert result["core_plot_path"] is None
    if method == "neural_network":
        saved = joblib.load(trained["model_artifacts"][method])
        assert saved["model"]["device"] == "cpu"
        assert str(next(saved["model"]["net"].parameters()).device) == "cpu"

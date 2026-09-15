import pytest

from proteonavigator_lite.models import PipelineStage
from proteonavigator_lite.services import (
    STAGE_DEFAULTS,
    _validate_parameter_choices,
    _validate_prediction_models,
    _validate_stage_order,
)


@pytest.mark.parametrize(
    "kinds",
    [
        ["tree_gating"],
        ["soft_gating", "weighted_knn"],
        ["tree_gating", "random_forest"],
        ["tree_gating", "hierarchical_knn"],
        ["soft_gating", "neural_network"],
        ["tree_gating", "hierarchical_knn", "rf_rescue"],
        ["tree_gating", "calculate_uncertainty"],
        ["tree_gating", "select_core_cells"],
        ["tree_gating", "select_core_cells", "random_forest"],
        ["tree_gating", "select_core_cells", "train_random_forest"],
        ["tree_gating", "select_core_cells", "train_random_forest", "predict_random_forest"],
        ["soft_gating", "weighted_knn", "calculate_uncertainty"],
        ["tree_gating", "hierarchical_knn", "rf_rescue", "calculate_uncertainty"],
    ],
)
def test_supported_stage_orders(kinds):
    _validate_stage_order([PipelineStage(kind=kind) for kind in kinds])


@pytest.mark.parametrize(
    "kinds",
    [
        ["random_forest"],
        ["tree_gating", "soft_gating"],
        ["tree_gating", "rf_rescue"],
        ["tree_gating", "random_forest", "weighted_knn"],
        ["tree_gating", "rf_rescue", "hierarchical_knn"],
        ["calculate_uncertainty", "tree_gating"],
        ["tree_gating", "calculate_uncertainty", "random_forest"],
        ["tree_gating", "calculate_uncertainty", "calculate_uncertainty"],
        ["tree_gating", "train_random_forest", "predict_weighted_knn"],
    ],
)
def test_invalid_stage_orders_are_rejected(kinds):
    with pytest.raises(ValueError):
        _validate_stage_order([PipelineStage(kind=kind) for kind in kinds])


@pytest.mark.parametrize(
    "kinds",
    [
        ["random_forest"],
        ["weighted_knn", "calculate_uncertainty"],
        ["calculate_uncertainty"],
        ["hierarchical_knn", "rf_rescue"],
        ["select_core_cells"],
        ["select_core_cells", "random_forest"],
        ["train_random_forest"],
        ["train_random_forest", "predict_random_forest"],
        ["predict_random_forest"],
    ],
)
def test_supported_continuation_orders(kinds):
    _validate_stage_order(
        [PipelineStage(kind=kind) for kind in kinds], continuation=True
    )


def test_every_stage_has_explicit_defaults():
    assert set(STAGE_DEFAULTS) == {
        "soft_gating",
        "tree_gating",
        "weighted_knn",
        "random_forest",
        "hierarchical_knn",
        "neural_network",
        "rf_rescue",
        "calculate_uncertainty",
        "select_core_cells",
        "train_weighted_knn",
        "predict_weighted_knn",
        "train_random_forest",
        "predict_random_forest",
        "train_hierarchical_knn",
        "predict_hierarchical_knn",
        "train_neural_network",
        "predict_neural_network",
    }


def test_invalid_cutoff_method_is_rejected():
    with pytest.raises(ValueError, match="equal_posteriors"):
        _validate_parameter_choices(
            [PipelineStage(kind="tree_gating", parameters={"cutoff_method": "median"})]
        )


def test_prediction_without_matching_model_requires_training():
    with pytest.raises(ValueError, match="train_neural_network followed by predict_neural_network"):
        _validate_prediction_models(
            [PipelineStage(kind="predict_neural_network")], {}
        )


def test_training_and_prediction_in_same_plan_needs_no_existing_model():
    _validate_prediction_models(
        [
            PipelineStage(kind="train_neural_network"),
            PipelineStage(kind="predict_neural_network"),
        ],
        {},
    )


def test_prediction_accepts_matching_saved_model(tmp_path):
    artifact = tmp_path / "neural_network.joblib"
    artifact.touch()
    _validate_prediction_models(
        [PipelineStage(kind="predict_neural_network")],
        {"neural_network": str(artifact)},
    )

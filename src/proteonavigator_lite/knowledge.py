"""Agent-facing knowledge about the installed cytoGater package."""

from __future__ import annotations

import inspect
from importlib.metadata import version
from typing import Any

import cytogater


STAGE_DESCRIPTIONS = {
    "soft_gating": "Marker-signature scoring with an unknown threshold.",
    "tree_gating": "Recursive marker-priority gating using two-component GMM cutoffs.",
    "weighted_knn": "Weighted kNN trained from high-confidence gating references.",
    "random_forest": "Random forest trained from high-confidence gating references.",
    "hierarchical_knn": "Recursive kNN classification along a learned cell-type hierarchy.",
    "neural_network": "Feed-forward neural network trained from gating references.",
    "rf_rescue": (
        "Legacy combined follow-up: keep hierarchical-kNN assignments, train RF from "
        "hierarchy-assigned cells, and attempt to label only hierarchy-Unassigned cells. "
        "Because those training labels are predictions, errors can propagate; prefer "
        "explicit training and prediction stages when inspection is important."
    ),
    "calculate_uncertainty": "Entropy, Gini, margin, and spatial-discordance metrics.",
    "select_core_cells": "Select unambiguous high-probability gating references for training.",
    "train_weighted_knn": "Clean candidate core labels by repeated CV and fit a reusable weighted-kNN reference.",
    "predict_weighted_knn": "Predict non-core cells with a saved weighted-kNN reference.",
    "train_random_forest": "Clean candidate core labels by repeated CV and fit a reusable random forest.",
    "predict_random_forest": "Predict non-core cells with a saved random forest.",
    "train_hierarchical_knn": "Clean core labels and build a reusable hierarchy and node references.",
    "predict_hierarchical_knn": "Traverse a saved hierarchy to annotate cells recursively.",
    "train_neural_network": "Fit a reusable neural network from candidate core labels.",
    "predict_neural_network": "Predict non-core cells with a saved neural network.",
}


PARAMETER_GUIDANCE: dict[str, dict[str, dict[str, Any]]] = {
    "tree_gating": {
        "cutoff_method": {
            "choices": ["mean", "equal_posteriors"],
            "description": (
                "How the cutoff between the two fitted Gaussian components is chosen: "
                "midpoint of component means, or the equal-posterior intersection."
            ),
        },
        "max_depth": {"range": "integer >= 1", "description": "Maximum marker-tree depth."},
        "min_cells": {"range": "integer >= 1", "description": "Minimum cells required to split a node."},
        "min_score": {"range": "number >= 0", "description": "Minimum marker separation score."},
        "uncert_thresh": {"range": "0 to 1", "description": "Minimum best gating score for a hard label."},
        "neg_strength": {"range": "0 to 1", "description": "Penalty strength for expressed negative markers."},
        "workers": {"range": "integer >= 1", "description": "Accepted for R API parity; current compact Python tree implementation is serial."},
    },
    "soft_gating": {
        "unknown_thresh": {"range": "0 to 1", "description": "Minimum soft score required to avoid Unknown."},
    },
    "weighted_knn": {
        "method": {"choices": ["pearson", "spearman", "cosine"], "description": "Cell-similarity method."},
        "k": {"range": "integer >= 1", "description": "Number of nearest neighbours."},
        "prediction_threshold": {"range": "0 to 1", "description": "Minimum winning probability for assignment."},
    },
    "random_forest": {
        "num_trees": {"range": "integer >= 1", "description": "Number of trees in the final forest."},
        "prediction_threshold": {"range": "0 to 1", "description": "Minimum class probability for assignment."},
        "num_threads": {"range": "integer >= 1", "description": "Parallel sklearn workers."},
    },
    "hierarchical_knn": {
        "distance_methods": {"choices": ["pearson", "spearman", "cosine"], "description": "Similarity methods used by the recursive ensemble."},
        "k": {"range": "integer >= 1", "description": "Neighbours per recursive prediction task."},
        "prediction_threshold": {"range": "0 to 1", "description": "Minimum average node probability."},
        "prediction_agreement": {"range": "0 to 1", "description": "Minimum ensemble vote agreement."},
    },
    "neural_network": {
        "hidden_dims": {"range": "list of positive integers", "description": "Hidden-layer widths."},
        "dropout": {"range": "0 to <1", "description": "Dropout fraction."},
        "epochs": {"range": "integer >= 1", "description": "Maximum training epochs."},
        "learning_rate": {"range": "number > 0", "description": "Optimizer learning rate."},
        "batch_size": {"range": "integer >= 1", "description": "Training batch size."},
        "prediction_threshold": {"range": "0 to 1", "description": "Minimum class probability."},
    },
    "rf_rescue": {
        "num_trees": {"range": "integer >= 1", "description": "Number of rescue-forest trees."},
        "prediction_threshold": {"range": "0 to 1", "description": "Minimum rescue probability."},
        "num_threads": {"range": "integer >= 1", "description": "Parallel sklearn workers."},
    },
    "calculate_uncertainty": {
        "k_spatial": {"range": "integer >= 1", "description": "Spatial neighbours used for discordance."},
    },
    "select_core_cells": {
        "cutoff_strategy": {
            "choices": ["quantile", "mad"],
            "description": "Per-cell-type probability cutoff rule.",
        },
        "probability": {
            "range": "0 to 1",
            "description": "Quantile used when cutoff_strategy is quantile.",
        },
        "mad_multiplier": {
            "range": "number >= 0",
            "description": "MAD multiplier used when cutoff_strategy is mad.",
        },
        "unknown_label": {
            "description": "Label assigned when zero or multiple cell types pass their cutoffs.",
        },
    },
    "train_weighted_knn": {
        "method": {"choices": ["pearson", "spearman", "cosine"], "description": "Training similarity method."},
        "agreement_thresh": {"range": "0 to 1", "description": "Minimum repeated-CV agreement for cleaned core cells."},
        "training_label_column": {"description": "obs column used as the training/core labels."},
        "exclude_training_labels": {"description": "Label values excluded from model training."},
    },
    "predict_weighted_knn": {
        "method": {"choices": ["pearson", "spearman", "cosine"], "description": "Prediction similarity method."},
        "prediction_threshold": {"range": "0 to 1", "description": "Minimum assignment probability."},
        "prediction_label_column": {"description": "obs column whose labels are preserved or replaced."},
        "target_labels": {"description": "Only cells with these labels are eligible for prediction."},
    },
    "train_random_forest": {
        "agreement_thresh": {"range": "0 to 1", "description": "Minimum repeated-CV agreement for cleaned core cells."},
        "num_trees": {"range": "integer >= 1", "description": "Trees in the fitted reusable forest."},
        "training_label_column": {"description": "obs column used as the training/core labels."},
        "exclude_training_labels": {"description": "Label values excluded from model training."},
    },
    "predict_random_forest": {
        "prediction_threshold": {"range": "0 to 1", "description": "Minimum assignment probability."},
        "prediction_label_column": {"description": "obs column whose labels are preserved or replaced."},
        "target_labels": {"description": "Only cells with these labels are eligible for prediction."},
    },
    "train_hierarchical_knn": {
        "reference_method": {"choices": ["pearson", "spearman", "cosine"], "description": "Similarity method for cleaning references."},
        "reference_agreement_thresh": {"range": "0 to 1", "description": "Minimum CV agreement for cleaned core cells."},
        "training_label_column": {"description": "obs column used as the training/core labels."},
        "exclude_training_labels": {"description": "Label values excluded from model training."},
    },
    "predict_hierarchical_knn": {
        "distance_methods": {"choices": ["pearson", "spearman", "cosine"], "description": "Recursive ensemble methods."},
        "prediction_threshold": {"range": "0 to 1", "description": "Minimum node probability."},
        "prediction_agreement": {"range": "0 to 1", "description": "Minimum recursive vote agreement."},
        "prediction_label_column": {"description": "obs column whose labels are preserved or replaced."},
        "target_labels": {"description": "Only cells with these labels are eligible for prediction."},
    },
    "train_neural_network": {
        "agreement_thresh": {"range": "0 to 1", "description": "Reserved for cytoGater training compatibility."},
        "training_label_column": {"description": "obs column used as the training/core labels."},
        "exclude_training_labels": {"description": "Label values excluded from model training."},
    },
    "predict_neural_network": {
        "prediction_threshold": {"range": "0 to 1", "description": "Minimum assignment probability."},
        "prediction_label_column": {"description": "obs column whose labels are preserved or replaced."},
        "target_labels": {"description": "Only cells with these labels are eligible for prediction."},
    },
}


def describe_options(defaults: dict[str, dict[str, Any]], stage: str | None = None,
                     parameter: str | None = None) -> dict[str, Any]:
    """Describe workflows and configured agent parameters."""
    if stage is None:
        return {
            name: {"description": STAGE_DESCRIPTIONS[name], "defaults": values}
            for name, values in defaults.items()
        }
    if stage not in defaults:
        raise ValueError(f"Unknown stage {stage!r}; choose from {sorted(defaults)}")
    guidance = PARAMETER_GUIDANCE.get(stage, {})
    if parameter is not None:
        if parameter not in defaults[stage]:
            raise ValueError(
                f"Unknown {stage} parameter {parameter!r}; choose from {sorted(defaults[stage])}"
            )
        return {
            "stage": stage,
            "parameter": parameter,
            "default": defaults[stage][parameter],
            **guidance.get(parameter, {"description": "See the live cytoGater API signature."}),
        }
    return {
        "stage": stage,
        "description": STAGE_DESCRIPTIONS[stage],
        "parameters": {
            name: {"default": value, **guidance.get(name, {})}
            for name, value in defaults[stage].items()
        },
    }


def inspect_public_api(function_name: str) -> dict[str, Any]:
    """Inspect a callable from the installed package's declared public API."""
    public = set(getattr(cytogater, "__all__", []))
    if function_name not in public or not hasattr(cytogater, function_name):
        matches = sorted(name for name in public if function_name.lower() in name.lower())
        raise ValueError(f"Unknown public function {function_name!r}. Close matches: {matches}")
    function = getattr(cytogater, function_name)
    if not callable(function):
        raise ValueError(f"{function_name!r} is public but is not callable")
    return {
        "package_version": version("cytogater"),
        "function": function_name,
        "module": function.__module__,
        "signature": str(inspect.signature(function)),
        "documentation": inspect.getdoc(function) or "No docstring is currently available.",
    }


def list_public_api(query: str = "") -> dict[str, Any]:
    """List matching installed public callables with their signatures."""
    names = sorted(getattr(cytogater, "__all__", []))
    if query:
        names = [name for name in names if query.lower() in name.lower()]
    functions = {}
    for name in names:
        value = getattr(cytogater, name, None)
        if callable(value):
            functions[name] = str(inspect.signature(value))
    return {"package_version": version("cytogater"), "functions": functions}

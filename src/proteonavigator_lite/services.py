"""Session-scoped services used by the Chainlit UI and agent tools."""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path
from typing import Any

import pandas as pd

from .backend import CytoGaterBackend
from .data import (
    lineage_from_definitions,
    lineage_revision,
    load_dataset,
    load_lineage_table,
    validate_lineage,
)
from .models import AnnotationPlan, DatasetRecord, PipelineStage, SessionWorkspace
from .knowledge import PARAMETER_GUIDANCE


def requested_tree_cell_type(text: str, available: list[str]) -> str | None:
    """Resolve an explicit natural-language tree-plot request."""
    lowered = text.casefold()
    plot_words = ("plot", "show", "draw", "view")
    if "tree" not in lowered or not any(word in lowered for word in plot_words):
        return None
    matches = [name for name in available if name.casefold() in lowered]
    return max(matches, key=len) if matches else None


STAGE_DEFAULTS: dict[str, dict[str, Any]] = {
    "soft_gating": {"unknown_thresh": 0.4},
    "tree_gating": {
        "max_depth": 4,
        "min_cells": 200,
        "min_score": 0.5,
        "uncert_thresh": 0.25,
        "neg_strength": 0.4,
        "cutoff_method": "mean",
        "workers": 1,
    },
    "weighted_knn": {
        "reference_quantile": 0.98,
        "cv_folds": 5,
        "repeats": 10,
        "agreement_thresh": 0.8,
        "k": 5,
        "method": "pearson",
        "prediction_threshold": 0.6,
        "seed": 0,
    },
    "random_forest": {
        "reference_quantile": 0.98,
        "num_trees": 200,
        "cv_folds": 5,
        "repeats": 10,
        "agreement_thresh": 0.8,
        "prediction_threshold": 0.5,
        "num_threads": 2,
        "seed": 0,
    },
    "hierarchical_knn": {
        "reference_quantile": 0.98,
        "reference_cv_folds": 5,
        "reference_repeats": 10,
        "reference_agreement_thresh": 0.8,
        "reference_method": "pearson",
        "top_n": 5,
        "k": 5,
        "prediction_threshold": 0.7,
        "prediction_agreement": 0.8,
        "prediction_repeats": 5,
        "distance_methods": ["pearson", "cosine"],
        "seed": 0,
    },
    "neural_network": {
        "reference_quantile": 0.98,
        "hidden_dims": [64, 32],
        "dropout": 0.3,
        "epochs": 50,
        "learning_rate": 0.001,
        "batch_size": 256,
        "prediction_threshold": 0.5,
        "seed": 0,
    },
    "rf_rescue": {
        "num_trees": 200,
        "cv_folds": 5,
        "repeats": 10,
        "agreement_thresh": 0.8,
        "prediction_threshold": 0.5,
        "num_threads": 2,
        "seed": 0,
    },
    "calculate_uncertainty": {"k_spatial": 15},
    "select_core_cells": {
        "cutoff_strategy": "quantile",
        "probability": 0.98,
        "mad_multiplier": 3.0,
        "unknown_label": "Unknown",
    },
    "train_weighted_knn": {
        "cv_folds": 5, "repeats": 10, "agreement_thresh": 0.8,
        "k": 5, "method": "pearson", "seed": 0, "training_chunk_size": 250,
        "training_label_column": "cutoff_label",
        "exclude_training_labels": ["Unknown", "Unassigned", "Uncertain"],
    },
    "predict_weighted_knn": {
        "prediction_threshold": 0.6, "k": 5, "method": "pearson",
        "prediction_chunk_size": 10000, "prediction_label_column": "cleaned_core_label",
        "target_labels": ["Unknown", "Unassigned"],
    },
    "train_random_forest": {
        "num_trees": 200, "cv_folds": 5, "repeats": 10,
        "agreement_thresh": 0.8, "num_threads": 2, "seed": 0,
        "training_label_column": "cutoff_label",
        "exclude_training_labels": ["Unknown", "Unassigned", "Uncertain"],
    },
    "predict_random_forest": {
        "prediction_threshold": 0.5, "prediction_label_column": "cleaned_core_label",
        "target_labels": ["Unknown", "Unassigned"],
    },
    "train_hierarchical_knn": {
        "reference_cv_folds": 5, "reference_repeats": 10,
        "reference_agreement_thresh": 0.8, "reference_method": "pearson",
        "top_n": 5, "k": 5, "seed": 0,
        "training_label_column": "cutoff_label",
        "exclude_training_labels": ["Unknown", "Unassigned", "Uncertain"],
    },
    "predict_hierarchical_knn": {
        "prediction_threshold": 0.7, "prediction_agreement": 0.8,
        "k": 5, "prediction_repeats": 5,
        "distance_methods": ["pearson", "cosine"], "prediction_chunk_size": 1000,
        "prediction_label_column": "cleaned_core_label",
        "target_labels": ["Unknown", "Unassigned"],
    },
    "train_neural_network": {
        "hidden_dims": [64, 32], "dropout": 0.3, "epochs": 50,
        "learning_rate": 0.001, "batch_size": 256, "seed": 0,
        "training_label_column": "cutoff_label",
        "exclude_training_labels": ["Unknown", "Unassigned", "Uncertain"],
    },
    "predict_neural_network": {
        "prediction_threshold": 0.5, "prediction_label_column": "cleaned_core_label",
        "target_labels": ["Unknown", "Unassigned"],
    },
}


class WorkspaceService:
    """Manage one user's data, lineage revisions, plans, and outputs."""

    def __init__(self, workspace: SessionWorkspace):
        self.workspace = workspace
        self.backend = CytoGaterBackend()

    def ingest_dataset(
        self, source_path: str, original_name: str | None = None
    ) -> dict[str, Any]:
        source = Path(source_path)
        stored = self._copy_input(source, original_name)
        dataset_id, adata, assay = load_dataset(stored, self.workspace.root)
        self.workspace.dataset = DatasetRecord(
            dataset_id=dataset_id,
            name=original_name or source.name,
            source_path=stored,
            adata=adata,
            assay_name=assay,
        )
        return self.dataset_summary()

    def ingest_lineage(
        self, source_path: str, original_name: str | None = None
    ) -> dict[str, Any]:
        self._require_dataset()
        stored = self._copy_input(Path(source_path), original_name)
        lineage = load_lineage_table(stored)
        return self.save_lineage(lineage)

    def save_lineage_definitions(self, definitions: list[dict]) -> dict[str, Any]:
        return self.save_lineage(lineage_from_definitions(definitions))

    def update_lineage_cell_type(
        self, cell_type: str, pos_markers: list[str], neg_markers: list[str]
    ) -> dict[str, Any]:
        """Replace or add one definition while preserving every other cell type."""
        if self.workspace.lineage is None:
            raise ValueError("Load or create a lineage table before editing one cell type")
        lineage = self.workspace.lineage.copy(deep=True)
        matches = lineage["cell_type"].astype(str).str.casefold() == cell_type.casefold()
        definition = {
            "cell_type": cell_type,
            "pos_markers": list(dict.fromkeys(map(str, pos_markers))),
            "neg_markers": list(dict.fromkeys(map(str, neg_markers))),
        }
        if matches.any():
            index = lineage.index[matches][0]
            definition["cell_type"] = str(lineage.at[index, "cell_type"])
            for key, value in definition.items():
                lineage.at[index, key] = value
            action = "updated"
        else:
            lineage = pd.concat([lineage, pd.DataFrame([definition])], ignore_index=True)
            action = "added"
        report = validate_lineage(
            lineage, list(map(str, self._require_dataset().adata.var_names))
        )
        if not report["valid"]:
            raise ValueError("The lineage edit is invalid: " + "; ".join(report["errors"]))
        result = self.save_lineage(lineage)
        result.update({"action": action, "changed_cell_type": definition})
        return result

    def save_lineage(self, lineage: pd.DataFrame) -> dict[str, Any]:
        dataset = self._require_dataset()
        report = validate_lineage(lineage, list(map(str, dataset.adata.var_names)))
        revision = lineage_revision(lineage)
        revisions = self.workspace.root / "lineage"
        revisions.mkdir(exist_ok=True)
        lineage.to_json(revisions / f"{revision}.json", orient="records", indent=2)
        self.workspace.lineage = lineage
        self.workspace.lineage_revision = revision
        return {"revision": revision, **report, "definitions": lineage.to_dict("records")}

    def dataset_summary(self) -> dict[str, Any]:
        dataset = self._require_dataset()
        adata = dataset.adata
        return {
            "dataset_id": dataset.dataset_id,
            "name": dataset.name,
            "n_cells": adata.n_obs,
            "n_markers": adata.n_vars,
            "markers": list(map(str, adata.var_names)),
            "layers": list(map(str, adata.layers.keys())),
            "has_spatial": "spatial" in adata.obsm,
            "assay_name": dataset.assay_name,
            "has_previous_output": self.workspace.last_output is not None,
            "previous_output": str(self.workspace.last_output) if self.workspace.last_output else None,
        }

    def current_lineage(self) -> dict[str, Any]:
        dataset = self._require_dataset()
        if self.workspace.lineage is None:
            return {
                "loaded": False,
                "message": "No lineage table is loaded. Propose one using only panel markers.",
                "panel_markers": list(map(str, dataset.adata.var_names)),
            }
        report = validate_lineage(
            self.workspace.lineage, list(map(str, dataset.adata.var_names))
        )
        return {
            "loaded": True,
            "revision": self.workspace.lineage_revision,
            **report,
            "definitions": self.workspace.lineage.to_dict("records"),
        }

    def create_plan(self, stages: list[dict[str, Any]]) -> dict[str, Any]:
        dataset = self._require_dataset()
        if self.workspace.lineage is None or self.workspace.lineage_revision is None:
            raise ValueError("Load or create a lineage table before planning annotation")
        validation = validate_lineage(
            self.workspace.lineage, list(map(str, dataset.adata.var_names))
        )
        if not validation["valid"]:
            raise ValueError("The lineage table is invalid: " + "; ".join(validation["errors"]))
        pipeline = [PipelineStage.model_validate(stage) for stage in stages]
        if not pipeline:
            raise ValueError("An annotation plan needs at least one stage")
        continuation = pipeline[0].kind not in {"soft_gating", "tree_gating"}
        _validate_stage_order(pipeline, continuation=continuation)
        _validate_parameter_choices(pipeline)
        _validate_prediction_models(
            pipeline, (self.workspace.last_result or {}).get("model_artifacts", {})
        )
        input_path = None
        if continuation:
            if self.workspace.last_output is None or not self.workspace.last_output.exists():
                raise ValueError(
                    "This stage needs a completed earlier output in the current session"
                )
            input_path = str(self.workspace.last_output)
            previous_revision = (self.workspace.last_result or {}).get("lineage_revision")
            if previous_revision is None:
                previous_plan_id = self.workspace.last_output.stem.removeprefix("annotated_")
                previous_plan = self.workspace.plans.get(previous_plan_id)
                previous_revision = (
                    previous_plan.lineage_revision if previous_plan is not None else None
                )
            if previous_revision and previous_revision != self.workspace.lineage_revision:
                raise ValueError(
                    "The lineage table changed after the latest pipeline output. Start a new "
                    "analysis with soft_gating or tree_gating; do not continue models trained "
                    "under the previous lineage revision. Historical outputs are preserved."
                )
        if any(stage.kind == "calculate_uncertainty" for stage in pipeline):
            if "spatial" not in dataset.adata.obsm:
                raise ValueError(
                    "calculate_uncertainty requires spatial coordinates in obsm['spatial']"
                )
        pipeline = [
            PipelineStage(
                kind=stage.kind,
                parameters={**STAGE_DEFAULTS[stage.kind], **stage.parameters},
            )
            for stage in pipeline
        ]
        warnings = list(validation["warnings"])
        workflow = " -> ".join(stage.kind for stage in pipeline)
        plan_id = uuid.uuid4().hex[:10]
        plan = AnnotationPlan(
            plan_id=plan_id,
            dataset_id=dataset.dataset_id,
            lineage_revision=self.workspace.lineage_revision,
            workflow=workflow,
            assay_name=dataset.assay_name,
            stages=pipeline,
            input_path=input_path,
            warnings=warnings,
        )
        self.workspace.plans[plan_id] = plan
        plan_dir = self.workspace.root / "plans"
        plan_dir.mkdir(exist_ok=True)
        (plan_dir / f"{plan_id}.json").write_text(plan.model_dump_json(indent=2))
        return plan.model_dump()

    def approve_plan(self, plan_id: str) -> AnnotationPlan:
        plan = self.workspace.plans.get(plan_id)
        if plan is None:
            raise KeyError(f"Plan {plan_id!r} was not found in this session")
        plan.approved = True
        return plan

    def run_plan(self, plan_id: str) -> dict[str, Any]:
        dataset = self._require_dataset()
        plan = self.workspace.plans.get(plan_id)
        if plan is None:
            raise KeyError(f"Plan {plan_id!r} was not found in this session")
        if self.workspace.lineage is None:
            raise RuntimeError("The approved lineage revision is unavailable")
        status_dir = self.workspace.root / "runs"
        status_dir.mkdir(parents=True, exist_ok=True)
        status_path = status_dir / f"{plan_id}.status"
        self.workspace.run_status[plan_id] = "starting"
        status_path.write_text("starting")

        def progress(message: str) -> None:
            self.workspace.run_status[plan_id] = message
            status_path.write_text(message)

        output = self.backend.run(
            dataset.adata,
            self.workspace.lineage,
            plan,
            self.workspace.root / "outputs",
            progress=progress,
        )
        self.workspace.run_status[plan_id] = "completed"
        status_path.write_text("completed")
        self.workspace.last_output = Path(output["output_path"])
        manifest = {"plan": plan.model_dump(), "result": output}
        (self.workspace.root / "outputs" / f"manifest_{plan_id}.json").write_text(
            json.dumps(manifest, indent=2)
        )
        return output

    def record_completed_output(self, output: dict[str, Any]) -> None:
        """Synchronize worker results back into the parent chat session."""
        self.workspace.last_output = Path(output["output_path"])
        self.workspace.last_result = output

    def latest_result_summary(self) -> dict[str, Any]:
        if self.workspace.last_result is None:
            return {"completed": False, "message": "No pipeline has completed in this session."}
        result = self.workspace.last_result
        return {
            "completed": True,
            "output_path": result.get("output_path"),
            "ordered_stages": result.get("ordered_stages", []),
            "final_label_column": result.get("final_label_column"),
            "label_counts": result.get("label_counts", {}),
            "tree_cell_types": result.get("tree_cell_types", []),
            "uncertainty_probability_source": result.get("uncertainty_probability_source"),
            "core_counts_path": result.get("core_counts_path"),
            "model_artifacts": result.get("model_artifacts", {}),
            "prediction_summary": result.get("prediction_summary"),
            "confidence_summary": result.get("confidence_summary"),
            "available_label_columns": result.get("available_label_columns", []),
        }

    def create_tree_plot(self, cell_type: str) -> dict[str, Any]:
        """Render one stored tree-gating model for display by Chainlit."""
        result = self.workspace.last_result or {}
        tree_path = result.get("tree_path")
        if not tree_path or not Path(tree_path).exists():
            raise RuntimeError(
                "No tree-gating models are available. Complete a tree_gating stage first."
            )
        trees = json.loads(Path(tree_path).read_text(encoding="utf-8"))
        matches = {name.casefold(): name for name in trees}
        selected = matches.get(cell_type.casefold())
        if selected is None:
            raise ValueError(
                f"Unknown cell type {cell_type!r}; choose from {sorted(trees)}"
            )
        import matplotlib.pyplot as plt
        from cytogater import plot_celltype_tree

        figure, axis = plt.subplots(figsize=(10, 6))
        plot_celltype_tree(
            trees[selected], title=f"{selected} marker-priority tree", ax=axis
        )
        figure.tight_layout()
        path = self.workspace.root / "outputs" / f"tree_{_safe_filename(selected)}.png"
        figure.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(figure)
        item = {"kind": "tree_plot", "cell_type": selected, "path": str(path)}
        self.workspace.pending_visualizations.append(item)
        return item

    def consume_visualizations(self) -> list[dict[str, str]]:
        items = list(self.workspace.pending_visualizations)
        self.workspace.pending_visualizations.clear()
        return items

    def _copy_input(self, source: Path, original_name: str | None = None) -> Path:
        inputs = self.workspace.root / "inputs"
        inputs.mkdir(parents=True, exist_ok=True)
        # Chainlit commonly stores an uploaded file under a generated `.bin`
        # path. Preserve the browser-provided filename so format detection sees
        # the real `.h5ad`, `.csv`, or `.rds` suffix.
        safe_name = Path(original_name).name if original_name else source.name
        target = inputs / safe_name
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
        return target

    def _require_dataset(self) -> DatasetRecord:
        if self.workspace.dataset is None:
            raise RuntimeError("No dataset is loaded")
        return self.workspace.dataset


def _validate_stage_order(
    stages: list[PipelineStage], *, continuation: bool = False
) -> None:
    """Reject pipeline orders that cannot produce the references they consume."""
    if not stages:
        raise ValueError("An annotation plan needs at least one stage")
    uncertainty_positions = [
        index for index, stage in enumerate(stages) if stage.kind == "calculate_uncertainty"
    ]
    if uncertainty_positions and uncertainty_positions != [len(stages) - 1]:
        raise ValueError("calculate_uncertainty may appear only once, as the final stage")
    annotation_stages = stages[:-1] if uncertainty_positions else stages
    if continuation:
        kinds = tuple(stage.kind for stage in annotation_stages)
        if not _valid_annotation_sequence(kinds, allow_prediction_only=True):
            raise ValueError(
                "Invalid continuation order. Use optional select_core_cells, one train stage, "
                "its matching predict stage, a legacy combined method, and/or final uncertainty."
            )
        return
    gating = {"soft_gating", "tree_gating"}
    if annotation_stages[0].kind not in gating:
        raise ValueError("The first stage must be soft_gating or tree_gating")
    if any(stage.kind in gating for stage in annotation_stages[1:]):
        raise ValueError("Gating may appear only once, as the first stage")
    downstream = [stage.kind for stage in annotation_stages[1:]]
    if not _valid_annotation_sequence(tuple(downstream), allow_prediction_only=False):
        readable = " -> ".join(stage.kind for stage in stages)
        raise ValueError(
            f"Unsupported or scientifically invalid stage order: {readable}. "
            "Supported forms are gating alone; gating followed by weighted_knn, "
            "random_forest, hierarchical_knn, or neural_network; and gating followed "
            "by hierarchical_knn then rf_rescue."
        )


def _valid_annotation_sequence(
    kinds: tuple[str, ...], *, allow_prediction_only: bool
) -> bool:
    if not kinds:
        return True
    if kinds[0] == "select_core_cells":
        kinds = kinds[1:]
        if not kinds:
            return True
    legacy = {
        ("weighted_knn",), ("random_forest",), ("hierarchical_knn",),
        ("neural_network",), ("hierarchical_knn", "rf_rescue"),
    }
    if kinds in legacy:
        return True
    pairs = {
        "train_weighted_knn": "predict_weighted_knn",
        "train_random_forest": "predict_random_forest",
        "train_hierarchical_knn": "predict_hierarchical_knn",
        "train_neural_network": "predict_neural_network",
    }
    if len(kinds) == 1 and kinds[0] in pairs:
        return True
    if len(kinds) == 2 and pairs.get(kinds[0]) == kinds[1]:
        return True
    return allow_prediction_only and len(kinds) == 1 and kinds[0] in pairs.values()


def _validate_parameter_choices(stages: list[PipelineStage]) -> None:
    """Reject explicit categorical values that cytoGater does not implement."""
    for stage in stages:
        guidance = PARAMETER_GUIDANCE.get(stage.kind, {})
        for name, value in stage.parameters.items():
            choices = guidance.get(name, {}).get("choices")
            if choices is None:
                continue
            values = value if isinstance(value, list) else [value]
            invalid = [item for item in values if item not in choices]
            if invalid:
                raise ValueError(
                    f"Invalid {stage.kind}.{name}: {invalid}. Valid choices: {choices}"
                )


def _validate_prediction_models(
    stages: list[PipelineStage], model_artifacts: dict[str, str]
) -> None:
    """Require training in this plan or a matching reusable model artifact."""
    trained_in_plan: set[str] = set()
    for stage in stages:
        if stage.kind.startswith("train_"):
            trained_in_plan.add(stage.kind.removeprefix("train_"))
            continue
        if not stage.kind.startswith("predict_"):
            continue
        method = stage.kind.removeprefix("predict_")
        artifact = model_artifacts.get(method)
        artifact_exists = bool(artifact and Path(artifact).exists())
        if method not in trained_in_plan and not artifact_exists:
            raise ValueError(
                f"No trained {method} model is available. The requested scientific goal "
                f"requires training first. Create one complete ordered plan with "
                f"train_{method} followed by predict_{method}, preserving the requested "
                "prediction parameters."
            )


def _safe_filename(value: str) -> str:
    safe = "".join(character if character.isalnum() else "_" for character in value)
    return safe.strip("_") or "cell_type"

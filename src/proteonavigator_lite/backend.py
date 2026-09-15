"""Compatibility boundary around the evolving cytoGater package."""

from __future__ import annotations

import json
import io
import uuid
from importlib.metadata import version
from pathlib import Path
from typing import Any, Callable

import anndata as ad
import joblib
import numpy as np
import pandas as pd

import cytogater

from .models import AnnotationPlan, PipelineStage
from .visualization import spatial_payload


class CytoGaterBackend:
    """Execute stable agent plans against the currently installed cytoGater API."""

    def package_version(self) -> str:
        return version("cytogater")

    def run(
        self,
        adata: ad.AnnData,
        lineage: pd.DataFrame,
        plan: AnnotationPlan,
        output_dir: Path,
        progress: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        """Execute the currently supported workflow and write reproducible outputs."""
        if not plan.approved:
            raise PermissionError(f"Plan {plan.plan_id} has not been approved")
        working = ad.read_h5ad(plan.input_path) if plan.input_path else adata.copy()
        previous_labels = (
            working.obs["annotation_label"].astype(str).copy()
            if "annotation_label" in working.obs
            else None
        )
        notify = progress or (lambda message: None)
        uncertainty_stage = next(
            (stage for stage in plan.stages if stage.kind == "calculate_uncertainty"), None
        )
        annotation_stages = [
            stage for stage in plan.stages if stage.kind != "calculate_uncertainty"
        ]
        continuation = plan.input_path is not None
        if continuation:
            gating_probabilities = _stored_gating_probabilities(working)
            gating_result = {"spe": working, "prob_mat": gating_probabilities}
            if annotation_stages:
                synthetic_stages = [
                    PipelineStage(kind="tree_gating"),
                    *annotation_stages,
                ]
                annotation_plan = plan.model_copy(update={"stages": synthetic_stages})
                final_column, downstream_probabilities = self._run_downstream(
                    working,
                    gating_result,
                    annotation_plan,
                    notify,
                    total_stages=len(plan.stages),
                    stage_offset=-1,
                    artifact_dir=output_dir.parent / "models",
                )
            else:
                final_column = "annotation_label"
                downstream_probabilities = _stored_annotation_probabilities(working)
        else:
            annotation_plan = plan.model_copy(update={"stages": annotation_stages})
            notify(f"stage 1/{len(plan.stages)}: {annotation_stages[0].kind}")
            gating_result = self._run_gating(working, lineage, annotation_plan)
            final_column, downstream_probabilities = self._run_downstream(
                working, gating_result, annotation_plan, notify,
                total_stages=len(plan.stages), artifact_dir=output_dir.parent / "models"
            )
        selected_probabilities = (
            downstream_probabilities
            if downstream_probabilities is not None
            else gating_result["prob_mat"]
        )
        _store_annotation_probabilities(working, selected_probabilities)
        probability_source = None
        if uncertainty_stage is not None:
            notify(f"stage {len(plan.stages)}/{len(plan.stages)}: calculate_uncertainty")
            probability_source = self._store_annotation_uncertainty(
                working,
                selected_probabilities,
                "downstream" if downstream_probabilities is not None else "gating",
                k_spatial=int(uncertainty_stage.parameters.get("k_spatial", 15)),
            )
        labels = (
            working.obs[final_column]
            if final_column is not None
            else _result_labels(gating_result, working)
        )
        working.obs["annotation_label"] = pd.Categorical(labels.astype(str))
        notify("writing annotated AnnData and label counts")
        output_dir.mkdir(parents=True, exist_ok=True)
        prediction_only = bool(annotation_stages) and all(
            stage.kind.startswith("predict_") for stage in annotation_stages
        )
        if prediction_only:
            # Core membership is unchanged by prediction. Redrawing the same
            # matplotlib report wastes memory and has caused native-library
            # failures after otherwise successful GPU inference.
            core_counts_path, core_plot_path = None, None
        else:
            core_counts_path, core_plot_path = _write_core_reports(
                working, output_dir, plan.plan_id
            )
        tree_path = None
        tree_cell_types = []
        if not continuation and "trees" in gating_result:
            tree_cell_types = sorted(map(str, gating_result["trees"].keys()))
            tree_path = output_dir / f"gating_trees_{plan.plan_id}.json"
            tree_path.write_text(
                json.dumps(_json_safe(gating_result["trees"])), encoding="utf-8"
            )
            working.uns["cytogater_gating_tree_path"] = str(tree_path)
            working.uns["cytogater_gating_tree_cell_types"] = tree_cell_types
        elif continuation and working.uns.get("cytogater_gating_tree_path"):
            tree_path = Path(str(working.uns["cytogater_gating_tree_path"]))
            tree_cell_types = list(
                map(str, working.uns.get("cytogater_gating_tree_cell_types", []))
            )
        output_path = output_dir / f"annotated_{plan.plan_id}.h5ad"
        working.write_h5ad(output_path)
        counts = labels.value_counts(dropna=False).rename_axis("label").reset_index(name="cells")
        counts_path = output_dir / f"label_counts_{plan.plan_id}.csv"
        counts.to_csv(counts_path, index=False)
        prediction_summary = _prediction_summary(previous_labels, labels.astype(str))
        confidence_summary = _confidence_summary(working)
        plot_path = None
        payload = spatial_payload(
            working,
            title=f"Annotated spatial map · {plan.plan_id}",
            label_column="annotation_label",
        )
        if payload is not None:
            plot_path = output_dir / f"spatial_{plan.plan_id}.json"
            plot_path.write_text(json.dumps(payload), encoding="utf-8")
        return {
            "output_path": str(output_path),
            "counts_path": str(counts_path),
            "label_counts": dict(zip(counts["label"].astype(str), counts["cells"].astype(int))),
            "cytogater_version": self.package_version(),
            "final_label_column": final_column or "gating result",
            "ordered_stages": [stage.kind for stage in plan.stages],
            "spatial_plot_path": str(plot_path) if plot_path else None,
            "uncertainty_probability_source": probability_source,
            "tree_path": str(tree_path) if tree_path else None,
            "tree_cell_types": tree_cell_types,
            "core_counts_path": str(core_counts_path) if core_counts_path else None,
            "core_plot_path": str(core_plot_path) if core_plot_path else None,
            "model_artifacts": {
                str(key).removeprefix("cytogater_model_"): str(value)
                for key, value in working.uns.items()
                if str(key).startswith("cytogater_model_")
            },
            "prediction_summary": prediction_summary,
            "confidence_summary": confidence_summary,
            "available_label_columns": _available_label_columns(working),
            "lineage_revision": plan.lineage_revision,
        }

    @staticmethod
    def _store_annotation_uncertainty(
        adata: ad.AnnData,
        probability_matrix: pd.DataFrame,
        source: str,
        k_spatial: int = 15,
    ) -> str:
        probabilities = pd.DataFrame(probability_matrix).reindex(adata.obs_names)
        probabilities = probabilities.apply(pd.to_numeric, errors="coerce").fillna(0).clip(lower=0)
        row_sums = probabilities.sum(axis=1)
        zero = row_sums <= 0
        if zero.any():
            probabilities.loc[zero] = 1 / max(probabilities.shape[1], 1)
            row_sums = probabilities.sum(axis=1)
        probabilities = probabilities.div(row_sums, axis=0)
        adata.uns["cytogater_uncertainty_probability_source"] = source

        if "spatial" in adata.obsm and probabilities.shape[1] >= 2:
            sample_column = next(
                (name for name in ("sample_id", "sample", "Sample", "image_id", "imageid")
                 if name in adata.obs),
                None,
            )
            temporary_sample = sample_column is None
            if temporary_sample:
                sample_column = "_cytogater_spatial_sample"
                adata.obs[sample_column] = "all"
            uncertainty = cytogater.calculate_uncertainty(
                probabilities, adata, sample_col=sample_column, k_spatial=k_spatial
            )
            for column in ("entropy", "gini_impurity", "margin_uncertainty", "spatial_discordance"):
                adata.obs[column] = uncertainty[column]
            if temporary_sample:
                del adata.obs[sample_column]
        return source

    def _run_gating(
        self, working: ad.AnnData, lineage: pd.DataFrame, plan: AnnotationPlan
    ) -> dict[str, Any]:
        stage = plan.stages[0]
        parameters = dict(stage.parameters)
        if stage.kind == "soft_gating":
            return cytogater.run_soft_gating(
                working,
                lineage,
                assay_name=plan.assay_name,
                unknown_thresh=float(parameters.get("unknown_thresh", 0.4)),
                store=True,
            )
        if stage.kind == "tree_gating":
            return cytogater.run_tree_gating(
                working,
                lineage,
                assay_name=plan.assay_name,
                max_depth=int(parameters.get("max_depth", 4)),
                min_cells=int(parameters.get("min_cells", 200)),
                min_score=float(parameters.get("min_score", 0.5)),
                uncert_thresh=float(parameters.get("uncert_thresh", 0.25)),
                neg_strength=float(parameters.get("neg_strength", 0.4)),
                cutoff_method=str(parameters.get("cutoff_method", "mean")),
                workers=int(parameters.get("workers", 1)),
            )
        raise ValueError(f"The first stage must be gating, got {stage.kind!r}")

    def _run_downstream(
        self,
        working: ad.AnnData,
        gating_result: dict[str, Any],
        plan: AnnotationPlan,
        progress: Callable[[str], None],
        total_stages: int | None = None,
        stage_offset: int = 0,
        artifact_dir: Path | None = None,
    ) -> tuple[str | None, pd.DataFrame | None]:
        if len(plan.stages) == 1:
            return None, None
        downstream = plan.stages[1:]
        stage_total = total_stages or len(plan.stages)
        stage_number = 2 + stage_offset
        if downstream[0].kind == "select_core_cells":
            core_stage = downstream.pop(0)
            progress(f"stage {stage_number}/{stage_total}: select_core_cells")
            self._select_core_cells(working, gating_result, core_stage.parameters)
            stage_number += 1
            if not downstream:
                return "cutoff_label", None

        # Preserve an explicitly reviewed core-cell stage. For compact plans,
        # retain backwards compatibility by constructing defaults automatically.
        if "cutoff_label" not in working.obs:
            reference_parameters = downstream[0].parameters
            self._select_core_cells(
                working,
                gating_result,
                {
                    "cutoff_strategy": "quantile",
                    "probability": reference_parameters.get("reference_quantile", 0.98),
                    "unknown_label": "Unknown",
                },
            )

        first = downstream[0]
        progress(f"stage {stage_number}/{stage_total}: {first.kind}")
        if first.kind.startswith("train_"):
            label_column, probabilities = self._train_stage(
                working, plan.assay_name, first, artifact_dir
            )
            if len(downstream) == 2:
                prediction = downstream[1]
                progress(f"stage {stage_number + 1}/{stage_total}: {prediction.kind}")
                return self._predict_stage(working, plan.assay_name, prediction)
            return label_column, probabilities
        if first.kind.startswith("predict_"):
            return self._predict_stage(working, plan.assay_name, first)
        if first.kind == "weighted_knn":
            return self._run_knn(working, plan.assay_name, first.parameters)
        if first.kind == "random_forest":
            return self._run_rf(working, plan.assay_name, first.parameters)
        if first.kind == "neural_network":
            return self._run_nn(working, plan.assay_name, first.parameters)
        if first.kind == "hierarchical_knn":
            hierarchical_column = self._run_hierarchical(
                working, plan.assay_name, first.parameters
            )
            if len(downstream) == 2 and downstream[1].kind == "rf_rescue":
                progress(f"stage {stage_number + 1}/{stage_total}: rf_rescue")
                return self._run_rf_rescue(
                    working,
                    plan.assay_name,
                    hierarchical_column,
                    downstream[1].parameters,
                )
            return hierarchical_column, None
        raise ValueError(f"Unsupported downstream stage {first.kind!r}")

    def _train_stage(
        self,
        adata: ad.AnnData,
        assay: str,
        stage: PipelineStage,
        artifact_dir: Path | None,
    ) -> tuple[str, pd.DataFrame | None]:
        artifact_dir = artifact_dir or Path("models")
        artifact_dir.mkdir(parents=True, exist_ok=True)
        p = stage.parameters
        method = stage.kind.removeprefix("train_")
        artifact_path = artifact_dir / f"{method}_{uuid.uuid4().hex[:10]}.joblib"
        training_column = _prepare_training_labels(adata, p)

        if stage.kind == "train_weighted_knn":
            fit = cytogater.train_custom_knn(
                adata, label_col=training_column, assay_name=assay,
                cv_folds=int(p.get("cv_folds", 5)), repeats=int(p.get("repeats", 10)),
                agreement_thresh=float(p.get("agreement_thresh", 0.8)),
                k=int(p.get("k", 5)), method=str(p.get("method", "pearson")),
                seed=int(p.get("seed", 0)),
                chunk_size=int(p.get("training_chunk_size", 250)),
            )
            probabilities = fit.get("core_prob_mat")
        elif stage.kind == "train_random_forest":
            fit = cytogater.train_custom_randomforest(
                adata, label_col=training_column, assay_name=assay,
                num_trees=int(p.get("num_trees", 200)),
                cv_folds=int(p.get("cv_folds", 5)), repeats=int(p.get("repeats", 10)),
                agreement_thresh=float(p.get("agreement_thresh", 0.8)),
                num_threads=int(p.get("num_threads", 2)), seed=int(p.get("seed", 0)),
            )
            probabilities = fit.get("core_prob_mat")
        elif stage.kind == "train_neural_network":
            fit = cytogater.train_custom_dl(
                adata, label_col=training_column, assay_name=assay,
                hidden_dims=tuple(p.get("hidden_dims", [64, 32])),
                dropout=float(p.get("dropout", 0.3)), epochs=int(p.get("epochs", 50)),
                lr=float(p.get("learning_rate", 0.001)),
                batch_size=int(p.get("batch_size", 256)), seed=int(p.get("seed", 0)),
            )
            adata.obs["cleaned_core_label"] = adata.obs[training_column].copy()
            # Persist portable CPU tensors. Prediction can move the network to
            # an available accelerator at runtime.
            fit["model"]["net"] = fit["model"]["net"].to("cpu")
            fit["model"]["device"] = "cpu"
            probabilities = None
        elif stage.kind == "train_hierarchical_knn":
            fit = cytogater.train_custom_knn(
                adata, label_col=training_column, assay_name=assay,
                cv_folds=int(p.get("reference_cv_folds", 5)),
                repeats=int(p.get("reference_repeats", 10)),
                agreement_thresh=float(p.get("reference_agreement_thresh", 0.8)),
                k=int(p.get("k", 5)),
                method=str(p.get("reference_method", "pearson")),
                seed=int(p.get("seed", 0)),
            )
            tree = cytogater.build_lineage_hierarchy(
                adata, label_col="cleaned_core_label", assay_name=assay
            )
            reference = cytogater.build_hierarchical_reference(
                adata, tree, label_col="cleaned_core_label",
                top_n=int(p.get("top_n", 5)), assay_name=assay,
                unknown_label="Unknown",
            )
            fit = {**fit, "hierarchy": tree, "hierarchical_reference": reference}
            probabilities = fit.get("core_prob_mat")
        else:
            raise ValueError(f"Unsupported training stage {stage.kind!r}")

        joblib.dump(fit, artifact_path)
        adata.uns[f"cytogater_model_{method}"] = str(artifact_path)
        if probabilities is not None:
            _store_named_probabilities(adata, f"cytogater_{method}_core_probabilities", probabilities)
        return "cleaned_core_label", probabilities

    def _predict_stage(
        self, adata: ad.AnnData, assay: str, stage: PipelineStage
    ) -> tuple[str, pd.DataFrame | None]:
        method = stage.kind.removeprefix("predict_")
        artifact_key = f"cytogater_model_{method}"
        artifact_path = adata.uns.get(artifact_key)
        if not artifact_path or not Path(str(artifact_path)).exists():
            raise RuntimeError(
                f"{stage.kind} requires a completed train_{method} stage"
            )
        fit = _load_model_artifact(Path(str(artifact_path)))
        p = stage.parameters
        prediction_column, target_sentinel = _prepare_prediction_targets(adata, p)
        if stage.kind == "predict_weighted_knn":
            prediction = cytogater.predict_unknown_with_knn(
                adata, fit, assay_name=assay, label_col=prediction_column,
                out_col="knn_label_filled", pred_col="knn_pred",
                unknown_label=target_sentinel,
                threshold=float(p.get("prediction_threshold", 0.6)),
                k=int(p.get("k", 5)), dist_method=str(p.get("method", "pearson")),
                chunk_size=int(p.get("prediction_chunk_size", 10000)),
            )
            column = "knn_label_filled"
        elif stage.kind == "predict_random_forest":
            prediction = cytogater.predict_unknown_with_randomforest(
                adata, fit["model"], assay_name=assay, label_col=prediction_column,
                out_col="rf_label_filled", pred_col="rf_pred",
                unknown_label=target_sentinel,
                threshold=float(p.get("prediction_threshold", 0.5)),
            )
            column = "rf_label_filled"
        elif stage.kind == "predict_neural_network":
            prediction = cytogater.predict_unknown_with_dl(
                adata, fit["model"], assay_name=assay, label_col=prediction_column,
                out_col="nn_label_filled", pred_col="nn_pred",
                unknown_label=target_sentinel,
                threshold=float(p.get("prediction_threshold", 0.5)),
            )
            column = "nn_label_filled"
        elif stage.kind == "predict_hierarchical_knn":
            cytogater.predict_hierarchical_knn_recursive(
                adata, fit["hierarchical_reference"], fit["hierarchy"],
                assay_name=assay,
                threshold=float(p.get("prediction_threshold", 0.7)),
                agreement_threshold=float(p.get("prediction_agreement", 0.8)),
                k=int(p.get("k", 5)), repeats=int(p.get("prediction_repeats", 5)),
                dist_methods=tuple(p.get("distance_methods", ["pearson", "cosine"])),
                out_col="hierarchical_label",
                chunk_size=int(p.get("prediction_chunk_size", 1000)),
            )
            predicted = adata.obs["hierarchical_label"].astype(object).copy()
            preserved = adata.obs[prediction_column].astype(object).copy()
            target_mask = preserved.astype(str) == target_sentinel
            preserved.loc[target_mask] = predicted.loc[target_mask]
            adata.obs["hierarchical_label"] = preserved
            prediction = {"prob_mat": None}
            column = "hierarchical_label"
        else:
            raise ValueError(f"Unsupported prediction stage {stage.kind!r}")

        core = _load_named_probabilities(
            adata, f"cytogater_{method}_core_probabilities"
        )
        complete = _combine_probability_pieces(core, prediction.get("prob_mat"), adata)
        return column, complete

    @staticmethod
    def _select_core_cells(
        working: ad.AnnData,
        gating_result: dict[str, Any],
        parameters: dict[str, Any],
    ) -> None:
        strategy = str(parameters.get("cutoff_strategy", "quantile"))
        if strategy == "quantile":
            probability = float(parameters.get("probability", 0.98))

            def cutoff(values):
                return cytogater.prob_quantile_cutoff(values, prob=probability)
        elif strategy == "mad":
            multiplier = float(parameters.get("mad_multiplier", 3.0))

            def cutoff(values):
                return cytogater.prob_mad_cutoff(values, mad_mult=multiplier)
        else:
            raise ValueError("cutoff_strategy must be 'quantile' or 'mad'")
        cytogater.apply_cutoff_labels(
            gating_result,
            cutoff_fn=cutoff,
            label_col="cutoff_label",
            unknown_label=str(parameters.get("unknown_label", "Unknown")),
        )
        # apply_cutoff_labels writes through gating_result['spe']; keep this
        # assertion explicit because continuation plans depend on that contract.
        if "cutoff_label" not in working.obs:
            raise RuntimeError("cytoGater did not store cutoff_label after core-cell selection")

    def _run_knn(
        self, adata: ad.AnnData, assay: str, p: dict[str, Any]
    ) -> tuple[str, pd.DataFrame | None]:
        fit = cytogater.train_custom_knn(
            adata,
            label_col="cutoff_label",
            assay_name=assay,
            cv_folds=int(p.get("cv_folds", 5)),
            repeats=int(p.get("repeats", 10)),
            agreement_thresh=float(p.get("agreement_thresh", 0.8)),
            k=int(p.get("k", 5)),
            method=str(p.get("method", "pearson")),
            seed=int(p.get("seed", 0)),
            chunk_size=int(p.get("training_chunk_size", 250)),
        )
        prediction = cytogater.predict_unknown_with_knn(
            adata,
            fit,
            assay_name=assay,
            label_col="cleaned_core_label",
            out_col="knn_label_filled",
            pred_col="knn_pred",
            threshold=float(p.get("prediction_threshold", 0.6)),
            k=int(p.get("k", 5)),
            dist_method=str(p.get("method", "pearson")),
            chunk_size=int(p.get("prediction_chunk_size", 10000)),
        )
        return "knn_label_filled", _complete_probabilities(fit, prediction, adata)

    def _run_rf(
        self, adata: ad.AnnData, assay: str, p: dict[str, Any]
    ) -> tuple[str, pd.DataFrame | None]:
        fit = cytogater.train_custom_randomforest(
            adata,
            label_col="cutoff_label",
            assay_name=assay,
            num_trees=int(p.get("num_trees", 200)),
            cv_folds=int(p.get("cv_folds", 5)),
            repeats=int(p.get("repeats", 10)),
            agreement_thresh=float(p.get("agreement_thresh", 0.8)),
            num_threads=int(p.get("num_threads", 2)),
            seed=int(p.get("seed", 0)),
        )
        prediction = cytogater.predict_unknown_with_randomforest(
            adata,
            fit["model"],
            assay_name=assay,
            label_col="cleaned_core_label",
            out_col="rf_label_filled",
            pred_col="rf_pred",
            threshold=float(p.get("prediction_threshold", 0.5)),
        )
        return "rf_label_filled", _complete_probabilities(fit, prediction, adata)

    def _run_nn(
        self, adata: ad.AnnData, assay: str, p: dict[str, Any]
    ) -> tuple[str, pd.DataFrame | None]:
        fit = cytogater.train_custom_dl(
            adata,
            label_col="cutoff_label",
            assay_name=assay,
            hidden_dims=tuple(p.get("hidden_dims", [64, 32])),
            dropout=float(p.get("dropout", 0.3)),
            epochs=int(p.get("epochs", 50)),
            lr=float(p.get("learning_rate", 0.001)),
            batch_size=int(p.get("batch_size", 256)),
            seed=int(p.get("seed", 0)),
        )
        cytogater.predict_unknown_with_dl(
            adata,
            fit["model"],
            assay_name=assay,
            label_col="cutoff_label",
            out_col="nn_label_filled",
            pred_col="nn_pred",
            threshold=float(p.get("prediction_threshold", 0.5)),
        )
        return "nn_label_filled", None

    def _run_hierarchical(
        self, adata: ad.AnnData, assay: str, p: dict[str, Any]
    ) -> str:
        # kNN consensus supplies the cleaned reference consumed by the hierarchy.
        cytogater.train_custom_knn(
            adata,
            label_col="cutoff_label",
            assay_name=assay,
            cv_folds=int(p.get("reference_cv_folds", 5)),
            repeats=int(p.get("reference_repeats", 10)),
            agreement_thresh=float(p.get("reference_agreement_thresh", 0.8)),
            k=int(p.get("k", 5)),
            method=str(p.get("reference_method", "pearson")),
            seed=int(p.get("seed", 0)),
        )
        tree = cytogater.build_lineage_hierarchy(
            adata, label_col="cleaned_core_label", assay_name=assay
        )
        reference = cytogater.build_hierarchical_reference(
            adata,
            tree,
            label_col="cleaned_core_label",
            top_n=int(p.get("top_n", 5)),
            assay_name=assay,
            unknown_label="Unknown",
        )
        cytogater.predict_hierarchical_knn_recursive(
            adata,
            reference,
            tree,
            assay_name=assay,
            threshold=float(p.get("prediction_threshold", 0.7)),
            agreement_threshold=float(p.get("prediction_agreement", 0.8)),
            k=int(p.get("k", 5)),
            repeats=int(p.get("prediction_repeats", 5)),
            dist_methods=tuple(p.get("distance_methods", ["pearson", "cosine"])),
            out_col="hierarchical_label",
            chunk_size=int(p.get("prediction_chunk_size", 1000)),
        )
        return "hierarchical_label"

    def _run_rf_rescue(
        self,
        adata: ad.AnnData,
        assay: str,
        source_column: str,
        p: dict[str, Any],
    ) -> tuple[str, pd.DataFrame | None]:
        # cytoGater's predictor returns early when the hierarchy left nothing
        # unassigned, so initialize the output to the primary labels first.
        adata.obs["hierarchical_rf_label"] = adata.obs[source_column].copy()
        fit = cytogater.train_custom_randomforest(
            adata,
            label_col=source_column,
            assay_name=assay,
            unknown_label="Unassigned",
            num_trees=int(p.get("num_trees", 200)),
            cv_folds=int(p.get("cv_folds", 5)),
            repeats=int(p.get("repeats", 10)),
            agreement_thresh=float(p.get("agreement_thresh", 0.8)),
            num_threads=int(p.get("num_threads", 2)),
            seed=int(p.get("seed", 0)),
        )
        cytogater.predict_unknown_with_randomforest(
            adata,
            fit["model"],
            assay_name=assay,
            label_col="cleaned_core_label",
            out_col="hierarchical_rf_label",
            pred_col="rf_rescue_pred",
            unknown_label="Unassigned",
            threshold=float(p.get("prediction_threshold", 0.5)),
            unassigned_label="Unassigned",
        )
        adata.obs["primary_label"] = adata.obs[source_column]
        adata.obs["final_label"] = adata.obs["hierarchical_rf_label"]
        adata.obs["label_source"] = np.where(
            adata.obs[source_column].astype(str) == "Unassigned",
            np.where(
                adata.obs["hierarchical_rf_label"].astype(str) == "Unassigned",
                "unassigned",
                "rf_rescue",
            ),
            "hierarchical",
        )
        return "final_label", None


def _load_model_artifact(path: Path) -> dict[str, Any]:
    """Load current CPU artifacts and legacy CUDA-pickled torch artifacts."""
    try:
        return joblib.load(path)
    except RuntimeError as exc:
        if "Attempting to deserialize object on a CUDA device" not in str(exc):
            raise
        import torch

        original = torch.storage._load_from_bytes

        def load_on_cpu(value):
            return torch.load(io.BytesIO(value), map_location="cpu", weights_only=False)

        try:
            torch.storage._load_from_bytes = load_on_cpu
            artifact = joblib.load(path)
        finally:
            torch.storage._load_from_bytes = original
        model = artifact.get("model", artifact)
        if isinstance(model, dict) and "net" in model:
            model["net"] = model["net"].to("cpu")
            model["device"] = "cpu"
        return artifact


def _prepare_training_labels(adata: ad.AnnData, parameters: dict[str, Any]) -> str:
    """Create a normalized training label from a user-selected observation column."""
    source = str(parameters.get("training_label_column", "cutoff_label"))
    if source not in adata.obs:
        raise ValueError(
            f"Training label column {source!r} is absent. Available label columns: "
            f"{_available_label_columns(adata)}"
        )
    excluded = set(map(str, parameters.get(
        "exclude_training_labels", ["Unknown", "Unassigned", "Uncertain"]
    )))
    labels = adata.obs[source].astype(object).copy()
    labels.loc[labels.isna() | labels.astype(str).isin(excluded)] = "Unknown"
    if int((labels.astype(str) != "Unknown").sum()) == 0:
        raise ValueError(f"Training label column {source!r} has no usable labelled cells")
    column = "cytogater_training_label"
    adata.obs[column] = labels
    adata.uns["cytogater_training_label_source"] = source
    adata.uns["cytogater_excluded_training_labels"] = sorted(excluded)
    return column


def _prepare_prediction_targets(
    adata: ad.AnnData, parameters: dict[str, Any]
) -> tuple[str, str]:
    """Normalize one or more target labels for cytoGater's single-unknown API."""
    source = str(parameters.get("prediction_label_column", "cleaned_core_label"))
    if source not in adata.obs:
        raise ValueError(
            f"Prediction label column {source!r} is absent. Available label columns: "
            f"{_available_label_columns(adata)}"
        )
    targets = set(map(str, parameters.get("target_labels", ["Unknown", "Unassigned"])))
    if not targets:
        raise ValueError("target_labels must contain at least one label to predict")
    sentinel = "__cytogater_prediction_target__"
    labels = adata.obs[source].astype(object).copy()
    target_mask = labels.isna() | labels.astype(str).isin(targets)
    labels.loc[target_mask] = sentinel
    column = "cytogater_prediction_input_label"
    adata.obs[column] = labels
    adata.uns["cytogater_prediction_label_source"] = source
    adata.uns["cytogater_prediction_target_labels"] = sorted(targets)
    adata.uns["cytogater_prediction_target_cells"] = int(target_mask.sum())
    return column, sentinel


def _available_label_columns(adata: ad.AnnData) -> list[str]:
    """List compact categorical/string observation columns suitable for planning."""
    columns = []
    for column in adata.obs.columns:
        values = adata.obs[column]
        if (
            isinstance(values.dtype, pd.CategoricalDtype)
            or pd.api.types.is_object_dtype(values.dtype)
            or pd.api.types.is_string_dtype(values.dtype)
        ) and values.nunique(dropna=True) <= 200:
            columns.append(str(column))
    return columns


def _result_labels(result: dict[str, Any], adata: ad.AnnData) -> pd.Series:
    for key in ("hard_label", "soft_label", "label", "labels"):
        if key in result:
            return pd.Series(np.asarray(result[key]), index=adata.obs_names, name="label")
        if key in adata.obs:
            return adata.obs[key]
    for column in ("cell_type_hard", "soft_tree_label", "tree_label", "soft_label"):
        if column in adata.obs:
            return adata.obs[column]
    raise RuntimeError("cytoGater completed but returned no recognized label field")


def _complete_probabilities(
    fit: dict[str, Any], prediction: dict[str, Any], adata: ad.AnnData
) -> pd.DataFrame | None:
    """Combine cross-validated reference and unknown-cell probabilities."""
    pieces = [fit.get("core_prob_mat"), prediction.get("prob_mat")]
    pieces = [piece for piece in pieces if piece is not None and not piece.empty]
    if not pieces:
        return None
    combined = pd.concat(pieces).reindex(adata.obs_names)
    if combined.isna().all(axis=1).any():
        return None
    return combined


def _combine_probability_pieces(
    core: pd.DataFrame | None,
    prediction: pd.DataFrame | None,
    adata: ad.AnnData,
) -> pd.DataFrame | None:
    pieces = [piece for piece in (core, prediction) if piece is not None and not piece.empty]
    if not pieces:
        return None
    combined = pd.concat(pieces).reindex(adata.obs_names)
    return None if combined.isna().all(axis=1).any() else combined


def _store_named_probabilities(
    adata: ad.AnnData, key: str, probabilities: pd.DataFrame
) -> None:
    frame = pd.DataFrame(probabilities).copy()
    adata.obsm[key] = frame.reindex(adata.obs_names)
    adata.uns[f"{key}_columns"] = list(map(str, frame.columns))


def _load_named_probabilities(adata: ad.AnnData, key: str) -> pd.DataFrame | None:
    if key not in adata.obsm:
        return None
    values = adata.obsm[key]
    if isinstance(values, pd.DataFrame):
        frame = values.reindex(adata.obs_names)
    else:
        columns = list(map(str, adata.uns.get(f"{key}_columns", [])))
        frame = pd.DataFrame(values, index=adata.obs_names, columns=columns or None)
    return frame.dropna(how="all")


def _store_annotation_probabilities(
    adata: ad.AnnData, probabilities: pd.DataFrame
) -> None:
    frame = pd.DataFrame(probabilities).reindex(adata.obs_names)
    adata.obsm["cytogater_annotation_probabilities"] = frame
    adata.uns["cytogater_annotation_cell_types"] = list(map(str, frame.columns))


def _stored_annotation_probabilities(adata: ad.AnnData) -> pd.DataFrame:
    key = "cytogater_annotation_probabilities"
    if key not in adata.obsm:
        return _stored_gating_probabilities(adata)
    values = adata.obsm[key]
    if isinstance(values, pd.DataFrame):
        return values.reindex(adata.obs_names)
    columns = list(map(str, adata.uns.get("cytogater_annotation_cell_types", [])))
    return pd.DataFrame(values, index=adata.obs_names, columns=columns or None)


def _stored_gating_probabilities(adata: ad.AnnData) -> pd.DataFrame:
    candidates = (
        ("cytogater_tree_probabilities", "cytogater_tree_cell_types"),
        ("cytogater_soft_probabilities", "cytogater_soft_cell_types"),
    )
    for matrix_key, columns_key in candidates:
        if matrix_key not in adata.obsm:
            continue
        values = adata.obsm[matrix_key]
        if isinstance(values, pd.DataFrame):
            return values.reindex(adata.obs_names)
        columns = list(map(str, adata.uns.get(columns_key, [])))
        return pd.DataFrame(values, index=adata.obs_names, columns=columns or None)
    raise RuntimeError(
        "The previous output has no stored cytoGater gating probabilities; "
        "rerun gating once with the current package version"
    )


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_core_reports(
    adata: ad.AnnData, output_dir: Path, plan_id: str
) -> tuple[Path | None, Path | None]:
    if "cutoff_label" not in adata.obs:
        return None, None
    candidate = adata.obs["cutoff_label"].astype(str)
    cleaned = (
        adata.obs["cleaned_core_label"].astype(str)
        if "cleaned_core_label" in adata.obs
        else None
    )
    unknown = {"Unknown", "Unassigned", "Uncertain", "nan", "None"}
    labels = sorted(set(candidate) | (set(cleaned) if cleaned is not None else set()))
    labels = [label for label in labels if label not in unknown]
    report = pd.DataFrame({"cell_type": labels})
    candidate_counts = candidate.value_counts()
    report["candidate_core"] = report["cell_type"].map(candidate_counts).fillna(0).astype(int)
    if cleaned is not None:
        cleaned_counts = cleaned.value_counts()
        report["cleaned_core"] = report["cell_type"].map(cleaned_counts).fillna(0).astype(int)
        report["removed_by_cleaning"] = report["candidate_core"] - report["cleaned_core"]
    counts_path = output_dir / f"core_cell_counts_{plan_id}.csv"
    report.to_csv(counts_path, index=False)

    import matplotlib.pyplot as plt

    figure_width = max(7, min(18, 0.65 * max(len(report), 1)))
    figure, axis = plt.subplots(figsize=(figure_width, 5))
    x = np.arange(len(report))
    if "cleaned_core" in report:
        axis.bar(x - 0.2, report["candidate_core"], width=0.4, label="Candidate core")
        axis.bar(x + 0.2, report["cleaned_core"], width=0.4, label="Cleaned core")
    else:
        axis.bar(x, report["candidate_core"], width=0.65, label="Candidate core")
    axis.set_xticks(x, report["cell_type"], rotation=45, ha="right")
    axis.set_ylabel("Cells")
    axis.set_title("Core cells per cell type")
    axis.legend(
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0,
    )
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    plot_path = output_dir / f"core_cell_counts_{plan_id}.png"
    figure.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return counts_path, plot_path


def _prediction_summary(
    previous: pd.Series | None, current: pd.Series
) -> dict[str, int] | None:
    if previous is None:
        return None
    unassigned = {"Unknown", "Unassigned", "Uncertain", "nan", "None"}
    before_mask = previous.isin(unassigned)
    after_mask = current.isin(unassigned)
    return {
        "assigned": int((~after_mask).sum()),
        "unassigned": int(after_mask.sum()),
        "unassigned_before": int(before_mask.sum()),
        "newly_assigned": int((before_mask & ~after_mask).sum()),
    }


def _confidence_summary(adata: ad.AnnData) -> dict[str, Any] | None:
    columns = [
        column for column in ("rf_confidence", "knn_confidence", "dl_confidence")
        if column in adata.obs
    ]
    if not columns:
        return None
    column = columns[-1]
    values = pd.to_numeric(adata.obs[column], errors="coerce").dropna()
    if values.empty:
        return {"column": column, "n": 0}
    return {
        "column": column,
        "n": int(len(values)),
        "min": float(values.min()),
        "median": float(values.median()),
        "mean": float(values.mean()),
        "max": float(values.max()),
    }

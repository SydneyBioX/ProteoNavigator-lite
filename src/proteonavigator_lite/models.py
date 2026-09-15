"""Typed records shared by the UI, agent, and scientific services."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import anndata as ad
import pandas as pd
from pydantic import BaseModel, Field


class CellTypeDefinition(BaseModel):
    """One cell type in a lineage table."""

    cell_type: str
    positive_markers: list[str] = Field(default_factory=list)
    negative_markers: list[str] = Field(default_factory=list)
    parent: str | None = None
    rationale: str = ""


class LineageProposal(BaseModel):
    """Structured lineage table proposed or revised by the model."""

    definitions: list[CellTypeDefinition]
    notes: list[str] = Field(default_factory=list)


class PipelineStage(BaseModel):
    """One executable stage in an annotation workflow."""

    kind: Literal[
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
    ]
    parameters: dict[str, Any] = Field(default_factory=dict)


class AnnotationPlan(BaseModel):
    """Reviewable, versioned plan that must be approved before execution."""

    plan_id: str
    dataset_id: str
    lineage_revision: str
    workflow: str
    assay_name: str = "exprs"
    stages: list[PipelineStage]
    input_path: str | None = None
    warnings: list[str] = Field(default_factory=list)
    approved: bool = False


@dataclass
class DatasetRecord:
    """A loaded dataset kept in a local session workspace."""

    dataset_id: str
    name: str
    source_path: Path
    adata: ad.AnnData
    assay_name: str


@dataclass
class SessionWorkspace:
    """In-memory scientific state for one Chainlit chat session."""

    root: Path
    dataset: DatasetRecord | None = None
    lineage: pd.DataFrame | None = None
    lineage_revision: str | None = None
    plans: dict[str, AnnotationPlan] = field(default_factory=dict)
    run_status: dict[str, str] = field(default_factory=dict)
    last_output: Path | None = None
    last_result: dict[str, Any] | None = None
    pending_visualizations: list[dict[str, str]] = field(default_factory=list)

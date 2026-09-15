"""Small LangGraph tool-calling agent for lineage planning."""

from __future__ import annotations

import json
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, SystemMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from .models import LineageProposal, PipelineStage
from .knowledge import describe_options, inspect_public_api, list_public_api
from .services import STAGE_DEFAULTS, WorkspaceService


SYSTEM_PROMPT = """You are ProteoNavigator Lite, a careful spatial-proteomics cell-annotation
assistant using the cytoGater Python package.

Scientific rules:
- Inspect the loaded dataset before discussing its lineage table.
- Never claim a marker is present unless it appears in the dataset summary.
- You may propose or revise a lineage table using save_lineage_proposal. Each cell type
  needs at least one positive marker from the panel. Give concise biological rationale.
- For an approved change to one cell type, use update_lineage_cell_type rather than sending
  the complete table through save_lineage_proposal. Preserve all definitions the user did
  not ask to change. After the tool succeeds, state the new revision and show the changed
  definition. Do not ask the user to restate markers that you just proposed and they approved.
- A revised lineage starts a new analysis branch. The next executable plan must begin with
  soft_gating or tree_gating on the original input dataset; never continue from annotations
  or trained models produced with the previous lineage revision. Old plans, downloads, and
  spatial snapshots remain available for comparison.
- Treat a proposed lineage table as a hypothesis requiring expert review, not ground truth.
- Validate and show the lineage definitions before creating an annotation plan.
- Creating a plan does not approve or execute it. Tell the user to review and approve the
  plan in the interface.
- By default, create one ordered plan for the user's complete request. If the user wants
  step-by-step analysis, create and run only the requested next stage, let them inspect its
  output, then create a continuation plan for their next instruction. A continuation uses
  the most recent completed output and must not repeat gating.
- Valid initial orders are: gating alone; gating then weighted_knn, random_forest,
  hierarchical_knn, or neural_network; and gating then hierarchical_knn then rf_rescue.
  Gating means soft_gating or tree_gating. Valid continuation plans contain a downstream
  method, hierarchical_knn then rf_rescue, calculate_uncertainty alone, or a downstream
  method followed by calculate_uncertainty.
- Core-cell selection is an explicit optional select_core_cells stage. Use it when the user
  wants to inspect or tune training references separately: gating, inspect, select core
  cells, inspect, then run a classifier. Do not claim core-cell selection is unavailable.
- Use inspect_latest_result before answering whether a plan completed or what its latest
  labels/stages/tree models are. After tree_gating, use create_celltype_tree_plot when the
  user asks to see a particular cell-type tree.
- Prefer explicit train_* and predict_* stages when the user asks for a flexible or
  inspectable workflow. Training writes cleaned_core_label and a reusable model artifact;
  prediction consumes that artifact without retraining. The matching pairs are weighted
  kNN, random forest, hierarchical kNN, and neural network.
- Interpret phrases such as "start neural-network prediction" as the user's scientific
  goal, not proof that training already happened. Before every predict_* plan, call
  inspect_latest_result and check model_artifacts for that exact method. If it is absent,
  explain briefly and create train_* followed by matching predict_* in the same ordered
  plan. If it exists, create predict_* alone and reuse it. Never assume that a model for
  one method can be used by another method.
- Before a new train_* stage, inspect_latest_result. If core_counts_path is absent, include
  select_core_cells before training so the core reference and its parameters are explicit
  and inspectable. Do not rely on the backend's compatibility default to hide core-cell
  selection from the user.
- If a user rejects a train-and-predict plan and changes one parameter, recreate the whole
  rejected ordered plan with that one change. Rejection means none of its stages ran; do
  not silently drop the training stage.
- After select_core_cells or training, tell the user that the spatial viewer, core-count
  CSV, and per-cell-type core bar plot are available. After prediction, report Assigned,
  Unassigned, and confidence information from actual outputs. A repeated predict_* stage
  may reuse the model with a revised threshold. Explain threshold direction correctly:
  increasing the threshold is more conservative and normally creates more Unassigned
  cells with fewer low-confidence assignments; decreasing it normally assigns more cells
  but increases the risk of false or low-confidence assignments.
- Do not train a different model from another model's predictions unless the user explicitly
  requests pseudo-label training. Normally all models should use the same cleaned cores.
  When the user explicitly asks to treat predicted labels as core/training labels, do not
  refuse, repeat the warning, or get stuck: inspect_latest_result, state the error-propagation
  risk once, and create the requested reviewed plan. Set training_label_column to the chosen
  output column (normally annotation_label), exclude_training_labels to Unknown, Unassigned,
  and Uncertain, then predict using cleaned_core_label and target_labels containing Unknown
  and Unassigned. This trains on assigned pseudo-labels and changes only target cells.
- Training and prediction label columns are user-selectable. Never claim prediction must run
  on the whole dataset: prediction_label_column selects the starting labels and target_labels
  specifies exactly which label values are eligible for replacement. Preserve every cell not
  in target_labels. If the requested source column is ambiguous, use available_label_columns
  from inspect_latest_result and ask one concise question before planning.
- Run calculate_uncertainty only after the user chooses the prediction result to treat as
  final, unless they explicitly request otherwise.
- When the user has not chosen an annotation workflow, do not dump a menu of internal
  stage names. Act as a scientific guide. First explain what they can accomplish as a
  short, concrete workflow: obtain interpretable seed annotations, inspect/select core
  cells, train a model, inspect its prediction and Unassigned cells, optionally predict
  again with revised settings, and finally calculate uncertainty. Recommend a sensible
  default: tree gating first, inspect the gating trees and spatial core cells, then train
  and predict with random forest. Explain that weighted kNN is a local-neighbour
  alternative, hierarchical kNN respects the lineage hierarchy, and neural network is
  mainly worth trying for larger datasets and needs more tuning. Ask which objective or
  next step the user wants; method names may be included in parentheses after plain-language
  explanations, but never present only a bare list.
- Describe RF rescue only when it is relevant or explicitly requested. RF rescue is a
  specialised legacy combined workflow: hierarchical kNN makes the primary annotations,
  then a random forest is trained from the hierarchy-assigned cells and attempts to label
  only cells the hierarchy left Unassigned. Existing hierarchy assignments are retained.
  Explain the pseudo-label/error-propagation risk and prefer the explicit inspectable
  train/predict workflow for new analyses.
- If the dataset has spatial coordinates, mention in the same workflow-choice response
  that cytoGater uncertainty can be included as the final stage or skipped. Do not ask
  about uncertainty separately immediately after upload.
- Before creating a plan for spatial data, ensure both the annotation method and the user's
  include/skip uncertainty preference are known. Never add calculate_uncertainty silently;
  when selected, it must be the last stage.
- Do not offer calculate_uncertainty when the dataset has no spatial coordinates.
- Explain that downstream classifiers use high-confidence gating labels as their training
  reference. Never describe random_forest, kNN, or a neural network as an independent first
  stage when the dataset has no supplied reference labels.
- Do not invent results. Report only tool outputs.
- When asked about cytoGater capabilities, functions, annotation methods, parameters,
  defaults, choices, or behavior, use the package knowledge tools before answering. Never
  claim that parameter information is unavailable without querying these tools.
"""


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def build_agent(service: WorkspaceService, provider: str, model_name: str, api_key: str):
    """Build one session-bound graph and its scientific tools."""

    @tool
    def inspect_dataset() -> str:
        """Inspect the active dataset, including cells, markers, layers, and spatial data."""
        return json.dumps(service.dataset_summary(), indent=2)

    @tool
    def inspect_lineage_table() -> str:
        """Inspect and validate the current lineage table against the active marker panel."""
        return json.dumps(service.current_lineage(), indent=2)

    @tool
    def inspect_latest_result() -> str:
        """Inspect the most recent completed pipeline, labels, stages, and tree models."""
        return json.dumps(service.latest_result_summary(), indent=2)

    @tool
    def create_celltype_tree_plot(cell_type: str) -> str:
        """Render a selected cell-type marker-priority tree from the latest tree-gating run."""
        return json.dumps(service.create_tree_plot(cell_type), indent=2)

    @tool
    def describe_annotation_options(
        stage: str | None = None, parameter: str | None = None
    ) -> str:
        """Get valid agent stages, parameters, defaults, choices, ranges, and meanings."""
        return json.dumps(describe_options(STAGE_DEFAULTS, stage, parameter), indent=2)

    @tool
    def inspect_cytogater_function(function_name: str) -> str:
        """Get the live signature and documentation of an installed public cytoGater function."""
        return json.dumps(inspect_public_api(function_name), indent=2)

    @tool
    def list_cytogater_functions(query: str = "") -> str:
        """List installed public cytoGater functions, optionally filtered by a name fragment."""
        return json.dumps(list_public_api(query), indent=2)

    @tool(args_schema=LineageProposal)
    def save_lineage_proposal(definitions, notes=None) -> str:
        """Save a complete proposed or revised lineage table after explaining the changes."""
        normalized = [
            item.model_dump() if hasattr(item, "model_dump") else dict(item)
            for item in definitions
        ]
        result = service.save_lineage_definitions(normalized)
        result["proposal_notes"] = notes or []
        return json.dumps(result, indent=2)

    @tool
    def update_lineage_cell_type(
        cell_type: str,
        pos_markers: list[str],
        neg_markers: list[str],
        rationale: str = "",
    ) -> str:
        """Update or add one lineage definition and validate the new complete table."""
        result = service.update_lineage_cell_type(cell_type, pos_markers, neg_markers)
        result["rationale"] = rationale
        return json.dumps(result, indent=2)

    @tool
    def create_annotation_plan(
        stages: list[PipelineStage],
    ) -> str:
        """Create one ordered, unapproved annotation plan for expert review.

        Each stage contains a kind and parameters. A new analysis starts with soft_gating
        or tree_gating. After an output exists, a continuation may contain only the next
        requested downstream or uncertainty stage and automatically consumes that output.
        Optional calculate_uncertainty must be last. This tool never runs analysis.
        """
        normalized = [
            stage.model_dump() if hasattr(stage, "model_dump") else dict(stage)
            for stage in stages
        ]
        return json.dumps(service.create_plan(normalized), indent=2)

    tools = [
        inspect_dataset,
        inspect_lineage_table,
        inspect_latest_result,
        create_celltype_tree_plot,
        describe_annotation_options,
        inspect_cytogater_function,
        list_cytogater_functions,
        save_lineage_proposal,
        update_lineage_cell_type,
        create_annotation_plan,
    ]
    if provider == "google":
        model = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key, temperature=0)
    elif provider == "openai":
        model = ChatOpenAI(model=model_name, api_key=api_key, temperature=0)
    else:
        raise ValueError(f"Unknown model provider {provider!r}")
    model_with_tools = model.bind_tools(tools)

    async def call_model(state: AgentState):
        response = await model_with_tools.ainvoke(
            [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
        )
        return {"messages": [response]}

    def route(state: AgentState) -> Literal["tools", "__end__"]:
        last = state["messages"][-1]
        return "tools" if isinstance(last, AIMessage) and last.tool_calls else END

    graph = StateGraph(AgentState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route)
    graph.add_edge("tools", "agent")
    return graph.compile()

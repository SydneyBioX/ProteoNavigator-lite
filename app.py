"""Chainlit interface for ProteoNavigator Lite."""

from __future__ import annotations

import asyncio
import faulthandler
import json
import multiprocessing as mp
import os
import re
import signal
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

import chainlit as cl
import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from proteonavigator_lite.agent import build_agent
from proteonavigator_lite.models import SessionWorkspace
from proteonavigator_lite.services import WorkspaceService, requested_tree_cell_type
from proteonavigator_lite.visualization import spatial_payload


ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT / "runtime"
PLAN_PATTERN = re.compile(r"\b[0-9a-f]{10}\b")
ACTIVE_RUNS: dict[str, dict] = {}
REVIEW_MESSAGES: dict[str, cl.Message] = {}


async def _send_tree_visualization(visualization: dict[str, str]) -> None:
    element_name = f"{visualization['cell_type']}_tree_plot"
    await cl.Message(
        content=f"Open **{element_name}** in the visualization panel.",
        elements=[
            cl.Image(
                name=element_name,
                path=visualization["path"],
                display="side",
                mime="image/png",
            )
        ],
    ).send()


async def _remove_message_actions(message: cl.Message | None) -> None:
    """Remove persisted Chainlit actions, not only their local references."""
    if message is None:
        return
    actions = list(message.actions or [])
    for action in actions:
        try:
            await action.remove()
        except Exception:
            # The action may already have been removed by the clicked callback.
            pass
    message.actions = []


def _save_conversation_event(role: str, content: str, **metadata) -> None:
    """Append one API-key-free event to the current session transcript."""
    service: WorkspaceService | None = cl.user_session.get("service")
    if service is None:
        return
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "role": role,
        "content": content,
        **metadata,
    }
    transcript = service.workspace.root / "conversation.jsonl"
    with transcript.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


async def _show_spatial_panel(
    payload: dict,
    content: str,
    *,
    snapshot_id: str | None = None,
    snapshot_path: str | None = None,
) -> None:
    """Show the live input viewer or create an immutable result snapshot."""
    # Chainlit creates the reopen link only when the exact side-element name is
    # present in its parent message content.
    content = f"{content}\n\nOpen or reopen **SpatialViewer**."
    if snapshot_id is not None:
        element = cl.CustomElement(
            name="SpatialViewer", display="side", size="large", props=payload
        )
        actions = (
            [
                cl.Action(
                    name="open_spatial_snapshot",
                    label=f"Open spatial result · {snapshot_id}",
                    payload={"snapshot_id": snapshot_id, "snapshot_path": snapshot_path},
                )
            ]
            if snapshot_path
            else []
        )
        await cl.Message(content=content, elements=[element], actions=actions).send()
        return
    panel: cl.Message | None = cl.user_session.get("spatial_panel")
    element: cl.CustomElement | None = cl.user_session.get("spatial_element")
    if panel is None:
        element = cl.CustomElement(
            name="SpatialViewer", display="side", size="large", props=payload
        )
        panel = cl.Message(content=content, elements=[element])
        cl.user_session.set("spatial_panel", panel)
        cl.user_session.set("spatial_element", element)
        await panel.send()
    else:
        panel.content = content
        if element is not None:
            element.props = payload
            element.content = json.dumps(payload)
            await element.update()
        await panel.update()


@cl.action_callback("open_spatial_snapshot")
async def open_spatial_snapshot(action: cl.Action):
    """Open the spatial payload attached to the clicked historical result."""
    service: WorkspaceService = cl.user_session.get("service")
    snapshot_path = Path(str(action.payload.get("snapshot_path", "")))
    try:
        resolved = snapshot_path.resolve(strict=True)
        resolved.relative_to(service.workspace.root.resolve())
    except (FileNotFoundError, ValueError):
        await cl.Message(content="That spatial snapshot is no longer available.").send()
        return
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    snapshot_id = str(action.payload.get("snapshot_id", "result"))
    element = cl.CustomElement(
        name="SpatialViewer", display="side", size="large", props=payload
    )
    await cl.Message(
        content=(
            f"Reopened spatial result `{snapshot_id}`. Open or reopen **SpatialViewer**."
        ),
        elements=[element],
    ).send()


def _run_plan_worker(service: WorkspaceService, plan_id: str, result_queue) -> None:
    """Run one plan in an isolated process that the UI can terminate safely."""
    crash_path = service.workspace.root / "runs" / f"{plan_id}.crash.log"
    crash_path.parent.mkdir(parents=True, exist_ok=True)
    with crash_path.open("w", encoding="utf-8") as crash_log:
        faulthandler.enable(file=crash_log, all_threads=True)
        try:
            # Give this run its own process group so cancellation also stops any
            # worker children created by sklearn/joblib/cytoGater.
            if hasattr(os, "setsid"):
                os.setsid()
            result_queue.put(("result", service.run_plan(plan_id)))
        except BaseException as exc:
            details = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
            crash_log.write(details)
            crash_log.flush()
            result_queue.put(("error", details))


def _terminate_run(run: dict) -> None:
    process = run["process"]
    if not process.is_alive():
        return
    try:
        if hasattr(os, "killpg"):
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
    except ProcessLookupError:
        pass


def _text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    return str(content)


async def _configure_model() -> dict | None:
    env_openai = os.environ.get("OPENAI_API_KEY", "")
    env_google = os.environ.get("GOOGLE_API_KEY", "")
    if env_google or env_openai:
        default_provider = "google" if env_google else "openai"
        default_model = "gemini-2.5-flash" if env_google else "gpt-4.1-mini"
        key = env_google or env_openai
        return {"provider": default_provider, "model": default_model, "api_key": key}
    response = await cl.AskElementMessage(
        content="Configure the LLM for this local session.",
        element=cl.CustomElement(
            name="ModelSetup",
            display="inline",
            props={"provider": "google", "model": "gemini-2.5-flash"},
        ),
        timeout=600,
    ).send()
    if response and response.get("submitted"):
        return {
            "provider": response["provider"],
            "model": response["model"],
            "api_key": response["api_key"],
        }
    return None


@cl.on_chat_start
async def on_chat_start():
    session_id = uuid.uuid4().hex
    root = RUNTIME / session_id
    root.mkdir(parents=True, exist_ok=True)
    service = WorkspaceService(SessionWorkspace(root=root))
    cl.user_session.set("service", service)
    cl.user_session.set("messages", [])

    settings = await _configure_model()
    if settings is None:
        await cl.Message(content="Model configuration was cancelled. Restart the chat to try again.").send()
        return
    graph = build_agent(service, settings["provider"], settings["model"], settings["api_key"])
    cl.user_session.set("graph", graph)
    # Never retain the key in a second settings object.
    cl.user_session.set("model_label", f"{settings['provider']}/{settings['model']}")
    _save_conversation_event(
        "system",
        "ProteoNavigator Lite session started.",
        session_id=session_id,
        model=f"{settings['provider']}/{settings['model']}",
    )

    await cl.Message(
        content=(
            f"Ready with **{settings['provider']}/{settings['model']}**. Upload a `.h5ad`, "
            "expression `.csv`, or `.rds` dataset. You may upload a lineage CSV at the same "
            "time or ask me to propose one from the panel.\n\n"
            "Plans may combine gating with weighted kNN, Random Forest, hierarchical kNN, "
            "a neural network, or hierarchical kNN followed by RF rescue. Every plan shows "
            "its ordered stages before approval. Spatial datasets can optionally add "
            "cytoGater uncertainty as the final stage."
        )
    ).send()


def _is_lineage_csv(path: str) -> bool:
    try:
        columns = set(pd.read_csv(path, nrows=2).columns)
    except Exception:
        return False
    return "Populations" in columns or {"cell_type", "pos_markers"}.issubset(columns)


async def _ingest_elements(elements) -> list[str]:
    service: WorkspaceService = cl.user_session.get("service")
    summaries: list[str] = []
    data_files: list[tuple[str, str]] = []
    lineage_files: list[tuple[str, str]] = []
    for element in elements or []:
        path = getattr(element, "path", None)
        if not path:
            continue
        original_name = getattr(element, "name", None) or Path(path).name
        if Path(original_name).suffix.lower() == ".csv" and _is_lineage_csv(path):
            lineage_files.append((path, original_name))
        else:
            data_files.append((path, original_name))
    for path, original_name in data_files:
        summary = await asyncio.to_thread(
            service.ingest_dataset, path, original_name
        )
        summaries.append(
            f"Loaded dataset **{summary['name']}**: {summary['n_cells']:,} cells × "
            f"{summary['n_markers']} markers; assay `{summary['assay_name']}`; "
            f"spatial coordinates: {summary['has_spatial']}."
        )
    for path, original_name in lineage_files:
        result = await asyncio.to_thread(
            service.ingest_lineage, path, original_name
        )
        status = "valid" if result["valid"] else "invalid"
        summaries.append(
            f"Loaded lineage revision `{result['revision']}` ({status}). "
            f"Errors: {result['errors'] or 'none'}. Warnings: {result['warnings'] or 'none'}."
        )
    return summaries


@cl.on_message
async def on_message(message: cl.Message):
    graph = cl.user_session.get("graph")
    if graph is None:
        await cl.Message(content="No model is configured. Start a new chat session.").send()
        return
    _save_conversation_event(
        "user",
        message.content,
        attachments=[getattr(item, "name", None) for item in (message.elements or [])],
    )
    try:
        ingestion = await _ingest_elements(message.elements)
    except Exception as exc:
        await cl.Message(content=f"Input could not be loaded: {exc}").send()
        return
    if ingestion:
        await cl.Message(content="\n\n".join(ingestion)).send()
        service: WorkspaceService = cl.user_session.get("service")
        dataset = service.workspace.dataset
        if dataset is not None:
            payload = spatial_payload(dataset.adata, title=f"Input · {dataset.name}")
            if payload is not None:
                await _show_spatial_panel(
                    payload,
                    "### Spatial viewer\nInput coordinates are shown before annotation.",
                )

    history = cl.user_session.get("messages", [])
    context = message.content
    if ingestion:
        context += "\n\nNew input status:\n" + "\n".join(ingestion)
    history.append(HumanMessage(content=context))
    service: WorkspaceService = cl.user_session.get("service")
    lineage_before = service.workspace.lineage_revision
    latest = service.latest_result_summary()
    requested_tree = requested_tree_cell_type(
        message.content, latest.get("tree_cell_types", [])
    )
    if requested_tree is not None:
        visualization = service.create_tree_plot(requested_tree)
        # create_tree_plot also queues tool-originated visualizations; consume it
        # here because this deterministic route bypasses the LLM tool loop.
        service.consume_visualizations()
        answer = (
            f"The `{requested_tree}` tree from the completed tree-gating run is ready. "
            "It shows the marker-priority decisions learned for that cell type."
        )
        history.append(AIMessage(content=answer))
        cl.user_session.set("messages", history)
        await cl.Message(content=answer).send()
        await _send_tree_visualization(visualization)
        _save_conversation_event("assistant", answer)
        return
    plans_before = set(service.workspace.plans)

    try:
        with cl.Step(name="ProteoNavigator agent", type="llm") as step:
            result = await graph.ainvoke({"messages": history}, {"recursion_limit": 20})
            final = result["messages"][-1]
            answer = _text(final.content)
            step.output = answer
    except Exception as exc:
        await cl.Message(content=f"Agent error: {exc}").send()
        return

    history = result["messages"]
    if not answer.strip():
        current = service.current_lineage()
        if service.workspace.lineage_revision != lineage_before:
            answer = (
                f"Updated and validated lineage revision "
                f"`{service.workspace.lineage_revision}`. The next run must start with "
                "fresh gating; previous outputs remain available for comparison."
            )
        else:
            last_tool = next(
                (item for item in reversed(history) if isinstance(item, ToolMessage)), None
            )
            answer = (
                f"Current lineage revision `{current.get('revision')}`:\n\n"
                f"```json\n{json.dumps(current.get('definitions', []), indent=2)}\n```"
                if last_tool is not None
                else "I could not complete that request. Please try it once more."
            )
        history.append(AIMessage(content=answer))
    cl.user_session.set("messages", history)
    await cl.Message(content=answer).send()
    _save_conversation_event("assistant", answer)
    for visualization in service.consume_visualizations():
        if visualization["kind"] == "tree_plot":
            await _send_tree_visualization(visualization)

    new_plans = set(service.workspace.plans) - plans_before
    for plan_id in sorted(new_plans):
        plan = service.workspace.plans[plan_id]
        ordered = "\n".join(
            f"{index}. **{stage.kind}**\n```json\n"
            f"{json.dumps(stage.parameters, indent=2)}\n```"
            for index, stage in enumerate(plan.stages, 1)
        )
        warnings = (
            "\n\nWarnings:\n" + "\n".join(f"- {warning}" for warning in plan.warnings)
            if plan.warnings
            else ""
        )
        review_message = cl.Message(
            content=(
                f"Review plan `{plan_id}` using lineage revision `{plan.lineage_revision}`.\n\n"
                f"Ordered pipeline:\n{ordered}{warnings}\n\n"
                "Approval will execute these stages in exactly this order."
            ),
            actions=[
                cl.Action(
                    name="approve_plan",
                    label="Approve and run",
                    payload={"plan_id": plan_id},
                ),
                cl.Action(name="reject_plan", label="Keep editing", payload={"plan_id": plan_id}),
            ],
        )
        REVIEW_MESSAGES[plan_id] = review_message
        await review_message.send()


@cl.action_callback("approve_plan")
async def approve_plan(action: cl.Action):
    service: WorkspaceService = cl.user_session.get("service")
    plan_id = action.payload["plan_id"]
    plan = service.approve_plan(plan_id)
    review_message = REVIEW_MESSAGES.pop(plan_id, None)
    if review_message is not None:
        await _remove_message_actions(review_message)
    else:
        await action.remove()
    service: WorkspaceService = cl.user_session.get("service")
    dataset = service.workspace.dataset
    n_cells = dataset.adata.n_obs if dataset is not None else 0
    message = cl.Message(
        content=(
            f"Running approved plan `{plan_id}`: **{plan.workflow}** on "
            f"**{n_cells:,} cells**. Elapsed: 0 seconds."
        ),
        actions=[
            cl.Action(
                name="stop_plan",
                label="Stop pipeline",
                payload={"plan_id": plan_id},
            )
        ],
    )
    await message.send()
    started = time.monotonic()
    context = mp.get_context("fork")
    result_queue = context.Queue()
    process = context.Process(
        target=_run_plan_worker,
        args=(service, plan_id, result_queue),
        name=f"proteonavigator-{plan_id}",
    )
    run = {
        "process": process,
        "queue": result_queue,
        "cancelled": False,
        "message": message,
    }
    ACTIVE_RUNS[plan_id] = run
    process.start()
    try:
        with cl.Step(name=f"Run {plan.workflow}", type="tool") as step:
            while process.is_alive():
                await asyncio.sleep(2)
                if run["cancelled"]:
                    break
                elapsed = int(time.monotonic() - started)
                status_path = service.workspace.root / "runs" / f"{plan_id}.status"
                status = (
                    status_path.read_text().strip()
                    if status_path.exists()
                    else "starting worker"
                )
                message.content = (
                    f"Plan `{plan_id}` is still running: **{plan.workflow}** on "
                    f"**{n_cells:,} cells**. Elapsed: **{elapsed} seconds**.\n\n"
                    f"Current phase: **{status}**.\n\n"
                    "The phase is exact, but cytoGater does not currently report a percentage "
                    "within each phase."
                )
                await message.update()
            await asyncio.to_thread(process.join, 5)
            if process.is_alive():
                process.kill()
                await asyncio.to_thread(process.join, 5)
            if run["cancelled"]:
                step.output = "Cancelled by user"
                return
            try:
                outcome, payload = await asyncio.to_thread(result_queue.get, True, 3)
            except Exception as exc:
                crash_path = service.workspace.root / "runs" / f"{plan_id}.crash.log"
                crash_details = (
                    crash_path.read_text(encoding="utf-8").strip()
                    if crash_path.exists()
                    else ""
                )
                raise RuntimeError(
                    f"Pipeline worker exited with code {process.exitcode} without returning "
                    f"a result{': ' + crash_details if crash_details else ''}"
                ) from exc
            if outcome == "error":
                raise RuntimeError(payload)
            output = payload
            step.output = str(output["label_counts"])
    except Exception as exc:
        elapsed = int(time.monotonic() - started)
        message.content = f"Plan `{plan_id}` failed after {elapsed} seconds: {exc}"
        _save_conversation_event(
            "pipeline", message.content, plan_id=plan_id, status="failed"
        )
        await _remove_message_actions(message)
        await message.update()
        return
    finally:
        ACTIVE_RUNS.pop(plan_id, None)
        result_queue.close()
    elapsed = int(time.monotonic() - started)
    service.record_completed_output(output)
    message.content = (
        f"Plan `{plan_id}` completed in **{elapsed} seconds** with cytoGater "
        f"{output['cytogater_version']}.\n\n"
        f"Label counts: `{output['label_counts']}`"
    )
    prediction = output.get("prediction_summary")
    if prediction:
        message.content += (
            "\n\nPrediction inspection: "
            f"**{prediction['assigned']:,} assigned**, "
            f"**{prediction['unassigned']:,} unassigned**, and "
            f"**{prediction['newly_assigned']:,} newly assigned** from "
            f"{prediction['unassigned_before']:,} previously unassigned cells."
        )
    confidence = output.get("confidence_summary")
    if confidence and confidence.get("n"):
        message.content += (
            f"\n\n`{confidence['column']}` confidence: median "
            f"**{confidence['median']:.3f}**, mean **{confidence['mean']:.3f}**, "
            f"range **{confidence['min']:.3f}–{confidence['max']:.3f}**."
        )
    history = cl.user_session.get("messages", [])
    history.append(
        SystemMessage(
            content=(
                f"Pipeline plan {plan_id} has completed successfully. "
                f"Ordered stages: {output.get('ordered_stages', [])}. "
                f"Latest result summary: {json.dumps(service.latest_result_summary(), default=str)}"
            )
        )
    )
    cl.user_session.set("messages", history)
    _save_conversation_event(
        "pipeline",
        message.content,
        plan_id=plan_id,
        status="completed",
        output_path=output["output_path"],
        counts_path=output["counts_path"],
    )
    message.elements = [
        cl.File(
            name=Path(output["output_path"]).name,
            path=output["output_path"],
            display="inline",
            mime="application/x-hdf5",
        ),
        cl.File(
            name=Path(output["counts_path"]).name,
            path=output["counts_path"],
            display="inline",
            mime="text/csv",
        ),
    ]
    if output.get("core_counts_path"):
        message.elements.append(
            cl.File(
                name=Path(output["core_counts_path"]).name,
                path=output["core_counts_path"],
                display="inline",
                mime="text/csv",
            )
        )
    if output.get("spatial_plot_path"):
        payload = json.loads(Path(output["spatial_plot_path"]).read_text(encoding="utf-8"))
        await _show_spatial_panel(
            payload,
            f"### Spatial result · `{plan_id}`\n"
            "This snapshot is fixed to this pipeline result. Switch between input marker "
            "intensity, cell type, and uncertainty.",
            snapshot_id=plan_id,
            snapshot_path=output["spatial_plot_path"],
        )
    await _remove_message_actions(message)
    await message.update()
    if output.get("core_plot_path"):
        core_plot_name = f"core_cell_counts_{plan_id}"
        await cl.Message(
            content=f"Open **{core_plot_name}** in the visualization panel.",
            elements=[
                cl.Image(
                    name=core_plot_name,
                    path=output["core_plot_path"],
                    display="side",
                    mime="image/png",
                )
            ],
        ).send()


@cl.action_callback("stop_plan")
async def stop_plan(action: cl.Action):
    plan_id = action.payload["plan_id"]
    run = ACTIVE_RUNS.get(plan_id)
    if run is None or not run["process"].is_alive():
        await action.remove()
        await cl.Message(content=f"Plan `{plan_id}` is no longer running.").send()
        return
    await _remove_message_actions(run["message"])
    run["cancelled"] = True
    _terminate_run(run)
    await asyncio.to_thread(run["process"].join, 5)
    if run["process"].is_alive():
        run["process"].kill()
        await asyncio.to_thread(run["process"].join, 5)
    elapsed_message = run["message"]
    elapsed_message.content = (
        f"Plan `{plan_id}` was **cancelled by the user**. The annotation worker "
        "process has been stopped; no partial result should be used."
    )
    _save_conversation_event(
        "pipeline", elapsed_message.content, plan_id=plan_id, status="cancelled"
    )
    await elapsed_message.update()


@cl.action_callback("reject_plan")
async def reject_plan(action: cl.Action):
    plan_id = action.payload["plan_id"]
    service: WorkspaceService = cl.user_session.get("service")
    rejected_plan = service.workspace.plans.get(plan_id)
    review_message = REVIEW_MESSAGES.pop(plan_id, None)
    if review_message is not None:
        await _remove_message_actions(review_message)
        review_message.content += (
            "\n\n**Not approved.** This plan will not run. Tell me what you want "
            "to change and I will create a new ordered plan."
        )
        await review_message.update()
    else:
        await action.remove()
        await cl.Message(content=f"Plan `{plan_id}` was not run.").send()
    if rejected_plan is not None:
        history = cl.user_session.get("messages", [])
        history.append(
            SystemMessage(
                content=(
                    f"Plan {plan_id} was rejected and no stage ran. If the user requests "
                    "a parameter change, recreate the complete ordered plan, retaining all "
                    f"other stages and parameters. Rejected plan: {rejected_plan.model_dump_json()}"
                )
            )
        )
        cl.user_session.set("messages", history)


@cl.on_chat_end
async def on_chat_end():
    for run in list(ACTIVE_RUNS.values()):
        run["cancelled"] = True
        _terminate_run(run)
    _save_conversation_event("system", "Chat connection ended; session files retained.")

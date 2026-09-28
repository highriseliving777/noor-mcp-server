"""
Human Approval Gate for Noor OS
Based on DaNg4RLmAzj tutorial architecture.
Single-file implementation with LangGraph + interrupt().
"""

import json
import re
import os
from pathlib import Path
from datetime import datetime, timezone
from typing import Literal, TypedDict, Annotated, Optional, List
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.types import interrupt

# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

# Paths
BASE_DIR = Path(__file__).parent
META_DIR = BASE_DIR / "_meta"
META_DIR.mkdir(exist_ok=True)

DB_PATH = META_DIR / "approval_checkpoints.sqlite"
AUDIT_LOG = META_DIR / "approval_audit.jsonl"
LEDGER = META_DIR / "approval_ledger.json"

# Model
OLLAMA_MODEL = os.getenv("OLLAMA_FAST_MODEL", "qwen2.5:3b")
MAX_REVISIONS = 3

# Email regex for validation
_EMAIL_RE = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+$")

# ============================================================
# 1. STATE SCHEMA
# ============================================================

class PlannedAction(BaseModel):
    """The irreversible action the agent wants to take."""
    action_type: Literal["email", "calendar", "ticket"] = Field(
        description="Which kind of action this is"
    )
    recipient: str = Field(description="Email to-address, calendar attendee, or ticket assignee")
    subject: str = Field(description="Email subject, event title, or ticket title")
    body: str = Field(description="Email body, event details, or ticket description")
    summary: str = Field(description="One short human-readable line describing the action")


class ApprovalState(TypedDict, total=False):
    """State flowing through the graph."""
    request: str                         # user's natural-language ask
    thread_id: str                       # unique run identifier
    action: dict                         # PlannedAction.model_dump()
    decision: str                        # approve | edit | reject
    feedback: str                        # edit instructions / rejection reason
    revision: int                        # how many times we've drafted
    valid: bool                          # validator verdict
    validation_errors: List[str]         # why the draft was bounced
    status: str                          # pending | sent | cancelled
    result: str                          # confirmation / cancellation message
    history: Annotated[List[str], lambda x, y: x + y]  # append-only log


# ============================================================
# 2. AUDIT TRAIL
# ============================================================

def record_event(thread_id: str, event: str, actor: str, details: dict) -> None:
    """Append one event to the audit log (JSONL)."""
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "thread_id": thread_id,
        "event": event,
        "actor": actor,
        "details": details,
    }
    with AUDIT_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def append_ledger(action: dict, thread_id: str) -> None:
    """Append executed action to ledger."""
    ledger = json.loads(LEDGER.read_text(encoding="utf-8")) if LEDGER.exists() else []
    ledger.append({"thread_id": thread_id, "timestamp": datetime.now(timezone.utc).isoformat(), **action})
    LEDGER.write_text(json.dumps(ledger, indent=2), encoding="utf-8")


# ============================================================
# 3. PROMPTS
# ============================================================

PLANNER_SYSTEM = """You are an executive assistant that turns natural-language requests into concrete, send-ready actions.

Classify the request into exactly one action_type:
- "email": an outbound email. recipient = the to-address, subject = email subject, body = the full email text.
- "calendar": booking a meeting/event. recipient = the attendee's email, subject = the event title, body = date, time, duration and agenda.
- "ticket": opening a support/issue ticket. recipient = the assignee or team, subject = the ticket title, body = a clear description with priority if given.

Rules:
- Write the body in full. Do not leave placeholders like [name] or [date].
- If the request names a concrete email address, use it exactly as the recipient.
- Keep the tone professional and concise.
- summary must be one short line a busy human can approve at a glance.

Return ONLY valid JSON with fields: action_type, recipient, subject, body, summary."""

def revision_block(feedback: str, validation_errors: List[str]) -> str:
    """Build revision instructions."""
    parts = ["\n\nThis is a REVISION of your previous draft."]
    if validation_errors:
        parts.append(f"Your last draft failed validation: {', '.join(validation_errors)}. Fix every issue.")
    if feedback:
        parts.append(f"The reviewer asked for this change: {feedback}")
    parts.append("Produce a corrected action that resolves all of the above.")
    return "\n".join(parts)


# ============================================================
# 4. NODES
# ============================================================

def planner_node(state: ApprovalState) -> dict:
    """Draft or re-draft the action with the LLM."""
    llm = ChatOllama(model=OLLAMA_MODEL, temperature=0.1)
    
    system = PLANNER_SYSTEM
    feedback = state.get("feedback", "")
    errors = state.get("validation_errors", [])
    if feedback or errors:
        system += revision_block(feedback, errors)
    
    messages = [
        SystemMessage(content=system),
        HumanMessage(content=state["request"])
    ]
    
    response = llm.invoke(messages)
    try:
        # Extract JSON from response
        content = response.content
        json_match = re.search(r'\{[^{}]*\}', content)
        if json_match:
            data = json.loads(json_match.group(0))
            planned = PlannedAction(**data)
        else:
            raise ValueError("No JSON found in response")
    except Exception as e:
        # Fallback: create a default action
        record_event(state["thread_id"], "parse_error", "agent", {"error": str(e), "raw": response.content[:200]})
        planned = PlannedAction(
            action_type="ticket",
            recipient="sam@noor-os.ai",
            subject="Failed to parse draft",
            body=f"Original request: {state['request']}\n\nLLM response: {response.content[:500]}",
            summary="Draft parsing failed, manual review required"
        )
    
    revision = state.get("revision", 0) + 1
    record_event(
        state["thread_id"], "drafted", "agent",
        {"revision": revision, "summary": planned.summary, "type": planned.action_type}
    )
    
    return {
        "action": planned.model_dump(),
        "revision": revision,
        "history": [f"planner: drafted v{revision} ({planned.action_type})"],
    }


def validator_node(state: ApprovalState) -> dict:
    """Check the draft against rules before human approval."""
    action = state.get("action", {})
    errors = []
    
    if not action.get("subject", "").strip():
        errors.append("subject is empty")
    if not action.get("body", "").strip():
        errors.append("body is empty")
    
    recipient = action.get("recipient", "").strip()
    if not recipient:
        errors.append("recipient is empty")
    elif action.get("action_type") == "email" and not _EMAIL_RE.match(recipient):
        errors.append(f"recipient '{recipient}' is not a valid email address")
    
    valid = not errors
    event = "validated" if valid else "validation_failed"
    record_event(state["thread_id"], event, "agent", {"errors": errors})
    
    line = f"validator: passed" if valid else f"validator: failed ({'; '.join(errors)})"
    return {"valid": valid, "validation_errors": errors, "history": [line]}


def approval_gate_node(state: ApprovalState) -> dict:
    """Pause the graph and wait for human decision."""
    # The interrupt freezes the graph here
    payload = interrupt({
        "draft": state["action"],
        "revision": state.get("revision", 0),
        "request": state.get("request", ""),
    })
    
    decision = payload.get("decision", "reject")
    feedback = payload.get("feedback", "")
    
    record_event(
        state["thread_id"], "decision", "human",
        {"decision": decision, "feedback": feedback}
    )
    
    return {
        "decision": decision,
        "feedback": feedback,
        "history": [f"human: {decision}" + (f" ({feedback})" if feedback else "")],
    }


def execute_node(state: ApprovalState) -> dict:
    """Carry out the approved action (simulated) and log to ledger."""
    action = state["action"]
    append_ledger(action, state["thread_id"])
    
    icons = {"email": "📧 EMAIL SENT", "calendar": "📅 EVENT BOOKED", "ticket": "🎫 TICKET CREATED"}
    label = icons.get(action.get("action_type"), "✅ ACTION DONE")
    result = f"{label} → {action['recipient']} | {action['subject']}"
    
    print(f"\n{result}\n")
    record_event(state["thread_id"], "executed", "agent", {"summary": action.get("summary")})
    
    return {"status": "sent", "result": result, "history": [f"execute: {result}"]}


def cancel_node(state: ApprovalState) -> dict:
    """Cancel the run without executing."""
    if state.get("decision") == "reject":
        reason = state.get("feedback") or "rejected by reviewer."
    else:
        reason = f"revision limit reached ({MAX_REVISIONS})"
    
    result = f"❌ Action cancelled: {reason}"
    print(f"\n{result}\n")
    record_event(state["thread_id"], "cancelled", "agent", {"reason": reason})
    
    return {"status": "cancelled", "result": result, "history": [f"cancel: {reason}"]}


# ============================================================
# 5. ROUTERS
# ============================================================

def route_after_validation(state: ApprovalState) -> str:
    """Route after validator: valid → approval, invalid → re-draft or cancel."""
    if state.get("valid"):
        return "approval"
    if state.get("revision", 0) < MAX_REVISIONS:
        return "planner"
    return "cancel"


def route_after_approval(state: ApprovalState) -> str:
    """Route after human decision: approve → execute, reject → cancel, edit → re-draft."""
    decision = state.get("decision")
    if decision == "approve":
        return "execute"
    if decision == "reject":
        return "cancel"
    # edit: re-draft if under revision limit
    if state.get("revision", 0) < MAX_REVISIONS:
        return "planner"
    return "cancel"


# ============================================================
# 6. GRAPH BUILDER
# ============================================================

def build_approval_graph() -> StateGraph:
    """Build and compile the human approval gate graph."""
    workflow = StateGraph(ApprovalState)
    
    # Add nodes
    workflow.add_node("planner", planner_node)
    workflow.add_node("validator", validator_node)
    workflow.add_node("approval", approval_gate_node)
    workflow.add_node("execute", execute_node)
    workflow.add_node("cancel", cancel_node)
    
    # Set entry point
    workflow.set_entry_point("planner")
    
    # Edges
    workflow.add_edge("planner", "validator")
    
    workflow.add_conditional_edges(
        "validator",
        route_after_validation,
        {
            "approval": "approval",
            "planner": "planner",
            "cancel": "cancel",
        }
    )
    
    workflow.add_conditional_edges(
        "approval",
        route_after_approval,
        {
            "execute": "execute",
            "planner": "planner",
            "cancel": "cancel",
        }
    )
    
    workflow.add_edge("execute", END)
    workflow.add_edge("cancel", END)
    
    return workflow


# ============================================================
# 7. MAIN ENTRY POINT
# ============================================================

def run_approval_gate(request: str, thread_id: str = None) -> dict:
    """
    Run the approval gate for a user request.
    
    Args:
        request: Natural language request (e.g., "Send an email to john@abc.com about the meeting")
        thread_id: Optional unique ID (auto-generated if not provided)
    
    Returns:
        dict: Final state with status, result, and action details
    """
    if not thread_id:
        thread_id = f"approval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Build graph
    workflow = build_approval_graph()
    
    # Set up checkpoint saver
    with SqliteSaver.from_conn_string(str(DB_PATH)) as checkpointer:
        graph = workflow.compile(checkpointer=checkpointer)
        
        # Initial state
        initial_state = {
            "request": request,
            "thread_id": thread_id,
            "revision": 0,
            "history": ["START"],
            "status": "pending",
        }
        
        config = {"configurable": {"thread_id": thread_id}}
        
        # Run the graph
        print(f"\n🚀 Starting approval gate: {thread_id}")
        print(f"📋 Request: {request}\n")
        
        try:
            final_state = graph.invoke(initial_state, config)
            record_event(thread_id, "completed", "agent", {"status": final_state.get("status", "unknown")})
            return final_state
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(f"❌ {error_msg}")
            record_event(thread_id, "error", "agent", {"error": error_msg})
            return {"status": "error", "result": error_msg, "thread_id": thread_id}


# ============================================================
# 8. TEST
# ============================================================

if __name__ == "__main__":
    print("🧪 Testing Human Approval Gate")
    print("=" * 60)
    
    # Test request
    test_request = "Send an email to john@abc.com about the project meeting tomorrow at 2pm"
    
    result = run_approval_gate(test_request, "test_approval_01")
    
    print("\n" + "=" * 60)
    print("RESULT:")
    print(json.dumps(result, indent=2, default=str))
    print("=" * 60)
    
    print("\n📁 Audit log:", AUDIT_LOG)
    print("📁 Ledger:", LEDGER)
    print("📁 Checkpoint DB:", DB_PATH)

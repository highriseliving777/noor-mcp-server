#!/usr/bin/env python3
"""
Noor Governance MCP Server
Exposes governance tools for any agent on any platform:
- lawful_check: Three-tier lawful/unlawful screening (CLEAR/FLAGGED/BLOCKED)
- request_approval: Human approval gate (uses run_approval_gate)
- budget_enforce: Per-agent spending limit check
- identity_register: Persistent agent identity
Uses mcp SDK, stores audit log in _meta/noor.db.

Run: python3 noor_governance_mcp.py
"""

import asyncio
import json
import sqlite3
import os
from datetime import datetime, timezone
from pathlib import Path

# MCP SDK
from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
import mcp.server.stdio
import mcp.types as types

# Import existing components
from noor_lawful_screener import screen_task, get_screener_status
from noor_approval_gate import run_approval_gate

# Configuration
DB_PATH = Path(__file__).parent / "_meta" / "noor.db"
BUDGET_LIMITS = {
    "default": 50.0,  # $50 per operation
    "construction-bid-coordinator": 100.0,
    "grant-specialist": 200.0,
}
IDENTITY_REGISTRY = Path(__file__).parent / "_meta" / "agent_identities.json"

# Initialize server
server = Server("noor-governance")

# Ensure audit table exists
def init_audit_table():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS governance_audit (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tool TEXT,
            agent_id TEXT,
            params TEXT,
            result TEXT,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()

init_audit_table()

# Helper to log audit
def log_audit(tool: str, agent_id: str, params: dict, result: dict):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO governance_audit (tool, agent_id, params, result, timestamp) VALUES (?, ?, ?, ?, ?)",
        (tool, agent_id, json.dumps(params), json.dumps(result), datetime.now(timezone.utc).isoformat())
    )
    conn.commit()
    conn.close()

# Load or init identity registry
def load_identities():
    if IDENTITY_REGISTRY.exists():
        return json.loads(IDENTITY_REGISTRY.read_text())
    return {}

def save_identities(identities: dict):
    IDENTITY_REGISTRY.write_text(json.dumps(identities, indent=2))

# --- Tool handlers ---

@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="lawful_check",
            description="Screen a task for unlawful-adjacent elements. Returns tier (CLEAR/FLAGGED/BLOCKED), categories, reasoning, and disclaimer. Deterministic — no manual review required.",
            inputSchema={
                "type": "object",
                "properties": {
                    "description": {"type": "string", "description": "Description of the business/project to check"},
                    "agent_id": {"type": "string", "description": "ID of the calling agent (for audit logging)"}
                },
                "required": ["description", "agent_id"]
            }
        ),
        types.Tool(
            name="request_approval",
            description="Request human approval for a risky action. The action will be blocked until Sam approves.",
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {"type": "string", "description": "Description of the action to be approved"},
                    "agent_id": {"type": "string", "description": "ID of the calling agent"},
                    "reason": {"type": "string", "description": "Why this action needs approval"}
                },
                "required": ["action", "agent_id", "reason"]
            }
        ),
        types.Tool(
            name="budget_enforce",
            description="Check if an agent's spending would exceed its monthly budget limit.",
            inputSchema={
                "type": "object",
                "properties": {
                    "agent_id": {"type": "string"},
                    "amount": {"type": "number"}
                },
                "required": ["agent_id", "amount"]
            }
        ),
        types.Tool(
            name="identity_register",
            description="Register or retrieve an agent's persistent identity. Returns the agent's registered info.",
            inputSchema={
                "type": "object",
                "properties": {
                    "agent_id": {"type": "string"},
                    "role": {"type": "string", "description": "Optional role if registering"}
                },
                "required": ["agent_id"]
            }
        ),
    ]

@server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict
) -> list[types.TextContent]:
    agent_id = arguments.get("agent_id", "unknown")
    result = {}

    if name == "lawful_check":
        description = arguments.get("description", "")
        if not description:
            result = {"error": "No description provided"}
        else:
            tier, screen_result = screen_task(description)
            result = {
                "tier": tier,
                "categories": screen_result.get("categories", []),
                "reasoning": screen_result.get("reasoning", ""),
                "task_hash": screen_result.get("task_hash", ""),
                "disclaimer": screen_result.get("disclaimer", ""),
                "screener_status": get_screener_status()
            }

    elif name == "request_approval":
        action = arguments.get("action", "")
        reason = arguments.get("reason", "")
        if not action:
            result = {"error": "No action provided"}
        else:
            # Generate a thread ID
            thread_id = f"gov_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            approval = run_approval_gate(action, thread_id)
            result = {
                "approval_id": thread_id,
                "status": approval.get("status", "unknown"),
                "message": approval.get("result", "Approval created.")
            }

    elif name == "budget_enforce":
        amount = arguments.get("amount", 0)
        limit = BUDGET_LIMITS.get(agent_id, BUDGET_LIMITS["default"])
        result = {
            "agent_id": agent_id,
            "limit": limit,
            "requested": amount,
            "allowed": amount <= limit,
        }

    elif name == "identity_register":
        identities = load_identities()
        role = arguments.get("role")
        if role and agent_id not in identities:
            identities[agent_id] = {"role": role, "registered": datetime.now(timezone.utc).isoformat()}
            save_identities(identities)
        result = {
            "agent_id": agent_id,
            "identity": identities.get(agent_id, {"role": "unregistered"}),
        }

    else:
        result = {"error": f"Unknown tool: {name}"}

    # Log to audit table
    log_audit(name, agent_id, arguments, result)

    # Return result as text
    return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

# --- Main entry ---
async def main():
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="noor-governance",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )

if __name__ == "__main__":
    asyncio.run(main())

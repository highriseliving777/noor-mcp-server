#!/usr/bin/env python3
"""Noor Governance MCP Server - HTTP/SSE transport."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import uvicorn
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route

from noor_governance_mcp import mcp


async def health(request):
    return JSONResponse({
        "status": "ok",
        "server": "noor-governance",
        "version": "1.0.0",
        "transport": "http+sse",
    })


app = Starlette(routes=[
    Route("/health", health),
    Mount("/", app=mcp.sse_app()),
])


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8090))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")

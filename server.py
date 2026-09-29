#!/usr/bin/env python3
"""Noor Governance MCP Server - Streamable HTTP transport."""

import os
import sys
from contextlib import asynccontextmanager
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
        "transport": "streamable-http",
        "endpoint": "/mcp",
    })


child = mcp.streamable_http_app()


@asynccontextmanager
async def lifespan(app):
    async with mcp.session_manager.run():
        yield


app = Starlette(
    routes=[
        Route("/health", health),
        Mount("/", app=child),
    ],
    lifespan=lifespan,
)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8090))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")

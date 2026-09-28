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

from mcp.server.sse import SseServerTransport
from mcp.server.models import InitializationOptions
from mcp.server import NotificationOptions

from noor_governance_mcp import server

sse = SseServerTransport("/messages/")


async def handle_sse(request):
    async with sse.connect_sse(
        request.scope, request.receive, request._send
    ) as streams:
        await server.run(
            streams[0],
            streams[1],
            InitializationOptions(
                server_name="noor-governance",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


async def health(request):
    return JSONResponse({
        "status": "ok",
        "server": "noor-governance",
        "version": "1.0.0",
        "transport": "http+sse",
    })


app = Starlette(routes=[
    Route("/health", health),
    Route("/sse", handle_sse),
    Mount("/messages/", app=sse.handle_post_message),
])


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8090))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")

# server.py
# MCP server to expose product discovery tools via FastAPI

from __future__ import annotations

import json
from typing import Any, Dict
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn

from tools.rag_search import (
    rag_search_tool,
    RAG_SEARCH_INPUT_SCHEMA,
    RAG_SEARCH_OUTPUT_SCHEMA,
)
from tools.web_search import (
    web_search_tool,
    WEB_SEARCH_INPUT_SCHEMA,
    WEB_SEARCH_OUTPUT_SCHEMA,
)

app = FastAPI(title="MCP Product Tools Server")

# ---------- MCP-like discovery endpoint ----------
@app.get("/tools")
async def list_tools() -> JSONResponse:
    """
    Discovery endpoint: returns available tools and their JSON schemas.

    This is not the full MCP spec but provides the same information:
      - tool names
      - input schema
      - output schema
    """
    tools = {
        "rag.search": {
            "input_schema": RAG_SEARCH_INPUT_SCHEMA,
            "output_schema": RAG_SEARCH_OUTPUT_SCHEMA,
        },
        "web.search": {
            "input_schema": WEB_SEARCH_INPUT_SCHEMA,
            "output_schema": WEB_SEARCH_OUTPUT_SCHEMA,
        },
    }
    return JSONResponse(content=tools)

# ---------- Tool invocation endpoints ----------
@app.post("/tools/rag.search")
async def call_rag_search(payload: Dict[str, Any]) -> JSONResponse:
    """
    Call the rag.search tool with JSON arguments.
    """
    result = rag_search_tool(payload)
    print("rag result: ", result)
    return JSONResponse(content=result)

@app.post("/tools/web.search")
async def call_web_search(payload: Dict[str, Any]) -> JSONResponse:
    """
    Call the web.search tool with JSON arguments.
    """
    result = web_search_tool(payload)
    print("web result: ", result)
    return JSONResponse(content=result)

def main():
    uvicorn.run(app, host="0.0.0.0", port=8001)

if __name__ == "__main__":
    main()

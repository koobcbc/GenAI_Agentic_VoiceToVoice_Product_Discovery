# server.py
# MCP server to expose product discovery tools via FastAPI

from __future__ import annotations

import json
from typing import Any, Dict
from fastapi import FastAPI
# from fastapi.responses import JSONResponse
import uvicorn
import threading
import time

from flask import Flask, request, Response, jsonify
from flask_cors import CORS
import requests
from termcolor import colored

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

# Server configuration
SERVER_HOST = '0.0.0.0'
SERVER_PORT = 8001
SERVER_URL = f'http://{SERVER_HOST}:{SERVER_PORT}'

# Create Flask app
app = Flask(__name__)
# app = FastAPI(title="MCP Product Tools Server")
CORS(app)  # Enable CORS for cross-origin requests


MCP_TOOLS = [
    {
        "name": "web_search_tool",
        "description": "Perform a safe web search using an external search API (e.g., Brave/Serper/Bing) to retrieve up-to-date information about products or general topics. Returns a list of results with `title`, `url`, `snippet`, and, when available, `price` and `availability`. Use this tool when the user's request requires live data or information not covered by the private catalog.",
        "inputSchema": WEB_SEARCH_INPUT_SCHEMA,
        "outputSchema": WEB_SEARCH_OUTPUT_SCHEMA

    },
    {
        "name": "rag_search_tool",
        "description": "Query the local Amazon 2020 vector database for product information from the private catalog. Returns a list of matching items with `sku`, `title`, `price`, `rating`, and, when available, `brand`, `ingredients`, and `doc_id`. Use this tool when the user's request should be answered from the internal Amazon 2020 slice rather than the public web.",
        "inputSchema": RAG_SEARCH_INPUT_SCHEMA,
        "outputSchema": RAG_SEARCH_OUTPUT_SCHEMA
    },
]

def create_sse_message(data: Dict[str, Any]) -> str:
    """
    Format a message for Server-Sent Events (SSE).
    SSE format: 'data: {json}\n\n'
    """
    return f"data: {json.dumps(data)}\n\n"

def handle_initialize(message: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle MCP initialize request.
    This is the first message in the MCP protocol handshake.
    """
    return {
        "jsonrpc": "2.0",
        "id": message.get("id"),
        "result": {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {},  # We support tools
            },
            "serverInfo": {
                "name": "customer-management-server",
                "version": "1.0.0"
            }
        }
    }

def handle_tools_list(message: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle tools/list request.
    Returns the list of available tools.
    """
    return {
        "jsonrpc": "2.0",
        "id": message.get("id"),
        "result": {
            "tools": MCP_TOOLS
        }
    }

def handle_tools_call(message: Dict[str, Any]) -> Dict[str, Any]:
    params = message.get("params", {})
    tool_name = params.get("name")
    arguments = params.get("arguments", {})

    tool_functions = {
        "rag_search_tool": rag_search_tool,
        "web_search_tool": web_search_tool,
    }

    if tool_name not in tool_functions:
        return {
            "jsonrpc": "2.0",
            "id": message.get("id"),
            "error": {
                "code": -32601,
                "message": f"Tool not found: {tool_name}"
            }
        }

    try:
        result = tool_functions[tool_name](**arguments)

        # MCP-COMPLIANT RETURN FORMAT
        return {
            "jsonrpc": "2.0",
            "id": message.get("id"),
            "result": {
                # Required for LLMs to "see" something
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(result)
                    }
                ],

                # REQUIRED for LangChain because outputSchema exists
                "structuredContent": result
            }
        }

    except Exception as e:
        return {
            "jsonrpc": "2.0",
            "id": message.get("id"),
            "error": {
                "code": -32603,
                "message": f"Tool execution error: {str(e)}"
            }
        }

def process_mcp_message(message: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process an MCP message and route it to the appropriate handler.
    """
    method = message.get("method")
    
    print(f"Processing MCP message: {method}")
    
    if method == "initialize":
        return handle_initialize(message)
    elif method == "tools/list":
        return handle_tools_list(message)
    elif method == "tools/call":
        return handle_tools_call(message)
    else:
        return {
            "jsonrpc": "2.0",
            "id": message.get("id"),
            "error": {
                "code": -32601,
                "message": f"Method not found: {method}"
            }
        }

# Flask Routes

@app.route('/mcp', methods=['POST'])
def mcp_endpoint():
    """
    Main MCP endpoint for MCP communication.
    Receives MCP messages and streams responses using Server-Sent Events.
    """
    # Get the MCP message from the request BEFORE entering the generator
    # This must be done in the request context
    message = request.get_json()
    
    def generate():
        try:
            print(f"📥 Received MCP message: {message.get('method')}")
            
            # Process the message
            response = process_mcp_message(message)
            
            print(f"📤 Sending MCP response")
            
            # Send the response as SSE
            yield create_sse_message(response)
            
        except Exception as e:
            error_response = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {
                    "code": -32700,
                    "message": f"Parse error: {str(e)}"
                }
            }
            yield create_sse_message(error_response)
    
    return Response(generate(), mimetype='text/event-stream')

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint to verify server is running."""
    return jsonify({
        "status": "healthy",
        "server": "customer-management-mcp-server",
        "version": "1.0.0"
    })


def run_server():
    """Run the Flask server in a separate thread."""
    global server_running
    server_running = True
    # uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT)
    app.run(host=SERVER_HOST, port=SERVER_PORT, debug=False, use_reloader=False)

def start_server():
    # Server state
    server_thread = None
    server_running = False

    """Start the MCP server in a background thread."""
    server_thread, server_running
    
    if server_thread and server_thread.is_alive():
        print(colored("⚠️  Server is already running!", "yellow"))
        return
    
    print(colored("🚀 Starting MCP server...", "cyan"))
    
    # Start server in background thread
    server_thread = threading.Thread(target=run_server, daemon=False)
    server_thread.start()
    
    # Wait for server to start
    time.sleep(2)
    
    # Check if server is healthy
    try:
        response = requests.get(f'{SERVER_URL}/health', timeout=5)
        if response.status_code == 200:
            print(colored("✅ MCP Server is running!", "green"))
            print(colored(f"📍 Local URL: {SERVER_URL}", "cyan"))
        
    except Exception as e:
        print(colored(f"❌ Failed to connect to server: {e}", "red"))

def main():
    run_server()
    # start_server() # running s
    # uvicorn.run(app, host="0.0.0.0", port=8001)

if __name__ == "__main__":
    # run_server()
    # start_server()
    main()

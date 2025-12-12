import asyncio
from langgraph.graph import StateGraph, END, START
import os
from dotenv import load_dotenv
from typing import Optional, Dict, Any, List
from langchain.agents.factory import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
import requests
from graph.pretty_print import debug_all

SERVER_HOST = '0.0.0.0'
SERVER_PORT = 8001
SERVER_URL = f'http://{SERVER_HOST}:{SERVER_PORT}'
MCP_SERVERS = {
        "my_server": {
            "url": SERVER_URL+"/mcp",
            "transport": "http",
        },
    }

# ============================================================
# Environment & LLM
# ============================================================

load_dotenv()

# Model provider selection: "openai" or "ollama" (free)
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "openai").lower()

if MODEL_PROVIDER == "openai":
    from langchain_openai import ChatOpenAI
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is not set in .env file")
    # Model configuration - can be overridden via environment variable
    # Available models: gpt-4, gpt-4o, gpt-4-turbo, gpt-4o-mini, gpt-3.5-turbo
    MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4")
    llm = ChatOpenAI(model=MODEL_NAME, temperature=0)
elif MODEL_PROVIDER == "ollama":
    from langchain_ollama import ChatOllama
    # Ollama models (free, local): llama3.1, llama3.2, mistral, qwen2.5, etc.
    MODEL_NAME = os.getenv("OLLAMA_MODEL", "llama3.1")
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    llm = ChatOllama(model=MODEL_NAME, base_url=OLLAMA_BASE_URL, temperature=0)
else:
    raise ValueError(f"Unknown MODEL_PROVIDER: {MODEL_PROVIDER}. Use 'openai' or 'ollama'")

# Where your MCP FastAPI server (server.py) is running
MCP_BASE_URL = os.getenv("MCP_BASE_URL", "http://0.0.0.0:8001")


# ============================================================
# MCP Helper
# ============================================================

def call_mcp_tool(name: str, arguments: Dict[str, Any]) -> Any:
    """
    Call FastAPI MCP tool endpoints defined in server.py.

    server.py exposes:
      - POST /tools/rag.search
      - POST /tools/web.search

    For your current tools:
      - rag.search returns: {"products": [...]}
      - web.search returns: {"results": [...], "note": ...}
    """
    endpoint = f"{MCP_BASE_URL}/tools/{name}"
    print(f"[MCP] Calling {endpoint} with args={arguments}")
    resp = requests.post(endpoint, json=arguments, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    print(f"[MCP] tool={name} response type: {type(data)}")
    return data


# ============================================================
# LangGraph State
# ============================================================

class AgentState(dict):
    input: str
    response: str
    done: bool

    intent: Optional[str]
    plan: Optional[str]
    knowledge: Optional[str]   # text based on MCP results

    retrieved_context: Optional[List[Dict[str, Any]]]


# ============================================================
# Router Node
# ============================================================

def router_node(state: AgentState) -> Dict[str, Any]:
    print("ROUTER NODE Started")
    user_input = state["input"]
    system_prompt = f"""You are the Router Agent. Read the user request and:
1) Identify the main task.
2) Extract constraints (budget, materials, brands).
3) Detect any safety concerns and flag them if necessary.

USER REQUEST:
{user_input}

Your response MUST follow this format:

* Task:
* Constraints:
  - Budget:
  - Material:
  - Brand:
* Safety Flags: (Yes/No — if Yes, provide a brief reason)
"""

    response = llm.invoke(system_prompt)
    print("router response: ", response.content)
    return {"intent": response.content}


# ============================================================
# Planner Node
# ============================================================

def planner_node(state: AgentState) -> Dict[str, Any]:
    print("Planner NODE Started")
    intent = state.get("intent", "No intent provided")
    query = state.get("input", "No query provided")
    system_prompt = system_prompt = f"""
You are the Planning Agent. Your task is to analyze the User Query and Intent and produce a precise retrieval plan that the Retrieval Agent can execute.

### Core Responsibilities
1) **Select the data source(s):** `private` or `both`
   • Choose **both** if the user asks for information that is *current, live, updated, latest, now,* or *real-time*.  
   • Choose **private** for all other queries.

2) **Identify the fields** that must be retrieved  
   (e.g., product descriptions, prices, ratings).

3) **Extract constraints** from the User Query and express them using valid ChromaDB filter syntax.  
   Allowed fields: `price`, `rating`.  
   Example filters:  
   • `"price": {{"$lt": 30}}`  
   • `"rating": {{"$gte": 3.5}}`

4) **Specify comparison criteria** that should be used to evaluate retrieved options  
   (e.g., price, rating).

---

### User Query
{query}

### Interpreted Intent
{intent}

### REQUIRED OUTPUT FORMAT (strict — produce ONLY this structure):
* Data Source (private / both):
* Fields to Retrieve:
* Constraints:
* Comparison Criteria:
"""

    response = llm.invoke(system_prompt)
    print("planner response: ", response.content)
    return {"plan": response.content}


# ============================================================
# Retriever Node (directly calls retrieval_tool → MCP)
# ============================================================

from langchain_core.messages import AIMessage, ToolMessage

def get_final_ai_message(response):
    msgs = response["messages"]
    for msg in reversed(msgs):
        if isinstance(msg, AIMessage):
            return msg
    return None

def get_tool_messages(response):
    msgs = response["messages"]
    return [m for m in msgs if isinstance(m, ToolMessage)]


async def retrieve_node(state: AgentState):
    print("Retriever NODE Started")

    # final_ai, tool_msgs = await async_get_retrieve_result(state)
    client = MultiServerMCPClient(MCP_SERVERS)
    tools = await client.get_tools()

    agent = create_agent(
        model=llm,
        tools=tools
    )

    plan = state.get("plan", "No plan provided")
    query = state.get("input", "No query provided")
    system_prompt = f"""
You are the Retrieval Agent. Your role is to fetch EXACTLY the information specified in the plan—no reasoning, no interpretation.

### Rules
• You MUST retrieve information according to the Data Source specified in the plan.  
• If Data Source includes **private**, use `rag.search` to query the private catalog.  
• If Data Source includes **both**, call BOTH `rag.search` and `web.search.  
• If Constraints are provided, use them to filter the rag.search results.
• Return ONLY the raw or minimally structured data retrieved—do NOT summarize, explain, or modify it.  
• If no data can be retrieved, respond EXACTLY with: **"No data found."**  
• If the plan indicates no retrieval is needed, respond EXACTLY with: **"Retrieval not applicable."**

### User Query
{query}

### Plan Details
{plan}

### Required Response Format
* Retrieved data:
<insert raw retrieved data OR the exact required message>
"""

    response = await agent.ainvoke({"messages": [{"role": "user", "content": system_prompt}]})
    # debug_all(state, system_prompt, response)

    final_ai = get_final_ai_message(response)
    tool_msgs = get_tool_messages(response)

    print(final_ai)

    return {
        "knowledge": final_ai.content if final_ai else None,
        "retrieved_context": [msg.content for msg in tool_msgs]
    }

# ============================================================
# Answer / Critic Node
# ============================================================

def answer_critic_node(state: AgentState) -> Dict[str, Any]:
    print("Answer/Critic NODE Started")
    user_input = state["input"]
    knowledge = state.get("knowledge", "No knowledge provided")
    system_prompt = f"""
You are the Answer Critic Agent. Your job is to synthesize a concise, well-grounded, citation-backed, and safe final answer using ONLY the Retrieved Knowledge.

### Core Rules
• Provide a final answer that directly and accurately responds to the User Request.  
• You MUST ground every statement in the Retrieved Knowledge — no new facts, no assumptions, no hallucinations.  
• Cite evidence for each key point using:
    - Document IDs for rag.search results  
    - URLs for web.search results  
• If the Retrieved Knowledge contains harmful, unsafe, or incomplete information, clearly flag it.  

### User Request
{user_input}

### Retrieved Knowledge
{knowledge}

### REQUIRED OUTPUT FORMAT (follow EXACTLY)

* Final Answer:
- Concise summary of the response based on the Retrieved Knowledge.

* Cited Sources:
  - <evidence snippet 1>  [source: rag.search | doc_id: <id>]
  - <evidence snippet 2>  [source: web.search | url: <url>]
  - <evidence snippet 3>  [source: rag.search | doc_id: <id>]

* Safety Flags: Yes/No  
  - <If Yes, provide a single brief sentence explaining the issue>
"""
    
    response = llm.invoke(system_prompt)
    # print("answer response: ", response.content)
    return {"response": response.content, "done": True}


# ============================================================
# Build LangGraph
# ============================================================

graph = StateGraph(AgentState)

graph.add_node("Router", router_node)
graph.add_node("Planner", planner_node)
graph.add_node("Retriver", retrieve_node)
graph.add_node("Answer", answer_critic_node)

graph.add_edge(START, "Router")
graph.add_edge("Router", "Planner")
graph.add_edge("Planner", "Retriver")
graph.add_edge("Retriver", "Answer")

graph.add_conditional_edges(
    "Answer",
    lambda state: "END" if state.get("done") else "Planner",
    {
        "Planner": "Planner",
        "END": END,
    },
)

app = graph.compile()

# if __name__ == "__main__":
#     result = asyncio.run(
#         app.ainvoke({"input": "I want to find a stuffed animal for kids less than $30 in private catalog and live web search"})
#     )

#     print("\n================ FINAL rag data ===============")
#     print(result.get('knowledge'))

#     print("\n================ FINAL ANSWER ===============")
#     print(result.get("response"))
#     print("============================================\n")

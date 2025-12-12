# AI Agent Assistant

A multi-agent system built with LangGraph that processes user queries through a pipeline of specialized agents. The application supports text and voice input, and provides intelligent responses using RAG (Retrieval-Augmented Generation) and web search capabilities.

## Features

- Text and voice input (using Whisper ASR)
- Multi-agent reasoning pipeline
- RAG-based product search
- Web search integration
- Text-to-speech output (optional)

## Prerequisites

- Python 3.8-3.12
- ffmpeg (for audio processing)
- OpenAI API key (for OpenAI models) or Ollama installed locally (for free local models)
- Serper API Key

## Installation

1. Clone the repository and navigate to the project directory.

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install ffmpeg:
```bash
# macOS
brew install ffmpeg

# Linux
sudo apt-get install ffmpeg
```

5. Create a `.env` file in the project root:
```env
OPENAI_API_KEY=your_api_key_here
MODEL_PROVIDER=openai  # or "ollama" for local models
OPENAI_MODEL=gpt-4  # optional, default: gpt-4
OLLAMA_MODEL=llama3.1  # optional, default: llama3.1
OLLAMA_BASE_URL=http://localhost:11434  # optional, default: http://localhost:11434
SERPER_API_KEY=your_serper_api_key_here
MCP_BASE_URL=http://0.0.0.0:8001
```

## Running the Application

1. Start the MCP server (required for tool access):
```bash
python agents/mcp_server.py
```
The MCP server will run on `http://0.0.0.0:8001`.

2. In a separate terminal, start the Streamlit app:
```bash
streamlit run streamlit_app.py
```

The application will open in your browser at `http://localhost:8501`.

## Usage

1. Enter your query in the text input field, or use voice input by uploading an audio file or recording.
2. The query is processed through the agent pipeline:
   - Router Agent identifies the task
   - Planner Agent creates a retrieval plan
   - Retriever Agent fetches relevant data
   - Answer/Critic Agent synthesizes the final response
3. View the response in the conversation history.

## Model Schema

### AgentState

The state object passed between agents in the LangGraph pipeline:

```
{
  "input": str,              # User query
  "response": str,           # Final response
  "done": bool,             # Completion flag
  "intent": Optional[str],   # Task identification from Router
  "plan": Optional[str],    # Retrieval plan from Planner
  "knowledge": Optional[str], # Retrieved data from Retriever
  "retrieved_context": Optional[List[Dict[str, Any]]] # Tool messages from retrieval
}
```

### RAG Search

**Input (RagSearchInput):**
```
{
  "query": str,                    # Required: Natural language query
  "top_k": int,                    # Optional: Number of results (1-20, default: 3)
  "price": Optional[object],       # Optional: ChromaDB-style price filter
                                    #   Example: {"$lt": 30} or {"$gte": 10}
                                    #   Supported operators: $lt, $lte, $gt, $gte, $eq, $in
  "rating": Optional[object],      # Optional: ChromaDB-style rating filter (0-5)
                                    #   Example: {"$gte": 3.5}
                                    #   Supported operators: $lt, $lte, $gt, $gte, $eq
  "brand": Optional[str]           # Optional: Brand name filter
}
```

**Output (RagSearchOutput):**
```
{
  "products": [
    {
      "sku": str,                  # Required: Product SKU
      "title": str,                # Required: Product title
      "doc_id": str,               # Required: Document ID
      "price": Optional[float],
      "rating": Optional[float],
      "brand": Optional[str],
      "category": Optional[str],
      "score": Optional[float]      # Similarity score
    }
  ]
}
```

### Web Search

**Input (WebSearchInput):**
```
{
  "query": str,                    # Required: Search query
  "max_results": int,              # Optional: Max results (1-10, default: 3)
  "mode": str                      # Optional: "shopping" or "web" (default: "shopping")
}
```

**Output (WebSearchOutput):**
```
{
  "results": [
    {
      "title": Optional[str],
      "url": Optional[str],
      "snippet": Optional[str],
      "price": Optional[float|str],
      "availability": Optional[str],
      "rating": Optional[float],
      "rating_count": Optional[int]
    }
  ],
  "note": Optional[str]            # Error or info message
}
```

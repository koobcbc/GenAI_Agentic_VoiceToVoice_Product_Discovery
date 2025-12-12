# GenAI Agentic Voice-to-Voice Product Discovery

A sophisticated multi-agent system built with LangGraph that enables voice-to-voice product discovery through an intelligent pipeline of specialized agents. The application processes natural language queries (text or voice) and provides intelligent product recommendations using RAG (Retrieval-Augmented Generation) and web search capabilities.

## 🎯 Features

- **🎤 Voice Input**: Automatic Speech Recognition (ASR) using OpenAI Whisper
- **🔊 Voice Output**: Text-to-Speech (TTS) using OpenAI's GPT-4o-mini-tts
- **📝 Text Input**: Traditional text-based query interface
- **🤖 Multi-Agent Pipeline**: Specialized agents working in sequence:
  - **Router Agent**: Identifies tasks and extracts constraints
  - **Planner Agent**: Creates retrieval strategies
  - **Retriever Agent**: Fetches relevant data from multiple sources
  - **Answer/Critic Agent**: Synthesizes final responses with citations
- **🔍 RAG-Based Search**: Semantic search over private product catalog using ChromaDB
- **🌐 Web Search Integration**: Live product comparison via Serper.dev API
- **💬 Streamlit UI**: Interactive web interface for easy interaction
- **🔄 MCP Server**: FastAPI-based Model Context Protocol server for tool exposure
- **🔌 Flexible LLM Support**: Works with OpenAI or local Ollama models

## 🏗️ Architecture

### System Components
<img width="621" height="1411" alt="agent pipeline drawio" src="https://github.com/user-attachments/assets/d48df720-7651-4684-b7e7-2b455e897e49" />


```
┌─────────────────┐
│  User Input     │ (Text/Voice)
└────────┬────────┘
         │
    ┌────▼────┐
    │ Whisper │ (ASR - if voice)
    └────┬────┘
         │
    ┌────▼──────────────────────────────────────────────┐
    │      LangGraph Agent Pipeline                     │
    │  ┌──────────┐  ┌──────────┐                       │
    │  │  Router  │─▶│  Planner │                       │
    │  └──────────┘  └─────┬────┘                       │
    │                      │                            │
    │                 ┌────▼────┐      ┌─────────────┐  │
    │                 │Retriever│◀────▶│   Tools     │  │
    │                 └─────┬───┘      │  ┌──────┐   │  │
    │                       │          │  │ RAG  │   │  │ (ChromaDB)
    │                       │          │  └──────┘   │  │
    │                       │          │  ┌──────┐   │  │
    │                       │          │  │ Web  │   │  │ (Serper.dev)
    │                       │          │  └──────┘   │  │
    │                       │          └─────────────┘  │
    │                  ┌────▼────┐                      │
    │                  │  Answer │                      │
    │                  └─────────┘                      │
    └─────────────────────┬─────────────────────────────┘
                          │
                    ┌─────▼─────┐
                    │   TTS     │
                    └───────────┘
```

### Agent Pipeline Flow

1. **Router Agent**: Analyzes user input to identify:
   - Main task/query
   - Constraints (budget, materials, brands)
   - Safety concerns

2. **Planner Agent**: Creates retrieval plan:
   - Data source selection (private catalog, live web, or both)
   - Fields to retrieve
   - Comparison criteria

3. **Retriever Agent**: Executes retrieval:
   - Calls RAG search on private ChromaDB catalog
   - Optionally calls web search for live price comparison
   - Aggregates results

4. **Answer/Critic Agent**: Synthesizes response:
   - Creates final answer using retrieved knowledge
   - Cites specific data points
   - Flags safety concerns if present

### MCP Server

The MCP (Model Context Protocol) server exposes tools via FastAPI:
- `POST /tools/rag.search` - Semantic product search
- `POST /tools/web.search` - Web/shopping search
- `GET /tools` - Tool discovery endpoint

## 📋 Prerequisites

- **Python**: 3.8-3.12
- **ffmpeg**: For audio processing
- **API Keys**:
  - OpenAI API key (for LLM and TTS) OR Ollama (for local LLM)
  - Serper API Key (for web search)
- **ChromaDB**: Vector database for product catalog (included in project)

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd GenAI_Agentic_VoiceToVoice_Product_Discovery
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install ffmpeg

**macOS:**
```bash
brew install ffmpeg
```

**Linux:**
```bash
sudo apt-get install ffmpeg
```

**Windows:**
Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH.

### 5. Optional: Install Audio Libraries for Recording

For microphone recording functionality:
```bash
pip install soundfile sounddevice
```

### 6. Set Up Environment Variables

Create a `.env` file in the project root:

```env
# LLM Configuration
OPENAI_API_KEY=your_openai_api_key_here
MODEL_PROVIDER=openai  # or "ollama" for local models
OPENAI_MODEL=gpt-4  # optional, default: gpt-4
OLLAMA_MODEL=llama3.1  # optional, default: llama3.1
OLLAMA_BASE_URL=http://localhost:11434  # optional, default: http://localhost:11434

# Web Search
SERPER_API_KEY=your_serper_api_key_here

# MCP Server
MCP_BASE_URL=http://0.0.0.0:8001
```

### 7. Set Up Product Catalog (Optional)

If you need to rebuild the product index:

1. Ensure `agents/tools/data/products.parquet` exists
2. Run the build index notebook: `2. build_index.ipynb`
3. The ChromaDB collection will be created at `agents/tools/data/chroma_toys/`

## 🎮 Running the Application

### Start the MCP Server

First, start the MCP server (required for tool access):

```bash
cd agents
python mcp_server.py
```

The server will run on `http://0.0.0.0:8001`.

### Start the Streamlit App

In a new terminal:

```bash
streamlit run streamlit_app.py
```

The application will open in your browser at `http://localhost:8501`.

### Alternative: Command-Line Interface

You can also run the voice-to-voice pipeline directly:

```bash
cd agents
python main.py
```

This will:
1. Record/load audio from `agents/recording/recording0.wav`
2. Transcribe using Whisper
3. Process through the agent graph
4. Output text response
5. Generate TTS audio output

## 📁 Project Structure

```
GenAI_Agentic_VoiceToVoice_Product_Discovery/
├── agents/
│   ├── graph/
│   │   ├── __init__.py
│   │   └── graph.py              # LangGraph multi-agent pipeline
│   ├── recording/
│   │   └── recording0.wav        # Sample audio input
│   └── tools/
│       ├── data/
│       │   ├── chroma_toys/      # ChromaDB persistent vector store
│       │   ├── products.parquet  # Cleaned product metadata
│       ├── rag_search.py         # RAG search (Chroma + filters)
│       ├── web_search.py         # Serper.dev web/shopping search tool
│       └── __init__.py
│    ├── data_analysis.ipynb       # Exploratory analysis (optional)
│    ├── llm_judge.ipynb           # LLM evaluation or testing notebook
│    ├── main.py                   # Potential CLI entry (if used)
│    ├── mcp_server.py             # FastAPI MCP server exposing rag.search & web.search
│    ├── tts.py                    # Text-to-speech implementation (gpt-4o-mini-tts)
│    ├── whisper_ars.ipynb         # Whisper ASR exploration notebook
│    └── whisper_ars.py            # Whisper “medium” ASR script
│
├── .env.example                  # Environment variable template
├── .gitignore
│
├── 1. data_preprocessing.ipynb   # Clean Amazon dataset → features/ingredients/brand
├── 2. build_index.ipynb          # Build embeddings + Chroma index
├── 3. rag_logic.ipynb            # Core RAG Engine
├── 4. eval_rag.ipynb             # Recall@K and custom query evaluation
│
├── README.md                     # Project documentation
├── requirements.txt              # Default environment
├── requirements_python12.txt     # Python 3.12 compatible environment
└── streamlit_app.py              # Streamlit UI for voice-to-voice demo

```

## 💡 Usage Examples

### Example 1: Text Query via Streamlit

1. Open the Streamlit app
2. Select "Text Input"
3. Enter: "I need an eco-friendly puzzle under $15"
4. Click "Submit"
5. View the agent's response with product recommendations

### Example 2: Voice Query

1. Select "Record Audio" or "Audio Upload"
2. Record/upload your query (e.g., "Find me a safe toy for a 3-year-old")
3. The system will:
   - Transcribe your voice
   - Process through agents
   - Return recommendations
   - Optionally speak the response (if TTS enabled)

### Example 3: Direct API Usage

```python
from agents.graph.graph import app

result = app.invoke({
    "input": "I need a stainless steel cleaner under $20"
})

print(result['response'])
```

## 🔧 Configuration

### Model Provider Options

**OpenAI (Cloud):**
- Models: `gpt-4`, `gpt-4o`, `gpt-4-turbo`, `gpt-4o-mini`, `gpt-3.5-turbo`
- Requires: `OPENAI_API_KEY`

**Ollama (Local):**
- Models: `llama3.1`, `llama3.2`, `mistral`, `qwen2.5`, etc.
- Requires: Ollama installed and running locally
- Set: `MODEL_PROVIDER=ollama` in `.env`

### Whisper Model

The default Whisper model is `medium`. To change:
- Edit `agents/whisper_ars.py` or `streamlit_app.py`
- Options: `tiny`, `base`, `small`, `medium`, `large`

### TTS Configuration

TTS uses OpenAI's `gpt-4o-mini-tts` model with voice `coral`. To customize:
- Edit `agents/tts.py`
- Available voices: `alloy`, `echo`, `fable`, `onyx`, `nova`, `shimmer`, `coral`

## 📊 Data Models

### AgentState

The state object passed between agents:

```python
{
    "input": str,              # User query
    "response": str,           # Final response
    "done": bool,              # Completion flag
    "intent": Optional[str],   # Task identification from Router
    "plan": Optional[str],     # Retrieval plan from Planner
    "knowledge": Optional[str]  # Retrieved data from Retriever
}
```

### RAG Search

**Input (RagSearchInput):**
```json
{
    "query": "eco-friendly puzzle",
    "top_k": 3,
    "max_price": 15.0,
    "min_rating": 4.0,
    "brand": "LEGO"
}
```

**Output (RagSearchOutput):**
```json
{
    "products": [
        {
            "sku": "PROD001",
            "title": "Eco-Friendly Puzzle Set",
            "doc_id": "PROD001",
            "price": 12.99,
            "rating": 4.5,
            "brand": "LEGO",
            "category": "Puzzles",
            "score": 0.85
        }
    ]
}
```

### Web Search

**Input (WebSearchInput):**
```json
{
    "query": "eco-friendly puzzle under $15",
    "max_results": 5,
    "mode": "shopping"
}
```

**Output (WebSearchOutput):**
```json
{
    "results": [
        {
            "title": "Eco-Friendly Puzzle - Amazon",
            "url": "https://amazon.com/...",
            "snippet": "Sustainable puzzle made from...",
            "price": 13.99,
            "availability": "In stock",
            "rating": 4.3,
            "rating_count": 1250
        }
    ],
    "note": null
}
```

## 🛠️ Development

### Adding New Tools

1. Create tool function in `agents/tools/`
2. Define input/output schemas
3. Add endpoint to `agents/mcp_server.py`
4. Update `retrieval_tool()` in `agents/graph/graph.py` if needed

### Customizing Agents

Edit agent prompts in `agents/graph/graph.py`:
- `router_node()`: Task identification logic
- `planner_node()`: Retrieval planning logic
- `answer_critic_node()`: Response synthesis logic

### Testing

Test individual components:

```python
# Test RAG search
from agents.tools.rag_search import rag_search_tool
result = rag_search_tool({"query": "puzzle", "top_k": 3})

# Test web search
from agents.tools.web_search import web_search_tool
result = web_search_tool({"query": "puzzle", "max_results": 3})
```

## 🐛 Troubleshooting

### MCP Server Connection Error

- Ensure MCP server is running: `python agents/mcp_server.py`
- Check `MCP_BASE_URL` in `.env` matches server address
- Verify port 8001 is not in use

### Whisper Model Loading Issues

- Ensure sufficient disk space (models can be 1-3GB)
- Check internet connection for first-time download
- Try smaller model (`base` instead of `medium`)

### ChromaDB Collection Not Found

- Run `2. build_index.ipynb` to create the collection
- Verify `agents/tools/data/chroma_toys/` exists
- Check collection name matches `products_toys` in `rag_search.py`

### Audio Recording Not Working

- Install audio libraries: `pip install soundfile sounddevice`
- Check microphone permissions
- Verify audio device is available

import streamlit as st
import sys
import os
from pathlib import Path
import asyncio
import tempfile
import numpy as np
from io import BytesIO
import time

# Optional audio libraries - only needed for recording functionality
try:
    import soundfile as sf
    import sounddevice as sd
    AUDIO_LIBS_AVAILABLE = True
except ImportError:
    AUDIO_LIBS_AVAILABLE = False
    sf = None
    sd = None

# Add agents directory to path
agents_path = Path(__file__).parent / "agents"
sys.path.insert(0, str(agents_path))
sys.path.insert(0, str(Path(__file__).parent))

# Import agents modules - try multiple import paths
try:
    from graph.graph import app as graph_app
    from tts import run_tts
except ImportError:
    try:
        from agents.graph.graph import app as graph_app
        from agents.tts import run_tts
    except ImportError:
        # Last resort: direct path
        import importlib.util
        graph_path = agents_path / "graph" / "graph.py"
        tts_path = agents_path / "tts.py"
        
        spec_graph = importlib.util.spec_from_file_location("graph_module", graph_path)
        graph_module = importlib.util.module_from_spec(spec_graph)
        spec_graph.loader.exec_module(graph_module)
        graph_app = graph_module.app
        
        spec_tts = importlib.util.spec_from_file_location("tts_module", tts_path)
        tts_module = importlib.util.module_from_spec(spec_tts)
        spec_tts.loader.exec_module(tts_module)
        run_tts = tts_module.run_tts

# Import whisper directly for flexible audio path handling
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="AI Agent Assistant",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'processing' not in st.session_state:
    st.session_state.processing = False

def process_query(user_input: str, use_tts: bool = False):
    """Process user query through the graph and optionally use TTS"""
    try:
        # Process through graph
        result = asyncio.run(graph_app.ainvoke({"input": user_input}))
        response = result.get('response', 'No response generated')
        
        # Optionally run TTS
        if use_tts:
            try:
                asyncio.run(run_tts(response))
            except Exception as e:
                st.warning(f"TTS failed: {str(e)}")
        
        return response
    except Exception as e:
        return f"Error processing query: {str(e)}"

@st.cache_resource
def load_whisper_model():
    """Load Whisper model (cached for performance)"""
    if not WHISPER_AVAILABLE:
        return None
    try:
        return whisper.load_model("medium")
    except Exception as e:
        st.error(f"Failed to load Whisper model: {str(e)}")
        return None

def transcribe_audio(audio_file):
    """Transcribe audio file using Whisper"""
    if not WHISPER_AVAILABLE:
        return "Error: Whisper is not available"
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(audio_file.read())
            tmp_path = tmp_file.name
        
        # Load model and transcribe
        model = load_whisper_model()
        if model is None:
            return "Error: Could not load Whisper model"
        
        result = model.transcribe(tmp_path)
        
        # Clean up
        os.unlink(tmp_path)
        
        return result["text"]
    except Exception as e:
        return f"Error transcribing audio: {str(e)}"

def record_audio(duration=5):
    """Record audio from microphone"""
    if not WHISPER_AVAILABLE:
        return "Error: Whisper is not available"
    
    if not AUDIO_LIBS_AVAILABLE:
        return "Error: soundfile and sounddevice are required for audio recording. Install with: pip install soundfile sounddevice"
    
    try:
        freq = 44100
        recording = sd.rec(int(duration * freq), samplerate=freq, channels=1)
        sd.wait()
        
        status_placeholder.success("✅ **Recording finished!** Processing transcription...")
        time.sleep(0.5)
            
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            sf.write(tmp_file.name, recording, freq)
            tmp_path = tmp_file.name
        
        # Load model and transcribe
        model = load_whisper_model()
        if model is None:
            return "Error: Could not load Whisper model"
        
        result = model.transcribe(tmp_path)
        
        # Clean up
        os.unlink(tmp_path)
        
        return result["text"]
    except Exception as e:
        return f"Error recording audio: {str(e)}"

def render_message(text: str):
    url_pattern = r"(https?://\S+)"
    
    def replace_url(match):
        url = match.group(0)
        return f"<a href='{html.escape(url)}' target='_blank'>{html.escape(url)}</a>"
    
    escaped = html.escape(text)
    linked = re.sub(url_pattern, replace_url, escaped)

    st.markdown(linked, unsafe_allow_html=True)

# Main UI
st.title("🤖 AI Agent Assistant")
st.markdown("Ask questions and get intelligent responses powered by LangGraph agents")

# Helper function to get available Ollama models
@st.cache_data(ttl=60)  # Cache for 60 seconds
def get_ollama_models():
    """Get list of available Ollama models"""
    try:
        import subprocess
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            models = []
            for line in lines[1:]:  # Skip header
                parts = line.split()
                if len(parts) >= 2:
                    model_name = parts[0].split(':')[0]
                    models.append(model_name)
            return models
    except:
        pass
    return []

# Settings (moved from sidebar)
use_tts = st.checkbox("Enable Text-to-Speech", value=False)

# Input method selection
input_options = ["Text Input"]
if WHISPER_AVAILABLE:
    input_options.append("Audio Upload")
    if AUDIO_LIBS_AVAILABLE:
        input_options.append("Record Audio")

input_method = st.radio(
    "Choose input method:",
    input_options,
    index=0,
    horizontal=True
)

if WHISPER_AVAILABLE and not AUDIO_LIBS_AVAILABLE:
    st.info("💡 Install soundfile and sounddevice for recording: `pip install soundfile sounddevice`")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("💬 Chat")
    
    # Input section based on selected method
    if input_method == "Text Input":
        user_input = st.text_area(
            "Enter your question:",
            height=100,
            placeholder="e.g., I need a puzzle under $15"
        )
        submit_button = st.button("Submit", type="primary", use_container_width=True)
        
    elif input_method == "Audio Upload":
        uploaded_file = st.file_uploader(
            "Upload audio file (WAV format)",
            type=['wav', 'mp3', 'm4a']
        )
        submit_button = st.button("Process Audio", type="primary", use_container_width=True)
        
        if uploaded_file is not None:
            user_input = None
            if submit_button:
                with st.spinner("Transcribing audio..."):
                    transcription = transcribe_audio(uploaded_file)
                    if not transcription.startswith("Error"):
                        user_input = transcription
                        st.success(f"Transcribed: {transcription}")
                    else:
                        st.error(transcription)
        else:
            user_input = None
            submit_button = False
            
    else:  # Record Audio
        duration = st.slider("Recording duration (seconds)", 1, 10, 5)
        record_button = st.button("🎤 Start Recording", type="primary", use_container_width=True)
        
        # Status placeholder for recording feedback
        status_placeholder = st.empty()
        
        if record_button:
            # Show countdown before recording
            status_placeholder.info("⏳ **Preparing to record...**")
            time.sleep(0.5)
            
            for i in range(3, 0, -1):
                status_placeholder.info(f"⏳ **Starting in {i}...**")
                time.sleep(1)
            
            # Show recording started
            status_placeholder.warning(f"🔴 **RECORDING NOW!** Speak clearly... (Recording for {duration} seconds)")
            
            # Record audio
            transcription = record_audio(duration)
            
            if not transcription.startswith("Error"):
                user_input = transcription
                status_placeholder.success(f"✅ **Transcription complete!**\n\n**Transcribed text:** {transcription}")
            else:
                status_placeholder.error(f"❌ **Error:** {transcription}")
                user_input = None
        else:
            user_input = None
            submit_button = False
            status_placeholder.empty()
    
    # Process query
    if (input_method == "Text Input" and submit_button and user_input) or \
       (input_method in ["Audio Upload", "Record Audio"] and user_input):
        
        if not st.session_state.processing:
            st.session_state.processing = True
            
            # Add user message to history
            st.session_state.conversation_history.append({
                "role": "user",
                "content": user_input
            })
            
            # Process query
            with st.spinner("Processing your query through the agent graph..."):
                response = process_query(user_input, use_tts=use_tts)
            
            # Add assistant response to history
            st.session_state.conversation_history.append({
                "role": "assistant",
                "content": response
            })
            
            st.session_state.processing = False
    
    # Display conversation history
    st.markdown("---")
    st.subheader("📜 Conversation History")
    
    if st.session_state.conversation_history:
        for i, message in enumerate(st.session_state.conversation_history):
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.write(message["content"])
            else:
                with st.chat_message("assistant"):
                    st.write(message["content"])
        
        # Clear history button
        if st.button("Clear History", use_container_width=True):
            st.session_state.conversation_history = []
            st.rerun()
    else:
        st.info("No conversation history yet. Start by asking a question!")

with col2:
    st.header("ℹ️ About")
    st.markdown("""
    This application uses a multi-agent system powered by LangGraph:
    
    **Agent Pipeline:**
    1. **Router Agent** - Identifies task and constraints
    2. **Planner Agent** - Creates retrieval plan
    3. **Retriever Agent** - Fetches relevant data
    4. **Answer/Critic Agent** - Synthesizes final response
    
    **Features:**
    - 📝 Text input
    - 🎤 Voice input (Whisper ASR)
    - 🔊 Text-to-speech output
    - 📊 Multi-agent reasoning
    """)
    
    st.markdown("---")
    st.header("🔧 System Status")
    
    # Check if required modules are available
    try:
        from graph.graph import app
        st.success("✅ Graph agent loaded")
    except Exception as e:
        st.error(f"❌ Graph agent error: {str(e)}")
    
    if WHISPER_AVAILABLE:
        try:
            model = load_whisper_model()
            if model:
                st.success("✅ Whisper ASR available")
            else:
                st.warning("⚠️ Whisper model loading failed")
        except Exception as e:
            st.error(f"❌ Whisper error: {str(e)}")
    else:
        st.error("❌ Whisper not installed")
    
    try:
        from openai import AsyncOpenAI
        st.success("✅ OpenAI TTS available")
    except Exception as e:
        st.error(f"❌ OpenAI TTS error: {str(e)}")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>Built with Streamlit, LangGraph, and OpenAI</div>",
    unsafe_allow_html=True
)

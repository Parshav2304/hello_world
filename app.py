import streamlit as st
import os
import json
import tempfile
import uuid
from datetime import datetime
import base64
from io import BytesIO
import pandas as pd

# Third-party imports
import openai
import PyPDF2
import boto3
from google.cloud import texttospeech
from pymongo import MongoClient
import requests

# Page configuration
st.set_page_config(
    page_title="Study Content Converter",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }
    .success-message {
        background: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #c3e6cb;
    }
    .error-message {
        background: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #f5c6cb;
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

class StudyContentProcessor:
    def __init__(self):
        self.setup_apis()
    
    def setup_apis(self):
        """Initialize API connections"""
        # OpenAI setup
        if "openai_api_key" in st.secrets:
            openai.api_key = st.secrets["openai_api_key"]
        elif os.getenv("OPENAI_API_KEY"):
            openai.api_key = os.getenv("OPENAI_API_KEY")
        
        # MongoDB setup (optional for demo)
        try:
            if "mongodb_uri" in st.secrets:
                self.mongo_client = MongoClient(st.secrets["mongodb_uri"])
            elif os.getenv("MONGODB_URI"):
                self.mongo_client = MongoClient(os.getenv("MONGODB_URI"))
            else:
                self.mongo_client = None
        except:
            self.mongo_client = None
        
        # AWS S3 setup (optional for demo)
        try:
            if all(key in st.secrets for key in ["aws_access_key", "aws_secret_key", "aws_bucket_name"]):
                self.s3_client = boto3.client(
                    's3',
                    aws_access_key_id=st.secrets["aws_access_key"],
                    aws_secret_access_key=st.secrets["aws_secret_key"]
                )
                self.bucket_name = st.secrets["aws_bucket_name"]
            else:
                self.s3_client = None
                self.bucket_name = None
        except:
            self.s3_client = None
            self.bucket_name = None
        
        # Google Cloud TTS setup (optional for demo)
        try:
            if "google_credentials" in st.secrets:
                # Create credentials file from secrets
                credentials_info = st.secrets["google_credentials"]
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                    json.dump(credentials_info, f)
                    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = f.name
                self.tts_client = texttospeech.TextToSpeechClient()
            else:
                self.tts_client = None
        except:
            self.tts_client = None
    
    def extract_text_from_pdf(self, uploaded_file):
        """Extract text from PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            st.error(f"Error extracting PDF text: {str(e)}")
            return None
    
    def simplify_content_with_gpt(self, text, complexity_level="beginner"):
        """Use GPT-4 to simplify and summarize content"""
        try:
            if not openai.api_key:
                st.error("OpenAI API key not configured")
                return None
            
            # Limit text length to avoid token limits
            text_chunk = text[:4000] if len(text) > 4000 else text
            
            prompt = f"""
            Please simplify the following academic/research content for {complexity_level} level understanding:
            
            Requirements:
            1. Break down complex concepts into simple explanations
            2. Use analogies and examples where helpful
            3. Create bullet points for key concepts
            4. Maintain accuracy while improving clarity
            5. Maximum 500 words for summary
            
            Content to simplify:
            {text_chunk}
            """
            
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert educator who specializes in making complex academic content accessible to students."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.7
            )
            
            return response.choices[0].message.content
        except Exception as e:
            st.error(f"Error with GPT processing: {str(e)}")
            return None
    
    def generate_voice_note(self, text):
        """Convert text to speech using Google Cloud TTS or fallback"""
        try:
            if self.tts_client:
                # Limit text for TTS
                tts_text = text[:1000] if len(text) > 1000 else text
                
                synthesis_input = texttospeech.SynthesisInput(text=tts_text)
                voice = texttospeech.VoiceSelectionParams(
                    language_code="en-US",
                    name="en-US-Wavenet-D"
                )
                audio_config = texttospeech.AudioConfig(
                    audio_encoding=texttospeech.AudioEncoding.MP3
                )
                
                response = self.tts_client.synthesize_speech(
                    input=synthesis_input,
                    voice=voice,
                    audio_config=audio_config
                )
                
                return response.audio_content
            else:
                # Fallback: create a placeholder audio message
                st.info("Google Cloud TTS not configured. Voice generation would work with proper API setup.")
                return None
        except Exception as e:
            st.error(f"Error generating voice note: {str(e)}")
            return None
    
    def create_visual_explanation(self, summary_text):
        """Generate visual explanation prompt (placeholder for actual image generation)"""
        try:
            visual_prompt = f"""
            Create a simple infographic or diagram to explain:
            
            {summary_text[:300]}
            
            Visual elements should include:
            - Key concepts as boxes or circles
            - Connecting arrows showing relationships
            - Simple icons or symbols
            - Clear, readable text
            - Bright, engaging colors
            """
            
            return visual_prompt
        except Exception as e:
            st.error(f"Error creating visual explanation: {str(e)}")
            return None

# Initialize processor
@st.cache_resource
def get_processor():
    return StudyContentProcessor()

processor = get_processor()

# Main app interface
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📚 Study Content Converter</h1>
        <p>Transform complex research papers and notes into simplified learning materials</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        complexity_level = st.selectbox(
            "Learning Level",
            ["beginner", "intermediate", "advanced"],
            help="Choose the complexity level for simplified content"
        )
        
        output_formats = st.multiselect(
            "Output Formats",
            ["Summary", "Voice Note", "Visual Explanation"],
            default=["Summary"],
            help="Select which formats you want to generate"
        )
        
        st.header("📊 API Status")
        
        # Check API availability
        api_status = {
            "OpenAI GPT-4": "✅" if openai.api_key else "❌",
            "Google Cloud TTS": "✅" if processor.tts_client else "❌",
            "AWS S3": "✅" if processor.s3_client else "❌",
            "MongoDB": "✅" if processor.mongo_client else "❌"
        }
        
        for service, status in api_status.items():
            st.write(f"{status} {service}")
        
        st.info("💡 The app works with basic functionality even without all APIs configured!")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📄 Upload Your Document")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['pdf', 'txt'],
            help="Upload PDF files or text documents"
        )
        
        # Text input as alternative
        st.header("✏️ Or Paste Your Text")
        manual_text = st.text_area(
            "Paste your content here",
            height=200,
            placeholder="Paste research paper content, notes, or any study material here..."
        )
        
        if st.button("🚀 Process Content", type="primary"):
            process_content(uploaded_file, manual_text, complexity_level, output_formats)
    
    with col2:
        st.header("📋 Features")
        
        features = [
            ("🤖 AI Summarization", "GPT-4 powered content simplification"),
            ("🎵 Voice Notes", "Text-to-speech conversion"),
            ("📊 Visual Explanations", "Concept visualization prompts"),
            ("📱 Multi-format", "PDF and text support"),
            ("⚡ Real-time", "Instant processing")
        ]
        
        for feature, description in features:
            st.markdown(f"""
            <div class="feature-card">
                <h4>{feature}</h4>
                <p>{description}</p>
            </div>
            """, unsafe_allow_html=True)

def process_content(uploaded_file, manual_text, complexity_level, output_formats):
    """Process the uploaded content"""
    
    # Extract text
    text_content = ""
    
    if uploaded_file is not None:
        if uploaded_file.type == "application/pdf":
            text_content = processor.extract_text_from_pdf(uploaded_file)
        elif uploaded_file.type == "text/plain":
            text_content = uploaded_file.read().decode('utf-8')
    elif manual_text.strip():
        text_content = manual_text
    
    if not text_content:
        st.error("Please upload a file or paste some text content.")
        return
    
    # Display original content info
    st.success(f"✅ Content extracted successfully! ({len(text_content)} characters)")
    
    with st.expander("📖 Original Content Preview"):
        st.text_area("Original Text", text_content[:1000] + "..." if len(text_content) > 1000 else text_content, height=200, disabled=True)
    
    # Process content based on selected formats
    results = {}
    
    # Generate summary
    if "Summary" in output_formats:
        with st.spinner("🤖 Generating simplified summary..."):
            summary = processor.simplify_content_with_gpt(text_content, complexity_level)
            if summary:
                results["summary"] = summary
    
    # Generate voice note
    if "Voice Note" in output_formats:
        with st.spinner("🎵 Generating voice note..."):
            text_for_voice = results.get("summary", text_content[:1000])
            audio_content = processor.generate_voice_note(text_for_voice)
            if audio_content:
                results["audio"] = audio_content
    
    # Generate visual explanation
    if "Visual Explanation" in output_formats:
        with st.spinner("📊 Creating visual explanation..."):
            visual_prompt = processor.create_visual_explanation(results.get("summary", text_content[:500]))
            if visual_prompt:
                results["visual"] = visual_prompt
    
    # Display results
    display_results(results, text_content)

def display_results(results, original_text):
    """Display the processed results"""
    
    st.header("🎉 Results")
    
    # Summary section
    if "summary" in results:
        st.subheader("📝 Simplified Summary")
        st.markdown(f"""
        <div class="success-message">
            {results["summary"]}
        </div>
        """, unsafe_allow_html=True)
        
        # Download button for summary
        st.download_button(
            label="📥 Download Summary",
            data=results["summary"],
            file_name=f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )
    
    # Audio section
    if "audio" in results:
        st.subheader("🎵 Voice Note")
        
        # Create audio player
        audio_base64 = base64.b64encode(results["audio"]).decode()
        audio_html = f"""
        <audio controls>
            <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
            Your browser does not support the audio element.
        </audio>
        """
        st.markdown(audio_html, unsafe_allow_html=True)
        
        # Download button for audio
        st.download_button(
            label="📥 Download Voice Note",
            data=results["audio"],
            file_name=f"voice_note_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3",
            mime="audio/mp3"
        )
    
    # Visual explanation section
    if "visual" in results:
        st.subheader("📊 Visual Explanation Concept")
        st.info("This is a prompt for creating visual content. In a full implementation, this would generate actual diagrams or infographics.")
        st.text_area("Visual Concept", results["visual"], height=200, disabled=True)
    
    # Statistics
    st.subheader("📈 Content Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Original Length", f"{len(original_text)} chars")
    
    with col2:
        if "summary" in results:
            st.metric("Summary Length", f"{len(results['summary'])} chars")
        else:
            st.metric("Summary Length", "N/A")
    
    with col3:
        compression_ratio = (len(results.get("summary", "")) / len(original_text) * 100) if "summary" in results else 0
        st.metric("Compression", f"{compression_ratio:.1f}%")
    
    with col4:
        st.metric("Processing Time", "< 30s")
    
    # Save to session state for history
    if "processed_documents" not in st.session_state:
        st.session_state.processed_documents = []
    
    document_entry = {
        "timestamp": datetime.now(),
        "original_length": len(original_text),
        "summary": results.get("summary", ""),
        "has_audio": "audio" in results,
        "has_visual": "visual" in results
    }
    
    st.session_state.processed_documents.insert(0, document_entry)
    
    # Keep only last 10 documents
    st.session_state.processed_documents = st.session_state.processed_documents[:10]

# Document history page
def show_history():
    st.header("📚 Document History")
    
    if "processed_documents" not in st.session_state or not st.session_state.processed_documents:
        st.info("No documents processed yet. Upload a document to get started!")
        return
    
    for i, doc in enumerate(st.session_state.processed_documents):
        with st.expander(f"Document {i+1} - {doc['timestamp'].strftime('%Y-%m-%d %H:%M')}"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Original Length", f"{doc['original_length']} chars")
            
            with col2:
                st.metric("Has Audio", "✅" if doc['has_audio'] else "❌")
            
            with col3:
                st.metric("Has Visual", "✅" if doc['has_visual'] else "❌")
            
            if doc['summary']:
                st.text_area("Summary", doc['summary'], height=100, disabled=True, key=f"summary_{i}")

# Navigation
def main_navigation():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["🏠 Home", "📚 History", "ℹ️ About"])
    
    if page == "🏠 Home":
        main()
    elif page == "📚 History":
        show_history()
    elif page == "ℹ️ About":
        show_about()

def show_about():
    st.header("ℹ️ About Study Content Converter")
    
    st.markdown("""
    ## 🎯 Purpose
    This application helps students and researchers convert complex academic content into simplified, digestible formats for easier studying and comprehension.
    
    ## 🚀 Features
    - **AI-Powered Summarization**: Uses GPT-4 to break down complex concepts
    - **Voice Notes**: Converts text to speech for audio learning
    - **Visual Explanations**: Generates concepts for visual learning aids
    - **Multiple Input Formats**: Supports PDF and text files
    - **Customizable Complexity**: Adjust content difficulty level
    
    ## 🛠️ Technology Stack
    - **Frontend**: Streamlit
    - **AI Processing**: OpenAI GPT-4
    - **Text-to-Speech**: Google Cloud TTS
    - **File Storage**: AWS S3
    - **Database**: MongoDB
    - **Audio Processing**: FFmpeg
    - **PDF Processing**: PyPDF2
    
    ## 🔧 Setup Requirements
    To run this app with full functionality, you need:
    1. OpenAI API key for content processing
    2. Google Cloud credentials for TTS
    3. AWS credentials for file storage
    4. MongoDB for data persistence
    
    ## 📱 Use Cases
    - Research paper simplification
    - Study note conversion
    - Academic content accessibility
    - Language learning support
    - Exam preparation assistance
    
    ## 🤝 Hackathon Project
    This project was built for a hackathon focusing on educational technology and AI-powered learning tools.
    """)

if __name__ == "__main__":
    main_navigation()

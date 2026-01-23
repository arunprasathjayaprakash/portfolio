"""
Streamlit application for Snowflake Cortex AI RAG Knowledge Base
"""
import streamlit as st
import snowflake_utils
import pdf_processor
import os
from dotenv import load_dotenv

# Load environment variables from .env file (for local development)
load_dotenv()

# Page configuration (must be called before any other Streamlit commands)
st.set_page_config(
    page_title="Snowflake RAG Knowledge Base",
    page_icon="❄️",
    layout="wide"
)

def get_config_value(key: str, default: str = "") -> str:
    """
    Get configuration value from Streamlit secrets (for cloud) or environment variables (for local).
    Priority: Streamlit secrets > Environment variables > default
    """
    # Try Streamlit secrets first (for Streamlit Cloud)
    try:
        if hasattr(st, "secrets") and hasattr(st.secrets, "snowflake"):
            snowflake_secrets = st.secrets.snowflake
            if isinstance(snowflake_secrets, dict) and key.lower() in snowflake_secrets:
                return str(snowflake_secrets[key.lower()])
    except (AttributeError, KeyError, TypeError):
        pass
    
    # Fall back to environment variables (for local development)
    env_key = f"SNOWFLAKE_{key}"
    return os.getenv(env_key, default)

# Validate required configuration values
required_keys = ["ACCOUNT", "USER", "PASSWORD"]
missing_keys = []

for key in required_keys:
    value = get_config_value(key)
    if not value:
        missing_keys.append(f"SNOWFLAKE_{key}")

if missing_keys:
    st.error(f"❌ Missing required configuration: {', '.join(missing_keys)}")
    st.info("""
    **For Local Development:**
    Please create a `.env` file based on `.env.example` and fill in your Snowflake credentials.
    
    **For Streamlit Cloud:**
    Please configure secrets in Streamlit Cloud dashboard:
    1. Go to your app settings
    2. Click on "Secrets"
    3. Add the required Snowflake credentials
    """)
    st.code("""
[snowflake]
account = "your-account-identifier"
user = "your-username"
password = "your-password"
role = "ACCOUNTADMIN"
warehouse = "COMPUTE_WH"
database = "DOCUMENT_CHUNKS_DB"
schema = "public"
    """, language="toml")
    st.stop()

# Snowflake connection parameters from secrets or environment variables
CONNECTION_PARAMS = {
    "account": get_config_value("ACCOUNT", ""),
    "user": get_config_value("USER", ""),
    "password": get_config_value("PASSWORD", ""),
    "role": get_config_value("ROLE", "ACCOUNTADMIN"),
    "warehouse": get_config_value("WAREHOUSE", "COMPUTE_WH"),
    "database": get_config_value("DATABASE", "DOCUMENT_CHUNKS_DB"),
    "schema": get_config_value("SCHEMA", "public")
}

# Get embedding model configuration (optional, defaults to None which uses Snowflake default)
EMBEDDING_MODEL = get_config_value("EMBEDDING_MODEL", "").strip() or None

# Initialize session state
if "snowflake_manager" not in st.session_state:
    st.session_state.snowflake_manager = None
    st.session_state.connected = False
    st.session_state.team_id = None

def initialize_snowflake():
    """Initialize Snowflake connection"""
    if st.session_state.snowflake_manager is None:
        manager = snowflake_utils.SnowflakeManager(CONNECTION_PARAMS, embedding_model=EMBEDDING_MODEL)
        if manager.connect():
            st.session_state.snowflake_manager = manager
            st.session_state.connected = True
            # Setup database tables
            manager.setup_database()
            return True
        else:
            st.error("Failed to connect to Snowflake")
            return False
    return st.session_state.connected

def main():
    st.title("❄️ Snowflake Cortex AI RAG Knowledge Base")
    st.markdown("Upload PDF documents and query them using Snowflake Cortex AI")
    
    # Initialize connection
    if not initialize_snowflake():
        st.stop()
    
    # Sidebar for team selection
    with st.sidebar:
        st.header("🔐 Team Access")
        team_id = st.text_input(
            "Enter Team ID",
            value=st.session_state.team_id or "",
            help="Enter your team ID to access your team's documents"
        )
        
        if team_id:
            st.session_state.team_id = team_id
            st.success(f"Team: {team_id}")
        else:
            st.warning("Please enter a Team ID to continue")
            st.stop()
        
        st.divider()
        
        # Show team documents
        if st.session_state.team_id:
            st.subheader("Your Documents")
            manager = st.session_state.snowflake_manager
            documents = manager.get_team_documents(st.session_state.team_id)
            
            if documents:
                for doc in documents:
                    with st.expander(f"📄 {doc['filename']}"):
                        st.write(f"**Chunks:** {doc['num_chunks']}")
                        st.write(f"**Size:** {doc['file_size']} chars")
                        st.write(f"**Uploaded:** {doc['uploaded_at']}")
            else:
                st.info("No documents uploaded yet")
    
    # Main content area
    tab1, tab2 = st.tabs(["📤 Upload Documents", "🔍 Query Knowledge Base"])
    
    with tab1:
        st.header("Upload PDF Documents")
        st.markdown("Upload PDF files to add them to your team's knowledge base")
        
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type=['pdf'],
            accept_multiple_files=True
        )
        
        if uploaded_files and st.session_state.team_id:
            if st.button("Process and Upload", type="primary"):
                manager = st.session_state.snowflake_manager
                processor = pdf_processor.PDFProcessor(chunk_size=1000, chunk_overlap=200)
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for idx, uploaded_file in enumerate(uploaded_files):
                    try:
                        status_text.text(f"Processing {uploaded_file.name}...")
                        
                        # Process PDF
                        result = processor.process_pdf(uploaded_file, uploaded_file.name)
                        
                        # Insert document metadata
                        manager.insert_document_metadata(
                            document_id=result["document_id"],
                            team_id=st.session_state.team_id,
                            filename=result["filename"],
                            file_size=result["file_size"],
                            num_chunks=result["num_chunks"]
                        )
                        
                        # Insert chunks with embeddings
                        manager.insert_chunks(
                            chunks=result["chunks"],
                            team_id=st.session_state.team_id,
                            document_id=result["document_id"]
                        )
                        
                        st.success(f"✅ Successfully processed {uploaded_file.name} ({result['num_chunks']} chunks)")
                        
                    except Exception as e:
                        error_msg = str(e)
                        st.error(f"❌ Error processing {uploaded_file.name}")
                        
                        # Check if it's a region availability error
                        if "unavailable in your region" in error_msg or "cross region inference" in error_msg:
                            st.error("**Region Availability Issue Detected**")
                            st.markdown("""
                            **The embedding model is not available in your Snowflake region.**
                            
                            **To fix this:**
                            1. Enable Cross-Region Inference in your Snowflake account:
                               - Go to Admin → Accounts → [Your Account]
                               - Enable cross-region inference with: AWS_APJ, AWS_EU, AWS_US, AZURE_EU, AZURE_US, or ANY_REGION
                               - See: https://docs.snowflake.com/en/user-guide/snowflake-cortex/cross-region-inference
                            
                            2. Or contact your Snowflake administrator to enable this feature
                            
                            3. After enabling, try uploading again
                            """)
                        else:
                            st.error(f"Error details: {error_msg}")
                    
                    progress_bar.progress((idx + 1) / len(uploaded_files))
                
                status_text.text("Upload complete!")
                st.balloons()
    
    with tab2:
        st.header("Query Your Knowledge Base")
        st.markdown("Ask questions about your uploaded documents using Snowflake Cortex AI")
        
        if not st.session_state.team_id:
            st.warning("Please enter a Team ID in the sidebar to query documents")
            st.stop()
        
        # Query input
        query = st.text_area(
            "Enter your question",
            height=100,
            placeholder="e.g., What are the main topics discussed in the documents?"
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            top_k = st.number_input("Top K Results", min_value=1, max_value=20, value=5)
        
        with col2:
            search_button = st.button("🔍 Search", type="primary", use_container_width=True)
        
        if search_button and query:
            manager = st.session_state.snowflake_manager
            
            with st.spinner("Searching knowledge base..."):
                # Get answer using Cortex AI
                result = manager.query_with_cortex_ai(
                    query=query,
                    team_id=st.session_state.team_id,
                    top_k=top_k
                )
                
                if isinstance(result, dict):
                    # Display answer
                    st.subheader("💡 Answer")
                    st.write(result.get("answer", "No answer generated"))
                    
                    # Display sources
                    if result.get("sources"):
                        st.subheader("📚 Sources")
                        for i, source in enumerate(result["sources"], 1):
                            with st.expander(f"Source {i} (Chunk ID: {source['chunk_id']})"):
                                st.write(source.get("text", ""))
                else:
                    st.write(result)
        
        # Show search history or examples
        with st.expander("💡 Example Queries"):
            st.markdown("""
            - What are the main topics discussed in the documents?
            - Summarize the key points from the uploaded PDFs
            - What information is available about [specific topic]?
            - Can you explain [concept] based on the documents?
            """)


if __name__ == "__main__":
    main()

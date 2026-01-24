# Snowflake Cortex AI RAG Knowledge Base

A Streamlit application that uses Snowflake Cortex AI platform as a RAG (Retrieval-Augmented Generation) knowledge base. Upload PDF documents, chunk them, store embeddings in Snowflake, and query them using AI-powered search with team-based access control.

## Features

- 📤 **PDF Upload**: Upload multiple PDF files and automatically chunk them
- 🔐 **Team-Based Access Control**: Documents are isolated by team ID - only users from the same team can access their team's documents
- ❄️ **Snowflake Integration**: Uses Snowflake's default embedding model (`e5-base-v2`) for vector embeddings
- 🤖 **Cortex AI RAG**: Query documents using Snowflake Cortex AI for intelligent answers
- 🔍 **Vector Search**: Efficient similarity search using Snowflake's vector operations

## Prerequisites

- Python 3.8 or higher
- Snowflake account with Cortex AI enabled
- Access to the specified Snowflake database and warehouse

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd snowflake_app
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

Or install using pip with the project:
```bash
pip install -e .
```

## Configuration

### Local Development

Create a `.streamlit/secrets.toml` file in your project root with your Snowflake credentials:

```toml
[snowflake]
account = "your-account-identifier"
user = "your-username"
password = "your-password"
role = "ACCOUNTADMIN"
warehouse = "COMPUTE_WH"
database = "DOCUMENT_CHUNKS_DB"
schema = "public"
```

**Important**: The `.streamlit/secrets.toml` file is automatically added to `.gitignore` and should never be committed to version control.

### Streamlit Cloud Deployment

The application automatically uses Streamlit Cloud secrets when deployed. To configure:

1. Push your code to GitHub

2. Go to [Streamlit Cloud](https://streamlit.io/cloud) and sign in

3. Click "New app" and connect your repository

4. In the app settings, go to "Secrets" and add your Snowflake credentials in TOML format:
```toml
[snowflake]
account = "your-account-identifier"
user = "your-username"
password = "your-password"
role = "ACCOUNTADMIN"
warehouse = "COMPUTE_WH"
database = "DOCUMENT_CHUNKS_DB"
schema = "public"
```

5. Set the main file path to `app.py`

6. Click "Deploy"

The application will automatically use Streamlit secrets for both local development and cloud deployment.

## Usage

1. Ensure your `.streamlit/secrets.toml` file is configured with valid Snowflake credentials

2. Run the Streamlit application:
```bash
streamlit run app.py
```

3. Open your browser and navigate to the URL shown in the terminal (typically `http://localhost:8501`)

4. **Enter Team ID**: In the sidebar, enter your team ID to access your team's documents

5. **Upload Documents**: 
   - Go to the "Upload Documents" tab
   - Select PDF files to upload
   - Click "Process and Upload"
   - The app will chunk the PDFs and store them with embeddings in Snowflake

6. **Query Knowledge Base**:
   - Go to the "Query Knowledge Base" tab
   - Enter your question
   - Click "Search"
   - View the AI-generated answer and source chunks

## Project Structure

- **`app.py`**: Main Streamlit application with UI
- **`snowflake_utils.py`**: Snowflake connection and database operations
- **`pdf_processor.py`**: PDF text extraction and chunking utilities
- **`.streamlit/secrets.toml`**: Streamlit secrets for Snowflake credentials (local development, not in git)
- **`.streamlit/secrets.example.toml`**: Example secrets structure for reference
- **`.streamlit/config.toml`**: Streamlit configuration
- **`requirements.txt`**: Python dependencies
- **`pyproject.toml`**: Project metadata and build configuration

## Database Schema

The application creates two tables:

1. **`document_chunks`**: Stores document chunks with embeddings
   - `chunk_id`: Unique identifier for each chunk
   - `document_id`: Links chunks to their source document
   - `team_id`: Team identifier for access control
   - `chunk_text`: The text content of the chunk
   - `embedding`: Vector embedding (768 dimensions)
   - `metadata`: Additional metadata in JSON format

2. **`documents`**: Stores document metadata
   - `document_id`: Unique document identifier
   - `team_id`: Team identifier
   - `filename`: Original filename
   - `file_size`: Size in characters
   - `num_chunks`: Number of chunks created

## Security

- **Team Isolation**: All queries filter by `team_id` to ensure teams can only access their own documents
- **Access Control**: Implemented at the database query level for security
- **Streamlit Secrets**: Credentials are stored securely in `.streamlit/secrets.toml` for local development and encrypted in Streamlit Cloud secrets (not accessible in code)

## Development

### Installing in Development Mode

```bash
pip install -e ".[dev]"
```

### Running Tests

```bash
pytest
```

## Notes

- The application uses Snowflake's `e5-base-v2` embedding model
- Chunks are created with 1000 characters per chunk and 200 character overlap
- Vector similarity search uses cosine distance
- Cortex AI uses `llama3-8b` model for answer generation

## Troubleshooting

### Connection Issues

If you encounter connection errors:
1. Verify your Snowflake credentials in `.streamlit/secrets.toml`
2. Ensure your Snowflake account has Cortex AI enabled
3. Check that the warehouse, database, and schema exist
4. Verify your user has the necessary permissions

### Embedding Generation Errors

If embeddings fail to generate:
1. Ensure Cortex AI is enabled in your Snowflake account
2. Verify you have access to the `SNOWFLAKE.CORTEX.EMBED_TEXT_768` function
3. Check that your warehouse is running

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]

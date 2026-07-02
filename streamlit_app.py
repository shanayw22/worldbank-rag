"""
World Bank Intelligence Assistant - RAG Application
Deployed on Streamlit Community Cloud
"""

import streamlit as st
import faiss
import pickle
import os
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# Page configuration
st.set_page_config(
    page_title="World Bank Intelligence Assistant",
    page_icon="🌍",
    layout="wide"
)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Title and description
st.title("🌍 World Bank Intelligence Assistant")
st.markdown("""
Ask questions about World Bank projects, policies, and data. This RAG system uses:
* 📚 **29,212 World Bank documents** (Bronze, Silver, Gold layers)
* 🔍 **FAISS vector search** for semantic retrieval
* 🤖 **GPT-4o-mini** for intelligent responses
""")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Model selection
    model_option = st.selectbox(
        "AI Model",
        ["gpt-4o-mini", "gpt-3.5-turbo", "gpt-4o"],
        index=0,
        help="gpt-4o-mini: Best balance of quality & cost (recommended)"
    )
    
    # Temperature slider
    temperature = st.slider(
        "Response Creativity",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.1,
        help="Lower = more focused, Higher = more creative"
    )
    
    # Top-k results
    top_k = st.slider(
        "Number of documents to retrieve",
        min_value=1,
        max_value=10,
        value=5,
        help="How many relevant documents to use"
    )
    
    st.markdown("---")
    st.markdown("### 📊 System Info")
    st.markdown(f"""
    * **Documents:** 29,212
    * **Vector Store:** FAISS
    * **Embeddings:** BAAI/bge-small-en-v1.5
    * **LLM:** {model_option}
    """)
    
    # Model cost info
    if model_option == "gpt-4o-mini":
        st.info("💰 Cost: ~$0.15 per 1M tokens (cheapest)")
    elif model_option == "gpt-3.5-turbo":
        st.info("💰 Cost: ~$0.50 per 1M tokens")
    else:
        st.info("💰 Cost: ~$2.50 per 1M tokens (best quality)")
    
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Load models and data
@st.cache_resource
def load_resources():
    """Load FAISS index, metadata, and embedding model"""
    with st.spinner("Loading AI models and vector database..."):
        # Load FAISS index
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        index = faiss.read_index(os.path.join(data_dir, "faiss.index"))
        
        # Load metadata (now a simple list of dictionaries)
        with open(os.path.join(data_dir, "metadata.pkl"), "rb") as f:
            metadata = pickle.load(f)
        
        # Load embedding model
        embedding_model = SentenceTransformer('BAAI/bge-small-en-v1.5')
        
        return index, metadata, embedding_model

try:
    # Check for OpenAI API key
    if "OPENAI_API_KEY" not in st.secrets:
        st.error("🔐 OpenAI API key not configured. Please add it to Streamlit secrets.")
        st.info("""
        **How to add your API key:**
        1. Click ⚙️ Settings (top right)
        2. Go to "Secrets" tab
        3. Add: `OPENAI_API_KEY = "sk-proj-your-key"`
        4. Get a key at: https://platform.openai.com/api-keys
        """)
        st.stop()
    
    # Initialize OpenAI client
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    
    # Load resources
    index, metadata, embedding_model = load_resources()
    
    st.success("✅ System ready! Ask me anything about the World Bank.")
    
except Exception as e:
    st.error(f"❌ Error loading resources: {str(e)}")
    import traceback
    st.code(traceback.format_exc())
    st.stop()

def retrieve_documents(query: str, k: int = 5):
    """Retrieve top-k relevant documents"""
    # Encode query
    query_embedding = embedding_model.encode([query])[0]
    
    # Search FAISS index
    distances, indices = index.search(query_embedding.reshape(1, -1), k)
    
    # Get documents
    documents = []
    for idx, distance in zip(indices[0], distances[0]):
        if idx < len(metadata):
            doc = metadata[idx].copy()
            doc['similarity_score'] = float(1 / (1 + distance))  # Convert distance to similarity
            documents.append(doc)
    
    return documents

def generate_response(query: str, context_docs: list, model: str, temp: float = 0.3):
    """Generate response using OpenAI"""
    # Build context from retrieved documents
    context = "\n\n".join([
        f"Document {i+1}:\n{doc.get('text', doc.get('content', 'N/A'))}"
        for i, doc in enumerate(context_docs)
    ])
    
    # Create prompt
    prompt = f"""You are an expert assistant specializing in World Bank data and projects. 
Answer the following question based ONLY on the provided context. Be specific and cite 
relevant information from the documents.

Context:
{context}

Question: {query}

Answer:"""
    
    # Call OpenAI API
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a World Bank expert assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=temp,
        max_tokens=1000
    )
    
    return response.choices[0].message.content

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("📚 View Sources"):
                for i, doc in enumerate(message["sources"]):
                    st.markdown(f"**Source {i+1}** (Similarity: {doc['similarity_score']:.2%})")
                    st.markdown(doc.get('text', doc.get('content', 'N/A'))[:500] + "...")
                    st.markdown("---")

# Chat input
if query := st.chat_input("Ask about World Bank projects, policies, or data..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching documents and generating response..."):
            try:
                # Retrieve documents
                context_docs = retrieve_documents(query, k=top_k)
                
                # Generate response
                response = generate_response(query, context_docs, model_option, temp=temperature)
                
                # Display response
                st.markdown(response)
                
                # Add to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "sources": context_docs
                })
                
                # Show sources
                with st.expander("📚 View Sources"):
                    for i, doc in enumerate(context_docs):
                        st.markdown(f"**Source {i+1}** (Similarity: {doc['similarity_score']:.2%})")
                        st.markdown(doc.get('text', doc.get('content', 'N/A'))[:500] + "...")
                        st.markdown("---")
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                import traceback
                with st.expander("🔍 Full Error Details"):
                    st.code(traceback.format_exc())

# Example questions
st.markdown("---")
st.markdown("### 💡 Example Questions")
col1, col2 = st.columns(2)

with col1:
    if st.button("💰 What are the largest World Bank projects?"):
        st.session_state.messages.append({"role": "user", "content": "What are the largest World Bank projects?"})
        st.rerun()
    
    if st.button("🌱 How does the World Bank address climate change?"):
        st.session_state.messages.append({"role": "user", "content": "How does the World Bank address climate change?"})
        st.rerun()

with col2:
    if st.button("🏥 What health initiatives does the World Bank support?"):
        st.session_state.messages.append({"role": "user", "content": "What health initiatives does the World Bank support?"})
        st.rerun()
    
    if st.button("📊 How are World Bank projects evaluated?"):
        st.session_state.messages.append({"role": "user", "content": "How are World Bank projects evaluated?"})
        st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with Databricks RAG Pipeline | Powered by OpenAI & FAISS</p>
    <p>29,212 World Bank documents | Medallion Architecture (Bronze/Silver/Gold)</p>
</div>
""", unsafe_allow_html=True)

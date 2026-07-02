# 🌍 World Bank Intelligence Assistant

A production-ready Retrieval-Augmented Generation (RAG) system for querying World Bank documents, projects, and policies.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-red)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4-green)

## 🚀 Live Demo

**[Try it here!](https://your-username-worldbank-rag.streamlit.app)** *(Update after deployment)*

## 📋 Overview

This RAG application enables natural language queries over **29,212 World Bank documents** using:

* **Vector Search:** FAISS (42.79 MB index)
* **Embeddings:** BAAI/bge-small-en-v1.5
* **LLM:** GPT-4 Turbo
* **Architecture:** Medallion (Bronze/Silver/Gold layers)
* **Hosting:** Streamlit Community Cloud

## ✨ Features

* 💬 **Conversational interface** with chat history
* 🔍 **Semantic search** across 29K+ documents
* 📚 **Source citations** with similarity scores
* ⚙️ **Adjustable parameters** (temperature, top-k retrieval)
* 🎨 **Clean, responsive UI**

## 🏗️ Architecture

```
User Query
    ↓
[Embedding Model] → Query Vector
    ↓
[FAISS Index] → Top-K Similar Documents
    ↓
[GPT-4 Turbo] → Generated Response + Sources
    ↓
Streamlit UI
```

### Data Pipeline (Built on Databricks)

```
Bronze Layer (Raw Data)
    ↓
Silver Layer (Cleaned & Deduplicated)
    ↓
Gold Layer (Enriched & Ready for ML)
    ↓
Vector Embeddings → FAISS Index
```

## 🛠️ Tech Stack

* **Frontend:** Streamlit
* **Vector Store:** FAISS
* **Embeddings:** Sentence Transformers
* **LLM:** OpenAI GPT-4 Turbo
* **Data Processing:** Databricks (Spark, Unity Catalog)

## 📦 Installation

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/worldbank-rag.git
   cd worldbank-rag
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up OpenAI API key**
   
   Create `.streamlit/secrets.toml`:
   ```toml
   OPENAI_API_KEY = "sk-your-key-here"
   ```

4. **Run the app**
   ```bash
   streamlit run streamlit_app.py
   ```

5. **Open browser**
   
   Navigate to: `http://localhost:8501`

## 🌐 Deploy to Streamlit Cloud

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Initial commit"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud**
   * Go to [share.streamlit.io](https://share.streamlit.io/)
   * Sign in with GitHub
   * Click "New app"
   * Select your repository
   * Add secret: `OPENAI_API_KEY = "sk-..."`
   * Click "Deploy"!

## 💡 Example Queries

* "What are the largest World Bank projects?"
* "How does the World Bank address climate change?"
* "What health initiatives does the World Bank support?"
* "How are World Bank projects evaluated?"

## 📊 Dataset

* **Source:** World Bank public data
* **Documents:** 29,212
* **Processing:** Medallion Architecture (Bronze → Silver → Gold)
* **Vector Store:** FAISS (L2 distance, 384 dimensions)
* **Index Size:** 42.79 MB
* **Metadata:** 13.94 MB

## 🔧 Configuration

Adjust in the sidebar:

* **Temperature:** 0.0 (focused) to 1.0 (creative)
* **Top-K Retrieval:** 1-10 documents

## 📁 Project Structure

```
worldbank-rag/
├── streamlit_app.py       # Main application
├── requirements.txt       # Python dependencies
├── data/
│   ├── faiss.index        # Vector store (42.79 MB)
│   └── metadata.pkl       # Document metadata (13.94 MB)
├── .streamlit/
│   └── secrets.toml       # API keys (local only)
└── README.md              # This file
```

## 🎯 Key Technical Achievements

✅ Built end-to-end RAG pipeline on Databricks  
✅ Implemented medallion architecture (Bronze/Silver/Gold)  
✅ Created vector embeddings for 29K+ documents  
✅ Integrated FAISS for efficient similarity search  
✅ Deployed production-ready Streamlit application  
✅ Optimized for cost-effective inference (GPT-4 Turbo)

## 💰 Cost Optimization

* **Hosting:** Free (Streamlit Community Cloud)
* **Embeddings:** One-time cost (already computed)
* **OpenAI API:** ~$0.03 per 1K tokens (GPT-4 Turbo)
* **Typical demo usage:** < $1

## 🚧 Future Enhancements

- [ ] Add support for document uploads
- [ ] Implement RAG evaluation metrics
- [ ] Add multi-language support
- [ ] Integrate feedback loop for continuous improvement
- [ ] Expand to other international organizations (IMF, UN, etc.)

## 📝 License

MIT License - feel free to use for your portfolio!

## 👤 Author

**Your Name**
* GitHub: [@your-username](https://github.com/your-username)
* LinkedIn: [Your Profile](https://linkedin.com/in/your-profile)
* Portfolio: [your-website.com](https://your-website.com)

## 🙏 Acknowledgments

* World Bank for public data
* Databricks for cloud infrastructure
* OpenAI for GPT-4 API
* Streamlit for hosting

---

**⭐ Star this repo if you found it helpful!**

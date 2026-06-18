# 🏨 Hotel Guest Support Chatbot

A RAG-based (Retrieval Augmented Generation) conversational chatbot built for luxury hotel environments. Guests and staff can ask natural language questions and get instant, accurate answers grounded in the hotel's own FAQ and policy documents.

Built using **LangChain**, **ChromaDB**, **OpenAI GPT-3.5**, and **Streamlit** — with **LangSmith** tracing for observability.

---

## 🖥️ Demo

> Ask questions like:
> - *"What time is check-in?"*
> - *"How do I cast Netflix to the TV?"*
> - *"Is room service available at night?"*
> - *"Can I bring my pet?"*

The chatbot retrieves the most relevant sections from the hotel FAQ and generates a natural, accurate response.

---

## 🏗️ Architecture

```
User Question
      │
      ▼
OpenAI Embeddings       ←── Hotel FAQ PDF
      │                         │
      ▼                         ▼
ChromaDB Vector Store  ←── Text Chunks (500 chars)
      │
      ▼
Top 4 Relevant Chunks
      │
      ▼
GPT-3.5-turbo  +  Conversation Memory
      │
      ▼
Natural Language Answer
```

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| AI Framework | LangChain (LCEL) |
| LLM | OpenAI GPT-3.5-turbo |
| Embeddings | OpenAI text-embedding-ada-002 |
| Vector Store | ChromaDB |
| UI | Streamlit |
| Observability | LangSmith |
| Language | Python 3.11+ |

---

## 📁 Project Structure

```
hotel-support-chatbot/
├── app.py                  ← Main Streamlit application
├── data/
│   └── hotel_faq.pdf       ← Hotel FAQ knowledge base
├── .env.example            ← Environment variable template
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/mullafarhanusman-arch/hotel-support-chatbot.git
cd hotel-support-chatbot
```

### 2. Create and activate virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
```bash
cp .env.example .env
```
Edit `.env` and add your API keys:
```
OPENAI_API_KEY=your-openai-api-key-here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=your-langsmith-api-key-here
LANGCHAIN_PROJECT=hotel-support-chatbot
```

### 5. Add your hotel FAQ
Replace `data/hotel_faq.pdf` with your own hotel's FAQ or policy document.

### 6. Run the app
```bash
streamlit run app.py
```

Open your browser and go to `http://localhost:8501`

---

## 🔍 How It Works

1. **Document Loading** — PyPDFLoader reads the hotel FAQ PDF
2. **Chunking** — The document is split into 500-character chunks with 50-character overlap to preserve context
3. **Embedding** — Each chunk is converted into a vector using OpenAI embeddings and stored in ChromaDB
4. **Retrieval** — When a user asks a question, the top 4 most semantically similar chunks are retrieved
5. **Generation** — GPT-3.5-turbo uses the retrieved chunks plus conversation history to generate a grounded response
6. **Tracing** — Every request is traced in LangSmith for debugging and observability

---

## 📊 Observability with LangSmith

This project integrates LangSmith for full pipeline tracing:
- View retrieved chunks per query
- Monitor token usage and cost
- Debug hallucinations and retrieval quality
- Track response latency

---

## 🏨 Real-World Context

This project was inspired by real deployments across luxury hospitality properties including **Grand Hyatt Dubai**, **Atlantis The Palm**, **Rosewood Abu Dhabi**, and **Waldorf Astoria Kuwait** — where repetitive guest queries are a daily operational challenge.

---

## 🔮 Roadmap

- [ ] Deploy on Streamlit Cloud with public URL
- [ ] Support multiple hotel properties with separate knowledge bases
- [ ] Replace OpenAI with a self-hosted Llama model for data privacy
- [ ] Add analytics dashboard for most frequently asked questions
- [ ] Integrate with hotel PMS for live room availability answers

---

## 👤 Author

**Farhan Usman Mulla**
Assistant Project Manager & AI Enthusiast
Dubai, UAE

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/farhan-usman-63b57a76/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black)](https://github.com/mullafarhanusman-arch)

---

## 📄 License

MIT License — feel free to use and adapt for your own hotel or hospitality project.

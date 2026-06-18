import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# Load environment variables
load_dotenv()

if os.getenv("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
    os.environ["LANGCHAIN_PROJECT"] = "hotel-support-chatbot"

# ─────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Grand Hyatt Dubai — Guest Support",
    page_icon="🏨",
    layout="centered"
)

st.title("🏨 Grand Hyatt Dubai")
st.subheader("Guest Support Assistant")
st.caption("Ask me anything about your stay — check-in, dining, IPTV, pool, transport and more.")

# ─────────────────────────────────────────────────────────────────
# LOAD & INDEX THE FAQ PDF (runs once, cached)
# ─────────────────────────────────────────────────────────────────

@st.cache_resource
def load_knowledge_base():
    loader = PyPDFLoader("data/hotel_faq.pdf")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    return retriever

@st.cache_resource
def load_llm():
    return ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# ─────────────────────────────────────────────────────────────────
# RAG CHAIN (LCEL — modern approach)
# ─────────────────────────────────────────────────────────────────

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def build_chain(retriever, llm):
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful guest support assistant for Grand Hyatt Dubai.
Use the following context from the hotel FAQ to answer the guest's question.
If the answer is not in the context, politely say you don't have that information
and suggest they contact the front desk.

Context:
{context}"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    chain = (
        RunnablePassthrough.assign(
            context=lambda x: format_docs(retriever.invoke(x["question"]))
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

# ─────────────────────────────────────────────────────────────────
# CHAT INTERFACE
# ─────────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Welcome to Grand Hyatt Dubai! How can I assist you today?"}
    ]

with st.spinner("Loading knowledge base..."):
    retriever = load_knowledge_base()
    llm = load_llm()
    chain = build_chain(retriever, llm)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about your stay..."):

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Build chat history in LangChain message format
    chat_history = []
    for msg in st.session_state.messages[:-1]:  # exclude current message
        if msg["role"] == "user":
            chat_history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            chat_history.append(AIMessage(content=msg["content"]))

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = chain.invoke({
                "question": prompt,
                "chat_history": chat_history
            })
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

# ─────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 💡 Try asking:")
    st.markdown("""
- What time is check-in?
- Is room service available at night?
- How do I cast Netflix to the TV?
- What restaurants are open for dinner?
- How do I get to the airport?
- Is the gym open 24 hours?
- What is the WiFi password?
- Can I bring my pet?
    """)
    st.divider()
    st.caption("Powered by LangChain + OpenAI")
    if st.button("🗑️ Clear chat"):
        st.session_state.messages = [
            {"role": "assistant", "content": "Welcome to Grand Hyatt Dubai! How can I assist you today?"}
        ]
        st.rerun()
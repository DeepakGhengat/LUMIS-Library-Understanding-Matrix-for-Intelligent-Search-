import streamlit as st
import os
import time
import re
import subprocess
import threading
import pyttsx3
import speech_recognition as sr
from langchain.chains import RetrievalQA
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

# Set page config to wide and full height
st.set_page_config(layout="wide")

# ========= Load External CSS =========
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ========= Custom CSS for Layout =========
st.markdown(
    """
    <style>
    /* Ensure the main container takes the full viewport height */
    .main {
        min-height: 100vh;
        position: relative;
        padding-bottom: 120px; /* space for the fixed chat input */
    }
    /* Chat history occupies space above the chat input */
    .chat-history {
        position: relative;
        overflow-y: auto;
        max-height: calc(100vh - 150px); /* adjust if needed */
        padding: 10px;
    }
    /* Fixed chat input container at the bottom */
    .chat-input-container {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background: white;
        padding: 10px 20px;
        border-top: 1px solid #ccc;
        z-index: 1000;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ========= Setup =========
os.makedirs("files", exist_ok=True)
os.makedirs("jj", exist_ok=True)

# ========= GPU Info =========
try:
    gpu_log = subprocess.check_output([
        "nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used",
        "--format=csv,noheader,nounits"
    ])
    st.sidebar.markdown("### GPU Status")
    st.sidebar.code(gpu_log.decode("utf-8"))
except Exception:
    st.sidebar.warning("⚠️ GPU status unavailable")

# ========= Session State =========
st.session_state.setdefault("template", """
You are a knowledgeable chatbot, here to help with questions of the user. Your tone should be professional and informative.

Context: {context}
History: {history}

User: {question}
Chatbot:""")
st.session_state.setdefault("prompt", PromptTemplate(
    input_variables=["history", "context", "question"],
    template=st.session_state.template
))
st.session_state.setdefault("memory", ConversationBufferMemory(
    memory_key="history",
    return_messages=True,
    input_key="question"
))
st.session_state.setdefault("vectorstore", Chroma(
    persist_directory="jj",
    embedding_function=OllamaEmbeddings(base_url="http://localhost:11434", model="deepseek-r1:7b")
))
st.session_state.setdefault("llm", OllamaLLM(
    base_url="http://localhost:11434",
    model="deepseek-r1:7b",
    verbose=False,
    callbacks=[StreamingStdOutCallbackHandler()]
))
st.session_state.setdefault("chat_history", [])
st.session_state.setdefault("last_response", "")

# ========= TTS =========
engine = pyttsx3.init()
engine.setProperty("rate", 150)

def save_to_cache(text):
    with open("cache.txt", "w", encoding="utf-8", errors="replace") as f:
        f.write(text)

def read_from_cache():
    try:
        with open("cache.txt", "r", encoding="utf-8") as f:
            text = f.read()
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        st.error(f"TTS Error: {e}")

def stop_reading():
    try:
        engine.stop()
    except Exception:
        pass

# ========= Helper: Clean Extracted Text =========
def clean_text(text):
    # Remove common header/footer patterns and extraneous lines
    lines = text.splitlines()
    clean_lines = []
    for line in lines:
        # Skip lines that are just numbers (likely page numbers)
        if re.match(r'^\s*\d+\s*$', line):
            continue
        # Skip lines with common header/footer keywords
        if re.search(r'(Contents|CONTENTS|Page|Printed|Springer|©)', line):
            continue
        clean_lines.append(line)
    return "\n".join(clean_lines)

# ========= Helper: Process Text Input =========
def process_text_input(user_input):
    if "qa_chain" not in st.session_state:
        st.error("Please upload a PDF to initialize the chat chain.")
        return
    # Append the user's message
    st.session_state.chat_history.append({"role": "user", "message": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    # Process the question via the QA chain
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            start = time.time()
            response = st.session_state.qa_chain(user_input)
            elapsed = time.time() - start
        raw = response["result"][:10000]
        styled = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", raw).replace("\n", "<br>")
        st.markdown(f"<div style='font-family:Segoe UI;'>{styled}<br><i>⏱ {elapsed:.2f}s</i></div>", unsafe_allow_html=True)
        st.session_state.last_response = raw
    st.session_state.chat_history.append({"role": "assistant", "message": raw})
    save_to_cache(raw)

# ========= Voice Input =========
def ask_by_voice():
    if "qa_chain" not in st.session_state:
        st.error("Please upload a PDF to initialize the chat chain.")
        return
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎤 Listening...")
        audio = recognizer.listen(source)
    try:
        query = recognizer.recognize_google(audio)
        if query.strip():
            st.write(f"**🧠 Question:** {query}")
            with st.spinner("Thinking..."):
                start = time.time()
                response = st.session_state.qa_chain(query)
                elapsed = time.time() - start
            raw = response["result"][:10000]
            styled = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", raw).replace("\n", "<br>")
            st.markdown(f"<div style='font-family:Segoe UI;'>{styled}<br><i>⏱ {elapsed:.2f}s</i></div>", unsafe_allow_html=True)
            st.session_state.last_response = raw
            st.session_state.chat_history.append({"role": "user", "message": query})
            st.session_state.chat_history.append({"role": "assistant", "message": raw})
            save_to_cache(raw)
    except Exception as e:
        st.warning(f"Could not understand: {e}")

# ========= Title =========
st.title("🤖 Welcome to L.U.M.I.S ")
st.title("Library Understanding Matrix for Intelligent Search")

# ========= PDF Upload =========
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")
if uploaded_file:
    file_path = os.path.join("files", uploaded_file.name)
    if not os.path.exists(file_path):
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())
        with st.spinner("Reading & indexing document..."):
            # Load the PDF pages
            pages = PyPDFLoader(file_path).load()
            # Clean the extracted text on each page
            for doc in pages:
                doc.page_content = clean_text(doc.page_content)
            # Use a text splitter with separators to better preserve structure
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=3500,
                chunk_overlap=300,
                separators=["\n\n", "\n", " "]
            )
            chunks = text_splitter.split_documents(pages)
            st.session_state.vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=OllamaEmbeddings(model="deepseek-r1:7b")
            )
            st.session_state.vectorstore.persist()
        st.success("✅ PDF indexed")

    st.session_state.retriever = st.session_state.vectorstore.as_retriever()
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = RetrievalQA.from_chain_type(
            llm=st.session_state.llm,
            chain_type='stuff',
            retriever=st.session_state.retriever,
            chain_type_kwargs={"prompt": st.session_state.prompt, "memory": st.session_state.memory}
        )

# ========= Main Container for Chat History =========
st.markdown('<div class="chat-history">', unsafe_allow_html=True)
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["message"])
st.markdown('</div>', unsafe_allow_html=True)

# ========= Fixed Chat Input Container at Bottom =========
st.markdown('<div class="chat-input-container">', unsafe_allow_html=True)
with st.form("chat_input_form", clear_on_submit=True):
    input_col, mic_col, stop_col, speak_col = st.columns([6, 1, 1, 1])
    with input_col:
        user_input = st.text_input("Ask follow-up", label_visibility="collapsed")
    with mic_col:
        mic_clicked = st.form_submit_button("🎙️", use_container_width=True)
    with stop_col:
        stop_clicked = st.form_submit_button("⏹", use_container_width=True)
    with speak_col:
        speak_clicked = st.form_submit_button("🔊", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# ========= Action Handling =========
if mic_clicked:
    if user_input.strip():
        process_text_input(user_input)
    else:
        ask_by_voice()
elif stop_clicked:
    stop_reading()
elif speak_clicked:
    threading.Thread(target=read_from_cache, daemon=True).start()
elif user_input.strip():
    process_text_input(user_input)

# LUMIS (Library Understanding Matrix for Intelligent Search)

**LUMIS** is a Windows‑focused, interactive document search assistant. It combines PDF indexing, semantic retrieval, and local LLM inference to let you ask natural‑language questions about any PDF. Whether you need to explore research papers, manuals, or reports, LUMIS surfaces precise answers drawn directly from your documents—via both text and voice.

---

## 📝 What Is LUMIS?

- **Library Understanding Matrix**: LUMIS builds a vectorized “matrix” of your library (uploaded PDFs), enabling fast semantic search.
- **Intelligent Search**: Rather than keyword matches, it retrieves and summarizes relevant passages using a local LLM.
- **Offline & Private**: Runs entirely on your Windows machine—no cloud API calls or external data sharing.
- **Multi‑Modal Interaction**: Supports text chat, speech recognition for voice queries, and offline text‑to‑speech for responses.
- **Use Cases**: Academic research, internal documentation lookup, compliance checks, training materials, and more.

---

## 🚀 Key Features

1. **PDF Ingestion & Cleaning**
   - Loads PDFs via `PyPDFLoader`.
   - Cleans headers, footers, and page numbers.
   - Splits content into overlapping chunks for context.
2. **Vector Embeddings & Retrieval**
   - Embeds chunks with OllamaEmbeddings.
   - Stores embeddings in a persistent Chroma vector store.
   - Retrieves top‑k relevant chunks for each query.
3. **Local LLM Answer Generation**
   - Uses `OllamaLLM` to run your chosen model (e.g., `deepseek-r1:7b`).
   - Streams tokens to the UI for faster feedback.
4. **Conversation Memory**
   - Maintains chat history in `ConversationBufferMemory` so follow‑ups respect context.
5. **Voice I/O**
   - **Speech Recognition**: `speech_recognition` captures mic input and converts to text.
   - **Text‑to‑Speech**: `pyttsx3` reads out answers offline.
6. **Streamlit Web UI**
   - Responsive chat interface with fixed input bar.
   - Sidebar GPU status (via `nvidia-smi`) with graceful fallback.
   - Custom CSS for full‑height layout and scrollable history.

---

## ⚙️ Core Requirements (Windows Only)

| Category                  | Minimum / Recommended                    |
|---------------------------|------------------------------------------|
| **Operating System**      | Windows 10 (21H1+) or Windows 11          |
| **CPU**                   | 4‑core x64 (Intel i5/Ryzen 5 or better)   |
| **RAM**                   | 8 GB (16 GB+ for large docs)              |
| **Disk Space**            | 500 MB for app + 1 GB per large PDF       |
| **Python**                | 3.9 – 3.11                                |
| **CUDA GPU (optional)**   | NVIDIA RTX (20-series+) + CUDA 11+        |
| **Microphone & Speakers** | Any Windows‑compatible audio devices      |

### Software Dependencies
Install via PowerShell or Command Prompt:
```powershell
# Upgrade pip
python -m pip install --upgrade pip

# Core libraries & requirements file installations.
pip install -r requirements.txt
=================================================
pip install streamlit
pip install langchain langchain-community
pip install langchain-ollama langchain-chroma
pip install pyttsx3 SpeechRecognition
=================================================
```
> **Note:** `pyttsx3` may require [Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/).

---

## 🔧 Installation & Setup

1. **Clone Repo**
   ```powershell
   git clone https://github.com/DeepakGhengat/LUMIS-Library-Understanding-Matrix-for-Intelligent-Search-
   cd LUMIS-Library-Understanding-Matrix-for-Intelligent-Search-
   ```

2. **Virtual Environment**
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate
   ```

3. **Dependencies**
   ```powershell
   pip install -r requirements.txt
   ```
   If `requirements.txt` is absent:
   ```powershell
   pip install streamlit langchain langchain-community                langchain-ollama langchain-chroma                pyttsx3 SpeechRecognition
   ```

4. **Ollama Installation**
   - Download the Windows installer from [ollama.com/download](https://ollama.com/download).
   - Run installer → follow prompts → ensure `ollama.exe` on your `PATH`.
   - Verify:
     ```powershell
     ollama version
     ```

5. **Pull & Run a Model**
   ```powershell
   ollama pull deepseek-r1:7b
   ollama run deepseek-r1:7b
   ```
   > By default, LUMIS expects the Ollama server at `http://localhost:11434`.

6. **Folder Permissions**
   - Ensure write access to:
     - `files/` (stores uploaded PDFs)
     - `jj/` (Chroma vectorstore data)

---

## ⚙️ Configuration

### Switching Models
To use another Ollama model (e.g., `openllama-7b`):
1. Stop the running model (Ctrl+C).
2. Pull & run new model:
   ```powershell
   ollama pull openllama-7b
   ollama run openllama-7b
   ```
3. Update `lumis.py`:
   ```python
   # In vectorstore setup
   OllamaEmbeddings(base_url="http://localhost:11434", model="openllama-7b")

   # In LLM setup
   OllamaLLM(base_url="http://localhost:11434", model="openllama-7b", ...)
   ```
4. Restart the app.

---

## ▶️ Running LUMIS

With Python env activated and Ollama running:
```powershell
streamlit run lumis.py
```
- Open `http://localhost:8501` in Chrome/Edge.
- Upload PDF → wait for indexing → start querying.

---

## 📂 Project Structure
```
├── lumis.py            # Main Streamlit application
├── style.css           # UI styling
├── requirements.txt    # Python deps
├── files/              # Uploaded PDFs
├── jj/                 # Chroma embeddings store
 └── README.md           # This file
```

---

## 🛠️ Troubleshooting

- **SpeechRecognition Errors**: 
  - Install `pip install pipwin` & `pipwin install pyaudio`.
  - Check Windows Privacy Settings → Microphone → Allow access.
- **OLLAMA_NOT_FOUND**: 
  - Confirm `ollama.exe` is on `%PATH%`.
  - Restart PowerShell after install.
- **Port Conflicts**:
  - If Ollama uses a different port, set `OLLAMA_BASE_URL` env var:
    ```powershell
    setx OLLAMA_BASE_URL "http://localhost:11435"
    ```
- **GPU Detection Fails**:
  - Install/update NVIDIA drivers & CUDA toolkit.
  - Or ignore—LUMIS will run on CPU.

---

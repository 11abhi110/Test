import streamlit as st
from groq import Groq
from file_parser import parse_file, chunk_text, extract_all_metadata
from semantic_memory import SemanticMemory
import streamlit.components.v1 as components
from streamlit_ace import st_ace


# ================== CONFIG ==================
MODEL = "llama-3.1-8b-instant"
TEMPERATURE = 0.15
MAX_CONTEXT_CHARS = 6000

SYSTEM_PROMPT = {
    "role": "system",
    "content": """
You are a Copilot-style coding assistant.

- Think before answering
- Use code provided in the editor
- Write clean, correct code
- Improve, debug, or refactor as requested
"""
}


# ================== AGENT ==================
def detect_intent(query: str) -> str:
    q = query.lower()

    if "error" in q or "bug" in q:
        return "debug_code"
    if "refactor" in q or "optimize" in q:
        return "refactor_code"

    return "write_code"


def build_prompt(user_input, intent, context, metadata, code):
    return f"""
You are a senior software engineer.

Intent: {intent}

Use the following code:

-------------- CODE START --------------
{code}
-------------- CODE END ----------------

Context:
{context}

User request:
{user_input}

Respond with improved or corrected code if applicable.
"""


def scroll_to_bottom():
    components.html(
        "<script>window.scrollTo(0, document.body.scrollHeight);</script>",
        height=0,
    )


# ================== INIT ==================
client = Groq()
st.set_page_config(page_title="Abhinav Copilot", layout="wide")


# ================== CSS (UNCHANGED) ==================
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', sans-serif;
}
.block-container {
    padding-top: 1.2rem;
}
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #020617, #0f172a);
}
section[data-testid="stSidebar"] * {
    color: #e5e7eb !important;
}
.stChatMessage {
    border-radius: 14px;
}
</style>
""", unsafe_allow_html=True)


# ================== SESSION ==================
def init_session():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "semantic_memory" not in st.session_state:
        st.session_state.semantic_memory = SemanticMemory()
    if "files_loaded" not in st.session_state:
        st.session_state.files_loaded = False
    if "metadata_store" not in st.session_state:
        st.session_state.metadata_store = []
    if "editor" not in st.session_state:
        st.session_state.editor = ""

    # ✅ NEW: store last response
    if "last_response" not in st.session_state:
        st.session_state.last_response = ""

init_session()


# ================== SIDEBAR ==================
with st.sidebar:
    st.markdown("## 🧠 Copilot")

    uploaded_files = st.file_uploader("Upload files", accept_multiple_files=True)

    if st.button("Reset"):
        st.session_state.clear()
        init_session()
        st.rerun()


# ================== FILE PROCESSING ==================
if uploaded_files and not st.session_state.files_loaded:
    for file in uploaded_files:
        text = parse_file(file)
        metadata = extract_all_metadata(file, file.size // 1024)

        st.session_state.metadata_store.append(metadata)

        chunks = chunk_text(text)
        st.session_state.semantic_memory.add_chunks(chunks)

    st.session_state.files_loaded = True
    st.success("✅ Files indexed")


# ================== CODE EDITOR ==================
st.markdown("## 🧑‍💻 Code Editor")

code_input = st_ace(
    value=st.session_state.editor,
    language="python",
    theme="monokai",
    height=300,
    key="editor"
)


# ================== CHAT ==================
st.markdown("## 💬 Copilot Chat")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ✅ APPLY BUTTON (ALWAYS VISIBLE)
if st.session_state.last_response:
    if st.button("✅ Apply to Editor"):
        st.session_state.editor = st.session_state.last_response
        st.rerun()


user_input = st.chat_input("Ask about your code...")


# ================== MAIN LOGIC ==================
if user_input:
    intent = detect_intent(user_input)

    context = ""
    if st.session_state.files_loaded:
        chunks = st.session_state.semantic_memory.search(user_input, top_k=3)

        for chunk in chunks:
            if len(context) + len(chunk) > MAX_CONTEXT_CHARS:  # ✅ FIXED
                break
            context += chunk + "\n"

    metadata = ""
    if st.session_state.metadata_store:
        metadata = str(st.session_state.metadata_store[:1])

    prompt = build_prompt(user_input, intent, context, metadata, code_input)

    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        stream = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT["content"]},
                {"role": "user", "content": prompt}
            ],
            temperature=TEMPERATURE,
            stream=True,
        )

        for chunk in stream:
            token = chunk.choices[0].delta.content or ""
            full_response += token
            placeholder.markdown(full_response + "▌")

        placeholder.markdown(full_response)

    st.session_state.messages.append(
        {"role": "assistant", "content": full_response}
    )

    # ✅ SAVE RESPONSE FOR APPLY BUTTON
    st.session_state.last_response = full_response

    scroll_to_bottom()
    st.rerun()

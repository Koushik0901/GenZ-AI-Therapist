import streamlit as st
import os
import json
import uuid
import re
from datetime import datetime
import plotly.express as px

from agent import run_crew_response, get_session_title

# --- Setup ---
os.environ['SERPER_API_KEY'] = st.secrets["SERPER_API_KEY"]
os.environ['GROQ_API_KEY'] = st.secrets["GROQ_API_KEY"]

CHAT_DIR = "chat_sessions"
os.makedirs(CHAT_DIR, exist_ok=True)

st.set_page_config(
    page_title="AI Therapist",
    page_icon=":robot:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Initialize session state ---
if "messages" not in st.session_state:
    st.session_state.update({
        "messages": [],
        "meta_log": [],
        "resources_last": [],
        "session_file": None,
        "session_id": None,
        "session_title": "Untitled Session",
        "session_mode": "new",
        "selected_session": "New Session",
    })

# --- Utility Functions ---
def sanitize_filename(title):
    title = title.strip().lower().replace(" ", "_")
    return re.sub(r"[^\w_]", "", title)

def save_session_data(base_dir, messages, log, title):
    session_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    safe_title = sanitize_filename(title)
    file_name = f"{safe_title}_{session_id}.json"
    file_path = os.path.join(base_dir, file_name)
    log_entry = {
        "session_id": session_id,
        "title": title,
        "messages": messages,
        "meta": {"log": log},
    }
    with open(file_path, "w") as f:
        json.dump(log_entry, f, indent=4)
    return file_path, session_id, title

def load_session_data(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def list_sessions():
    sessions = []
    for f in os.listdir(CHAT_DIR):
        if f.endswith(".json"):
            try:
                with open(os.path.join(CHAT_DIR, f), "r") as file:
                    data = json.load(file)
                    title = data.get("title", f)
                    sessions.append((title, f))
            except Exception:
                sessions.append((f, f))
    return sessions

# --- Sidebar: Session Selection ---
session_options = list_sessions()
titles = [t[0] for t in session_options]
files = [t[1] for t in session_options]

selected_title = st.sidebar.selectbox("\U0001F4C2 Select Session", ["New Session"] + titles)
if selected_title != st.session_state.selected_session:
    st.session_state.selected_session = selected_title
    if selected_title == "New Session":
        st.session_state.update({
            "messages": [],
            "meta_log": [],
            "resources_last": [],
            "session_file": None,
            "session_id": None,
            "session_title": "Untitled Session",
            "session_mode": "new",
        })
    else:
        selected_file = files[titles.index(selected_title)]
        file_path = os.path.join(CHAT_DIR, selected_file)
        data = load_session_data(file_path)
        st.session_state.update({
            "session_id": data.get("session_id"),
            "session_title": data.get("title", "Untitled Session"),
            "messages": data.get("messages", []),
            "meta_log": data.get("meta", {}).get("log", []),
            "session_file": file_path,
            "session_mode": selected_file
        })

# --- Chat Input Handling ---
if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    if st.session_state.session_mode == "new" and not st.session_state.session_file:
        title = get_session_title(prompt)
        file_path, sid, title = save_session_data(
            CHAT_DIR, st.session_state.messages, st.session_state.meta_log, title
        )
        st.session_state.update({
            "session_title": title,
            "session_id": sid,
            "session_file": file_path,
            "session_mode": "active"
        })

    result = run_crew_response(prompt)
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["response"],
        "resources": result.get("resources", [])
    })
    st.session_state.meta_log.append({
        "timestamp": datetime.now().isoformat(),
        "sentiment": result["sentiment"],
        "intent": result["intent"]
    })

    with open(st.session_state.session_file, "w") as f:
        json.dump({
            "session_id": st.session_state.session_id,
            "title": st.session_state.session_title,
            "messages": st.session_state.messages,
            "meta": {"log": st.session_state.meta_log}
        }, f, indent=4)

# --- Display Title ---
st.title(f"GenZ AI Therapist 🤖 - {st.session_state.session_title}")

# --- Display Chat History ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "resources" in message and message["resources"]:
            st.markdown("### 🧰 Resources")
            for r in message["resources"]:
                if isinstance(r, dict) and "url" in r and "description" in r:
                    st.markdown(f"- [{r['description']}]({r['url']})")
                elif isinstance(r, str):
                    st.markdown(f"- 🔗 {r}")

# --- Sidebar: Sentiment Trend ---
if st.session_state.meta_log:
    df = {
        "timestamp": [entry["timestamp"] for entry in st.session_state.meta_log],
        "sentiment": [entry["sentiment"] for entry in st.session_state.meta_log],
        "intent": [entry["intent"] for entry in st.session_state.meta_log]
    }
    fig = px.line(df, x="timestamp", y="sentiment", title="Sentiment Trend Over Time", markers=True)
    st.sidebar.plotly_chart(fig, use_container_width=True)

# --- Sidebar: Export Session ---
if st.sidebar.button("Export Session as JSON") and st.session_state.session_file:
    with open(st.session_state.session_file, "r") as f:
        exported = f.read()
    st.sidebar.download_button(
        label="Download JSON",
        data=exported,
        file_name=os.path.basename(st.session_state.session_file),
        mime="application/json"
    )

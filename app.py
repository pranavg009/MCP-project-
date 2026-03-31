
import streamlit as st
from groq import Groq, RateLimitError
import os, json, requests, time
from duckduckgo_search import DDGS
from bs4 import BeautifulSoup

# -- Config --
client = Groq(api_key=os.environ["GROQ_API_KEY"])

st.set_page_config(page_title="MCP Tools Server", page_icon="🔧", layout="centered")
st.title("🔧 MCP Tools Server")
st.caption("AI agent with live web search, file I/O, and webpage reading — powered by Groq + Mixtral")

# -- Tool functions --
def search_web(query):
    if not query or not query.strip():
        return "Error: Query cannot be empty."
    try:
        with DDGS() as d:
            results = list(d.text(query, max_results=5))
        return "\n".join(
            f"{i+1}. {r['title']}\n   {r['href']}\n   {r['body'][:200]}"
            for i, r in enumerate(results)
        )
    except Exception as e:
        return f"Search error: {e}"

def fetch_webpage(url):
    if not url or not url.strip().startswith(("http://", "https://")):
        return "Error: URL must start with http:// or https://"
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        for t in soup(["script", "style", "nav", "footer"]): t.decompose()
        lines = [l for l in soup.get_text(separator="\n", strip=True).splitlines() if len(l.strip()) > 40]
        return "\n".join(lines[:80])
    except Exception as e:
        return f"Error: {e}"

def read_file(filepath):
    if not filepath or not filepath.strip():
        return "Error: Filepath cannot be empty."
    for bad in ["../", "..\\", ";", "|", "&", "$", "`"]:
        if bad in filepath: return "Error: Invalid characters."
    try:
        with open(filepath.strip(), "r") as f: return f.read()
    except Exception as e: return f"Error: {e}"

def write_file(filepath, content):
    try:
        with open(filepath.strip(), "w") as f: f.write(content)
        return f"Successfully written to {filepath}"
    except Exception as e: return f"Error: {e}"

TOOL_MAP = {"search_web": search_web, "fetch_webpage": fetch_webpage, "read_file": read_file, "write_file": write_file}
TOOLS = [
    {"type": "function", "function": {"name": "search_web", "description": "Search the internet", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "fetch_webpage", "description": "Read URL content", "parameters": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}}},
    {"type": "function", "function": {"name": "read_file", "description": "Read file", "parameters": {"type": "object", "properties": {"filepath": {"type": "string"}}, "required": ["filepath"]}}},
    {"type": "function", "function": {"name": "write_file", "description": "Write file", "parameters": {"type": "object", "properties": {"filepath": {"type": "string"}, "content": {"type": "string"}}, "required": ["filepath", "content"]}}}
]

SYSTEM_PROMPT = "You are a helpful assistant with tools. Call tools using the function interface."

if "messages" not in st.session_state: st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

if prompt := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)
    with st.chat_message("assistant"):
        response = client.chat.completions.create(model="llama-3.3-70b-versatile", messages=[{"role": "system", "content": SYSTEM_PROMPT}] + st.session_state.messages, tools=TOOLS)
        ans = response.choices[0].message.content or "Processing tool call..."
        st.write(ans)
        st.session_state.messages.append({"role": "assistant", "content": ans})

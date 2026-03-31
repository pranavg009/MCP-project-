import streamlit as st
from groq import Groq, RateLimitError
import os, json, requests, time
from duckduckgo_search import DDGS
from bs4 import BeautifulSoup

# ── Config ────────────────────────────────────────────────────
client = Groq(api_key=os.environ["GROQ_API_KEY"])

st.set_page_config(page_title="MCP Tools Server", page_icon="🔧", layout="centered")
st.title("🔧 MCP Tools Server")
st.caption("AI agent with live web search, file I/O, and webpage reading — powered by Groq + Llama 3.3")

# ── Tool functions ─────────────────────────────────────────────
def search_web(query):
    if not query or not query.strip():
        return "Error: Query cannot be empty."
    try:
        with DDGS() as d:
            results = list(d.text(query, max_results=5))
        if not results:
            return "No results found."
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
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        for t in soup(["script", "style", "nav", "footer", "header", "aside"]):
            t.decompose()
        lines = [l for l in soup.get_text(separator="\n", strip=True).splitlines() if len(l.strip()) > 40]
        return "\n".join(lines[:80])
    except Exception as e:
        return f"Error: {e}"

def read_file(filepath):
    if not filepath or not filepath.strip():
        return "Error: Filepath cannot be empty."
    for bad in ["../", "..\\", ";", "|", "&", "$", "`"]:
        if bad in filepath:
            return "Error: Invalid characters in filepath."
    try:
        with open(filepath.strip(), "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return f"Error: File not found: {filepath}"
    except Exception as e:
        return f"Error: {e}"

def write_file(filepath, content):
    if not filepath or not filepath.strip():
        return "Error: Filepath cannot be empty."
    for bad in ["../", "..\\", ";", "|", "&", "$", "`"]:
        if bad in filepath:
            return "Error: Invalid characters in filepath."
    if content is None:
        return "Error: Content cannot be None."
    try:
        with open(filepath.strip(), "w", encoding="utf-8") as f:
            f.write(content)
        return f"File written successfully: {filepath}"
    except Exception as e:
        return f"Error: {e}"

TOOL_MAP = {
    "search_web": search_web, "fetch_webpage": fetch_webpage,
    "read_file": read_file,   "write_file": write_file
}
TOOL_ICONS = {
    "search_web": "🔍", "fetch_webpage": "🌐",
    "read_file": "📖",  "write_file": "📝"
}

TOOLS = [
    {"type": "function", "function": {
        "name": "search_web",
        "description": "Search the internet using DuckDuckGo for current live information. Use for recent events, news, or anything needing up to date web data.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "search query"}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "fetch_webpage",
        "description": "Fetch and read text content from a webpage URL.",
        "parameters": {"type": "object", "properties": {
            "url": {"type": "string", "description": "full URL starting with http:// or https://"}},
            "required": ["url"]}}},
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Read and return the contents of a local file.",
        "parameters": {"type": "object", "properties": {
            "filepath": {"type": "string", "description": "path to the file"}},
            "required": ["filepath"]}}},
    {"type": "function", "function": {
        "name": "write_file",
        "description": "Write text content to a local file. Creates file if it does not exist.",
        "parameters": {"type": "object", "properties": {
            "filepath": {"type": "string", "description": "path to save the file"},
            "content":  {"type": "string", "description": "text content to write"}},
            "required": ["filepath", "content"]}}}
]

SYSTEM_PROMPT = """You are a strict function-calling AI assistant.

TOOLS:
- search_web
- fetch_webpage
- read_file
- write_file

STRICT RULES:
- ONLY call tools using JSON function calling
- NEVER output <function=...> or XML
- NEVER simulate tool calls in text
- If unsure, respond normally without tools
- After tool execution, return final answer
"""

# ── Session State ──────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "history" not in st.session_state:
    st.session_state.history = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Chat Input ─────────────────────────────────────────────────
if prompt := st.chat_input("Ask anything — search web, read files, fetch URLs..."):

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        status        = st.empty()
        tools_display = st.empty()
        answer        = st.empty()

        status.markdown("*🤔 Thinking...*")

        messages = (
            [{"role": "system", "content": SYSTEM_PROMPT}]
            + st.session_state.history
            + [{"role": "user", "content": prompt}]
        )

        tools_used     = []
        tools_called   = set()
        reply          = "No response generated."
        MAX_TOOL_ROUNDS = 5
        tool_round      = 0

        while tool_round < MAX_TOOL_ROUNDS:

            # Force final answer after first tool round — stops infinite loop
            current_tool_choice = "none" if tool_round > 0 else "auto"

            try:
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=messages,
                    tools=TOOLS,
                    tool_choice=current_tool_choice,
                    max_tokens=4096
                )
            except RateLimitError:
                status.markdown("*⏳ Rate limit — waiting 30s...*")
                time.sleep(30)
                continue
            except Exception as e:
                reply = f"❌ API error: {str(e)}"
                break

            assistant_message = response.choices[0].message

            messages.append({
                "role": "assistant",
                "content": assistant_message.content or "",
                "tool_calls": [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in (assistant_message.tool_calls or [])
                ] or None
            })

            # No tool calls → final answer
            if not assistant_message.tool_calls:
                reply = assistant_message.content or "No response generated."
                break

            # Execute tools
            for tc in assistant_message.tool_calls:
                tool_name = tc.function.name
                try:
                    tool_args = json.loads(tc.function.arguments)
                except Exception:
                    tool_args = {}

                call_sig = f"{tool_name}:{tc.function.arguments}"
                if call_sig in tools_called:
                    tool_result = "Already called — skipping duplicate."
                else:
                    tools_called.add(call_sig)
                    tools_used.append(tool_name)

                    icon = TOOL_ICONS.get(tool_name, "🔧")
                    hint = tool_args.get("query") or tool_args.get("url") or tool_args.get("filepath") or ""
                    status.markdown(f"*{icon} Running `{tool_name}`: `{hint[:55]}`...*")

                    try:
                        tool_result = (TOOL_MAP[tool_name](**tool_args)
                                       if tool_name in TOOL_MAP
                                       else f"Tool {tool_name} not found.")
                    except Exception as e:
                        tool_result = f"Tool error: {str(e)}"

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": str(tool_result)
                })

            tool_round += 1

        # Render output
        status.empty()

        if tools_used:
            chain = " → ".join(f"{TOOL_ICONS.get(t, '🔧')} {t}" for t in tools_used)
            tools_display.caption(f"Tools used: {chain}")

        answer.markdown(reply)

    st.session_state.history.append({"role": "user",      "content": prompt})
    st.session_state.history.append({"role": "assistant", "content": reply})
    st.session_state.messages.append({"role": "assistant", "content": reply})

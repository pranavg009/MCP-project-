
import streamlit as st
from groq import Groq, RateLimitError
import os, json, requests, time
from duckduckgo_search import DDGS
from bs4 import BeautifulSoup

# ── Config ───────────────────────────────────────────────────
client = Groq(api_key=os.environ["GROQ_API_KEY"])

st.set_page_config(page_title="MCP Tools Server", page_icon="🔧", layout="centered")
st.title("🔧 MCP Tools Server")
st.caption("AI agent with live web search, file I/O, and webpage reading — powered by Groq + Mixtral")

# ── Tool functions ────────────────────────────────────────────
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
        if bad in filepath:
            return "Error: Invalid characters in filepath."
    try:
        with open(filepath.strip(), "r") as f:
            return f.read()
    except Exception as e:
        return f"Error: {e}"

def write_file(filepath, content):
    if not filepath or not filepath.strip():
        return "Error: Filepath cannot be empty."
    for bad in ["../", "..\\", ";", "|", "&", "$", "`"]:
        if bad in filepath:
            return "Error: Invalid characters in filepath."
    try:
        with open(filepath.strip(), "w") as f:
            f.write(content)
        return f"Successfully written to {filepath}"
    except Exception as e:
        return f"Error: {e}"

TOOL_MAP = {"search_web": search_web, "fetch_webpage": fetch_webpage,
            "read_file": read_file, "write_file": write_file}

TOOL_ICONS = {"search_web": "🔍", "fetch_webpage": "🌐",
              "read_file": "📖", "write_file": "📝"}

TOOLS = [
    {"type": "function", "function": {
        "name": "search_web",
        "description": "Search the internet using DuckDuckGo for current information",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "search query"}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "fetch_webpage",
        "description": "Fetch and read content from a webpage URL",
        "parameters": {"type": "object", "properties": {
            "url": {"type": "string", "description": "full URL starting with http:// or https://"}},
            "required": ["url"]}}},
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Read a local file and return its contents",
        "parameters": {"type": "object", "properties": {
            "filepath": {"type": "string", "description": "file path"}},
            "required": ["filepath"]}}},
    {"type": "function", "function": {
        "name": "write_file",
        "description": "Write content to a local file",
        "parameters": {"type": "object", "properties": {
            "filepath": {"type": "string", "description": "file path"},
            "content": {"type": "string", "description": "content to write"}},
            "required": ["filepath", "content"]}}}
]

SYSTEM_PROMPT = """You are a helpful AI assistant with 4 tools: search_web, fetch_webpage, read_file, write_file.
IMPORTANT: Always call tools using the proper function calling interface.
Never write tool calls as plain text or XML tags.
Use tools when helpful. Be accurate and concise."""

# ── Session State ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "history" not in st.session_state:
    st.session_state.history = []

# Render existing chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Chat Input ────────────────────────────────────────────────
if prompt := st.chat_input("Ask anything — I can search web, read files, fetch URLs..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):

        # Individual placeholders — update each one independently
        status = st.empty()
        tools_display = st.empty()
        answer = st.empty()

        status.markdown("*🤔 Thinking...*")

        messages = (
            [{"role": "system", "content": SYSTEM_PROMPT}]
            + st.session_state.history
            + [{"role": "user", "content": prompt}]
        )

        tools_used = []
        reply = "No response generated."
        error_occurred = False
        MAX_TOOL_ROUNDS = 6
        tool_round = 0

        while tool_round < MAX_TOOL_ROUNDS:

            # ── Call Groq API ──────────────────────────────
            try:
                response = client.chat.completions.create(
                    model="mixtral-8x7b-32768",
                    messages=messages,
                    tools=TOOLS,
                    tool_choice="auto",
                    max_tokens=4096
                )
            except RateLimitError:
                status.markdown("*⏳ Rate limit — waiting 30s then retrying...*")
                time.sleep(30)
                continue
            except Exception as e:
                reply = f"❌ API error: {str(e)}"
                error_occurred = True
                break

            assistant_message = response.choices[0].message

            # Append assistant turn to message history
            messages.append({
                "role": "assistant",
                "content": assistant_message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in (assistant_message.tool_calls or [])
                ] or None
            })

            # ── No tool calls → final answer ──────────────
            if not assistant_message.tool_calls:
                reply = assistant_message.content or "No response generated."
                break

            # ── Execute tool calls ─────────────────────────
            for tc in assistant_message.tool_calls:
                tool_name = tc.function.name
                try:
                    tool_args = json.loads(tc.function.arguments)
                except Exception:
                    tool_args = {}

                tools_used.append(tool_name)

                # Update status to show which tool is running
                icon = TOOL_ICONS.get(tool_name, "🔧")
                hint = (tool_args.get("query") or
                        tool_args.get("url") or
                        tool_args.get("filepath") or "")
                status.markdown(f"*{icon} Running `{tool_name}`: `{hint[:60]}`...*")

                # Execute the tool
                try:
                    tool_result = (TOOL_MAP[tool_name](**tool_args)
                                   if tool_name in TOOL_MAP
                                   else f"Tool {tool_name} not found.")
                except Exception as e:
                    tool_result = f"Tool execution error: {str(e)}"

                # Append tool result to messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": str(tool_result)
                })

            tool_round += 1

        # ── Render final output ────────────────────────────
        # Clear the spinning status
        status.empty()

        # Show which tools were used
        if tools_used:
            chain = " → ".join(
                f"{TOOL_ICONS.get(t, '🔧')} {t}" for t in tools_used
            )
            tools_display.caption(f"Tools used: {chain}")

        # Show the final reply — st.markdown instead of st.write_stream
        # write_stream was causing the UI to hang on tool responses
        answer.markdown(reply)

    # Save to history for memory across turns
    st.session_state.history.append({"role": "user", "content": prompt})
    st.session_state.history.append({"role": "assistant", "content": reply})
    st.session_state.messages.append({"role": "assistant", "content": reply})

import streamlit as st
from groq import Groq
import os, json, time, datetime, requests
from duckduckgo_search import DDGS
from bs4 import BeautifulSoup

st.set_page_config(
    page_title="MCP Tools Server",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

client = Groq(api_key=os.environ["GROQ_API_KEY"])
MODEL  = "llama-3.3-70b-specdec"

MAX_HISTORY_TURNS = 10
MAX_TOOL_CALLS    = 6
TOOL_ICONS = {
    "search_web":     "🔍",
    "fetch_webpage":  "🌐",
    "read_file":      "📖",
    "write_file":     "📝",
    "summarise_text": "✂️"
}
BLOCKED_PATHS = ["..", "/etc", "/sys", "/proc", "/root", "/bin"]

# ── Tool functions ───────────────────────────────────────────
def search_web(query: str, max_results: int = 5) -> str:
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return f"No results found for: '{query}'"
        out = [f"Search results for: '{query}'\n"]
        for i, r in enumerate(results, 1):
            out.append(f"{i}. {r['title']}\n   URL: {r['href']}\n   {r['body'][:300]}\n")
        return "\n".join(out)
    except Exception as e:
        return f"Search failed: {str(e)}"

def fetch_webpage(url: str) -> str:
    if not url.startswith(("http://", "https://")):
        return "Error: URL must start with http:// or https://"
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; MCPToolsBot/1.0)"}
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script","style","nav","footer","header","aside","iframe","form"]):
            tag.decompose()
        parts = []
        for tag in soup.find_all(["h1","h2","h3","h4","p"]):
            text = tag.get_text(strip=True)
            if len(text) > 30:
                prefix = "## " if tag.name in ["h1","h2"] else "# " if tag.name in ["h3","h4"] else ""
                parts.append(prefix + text)
        if not parts:
            return f"Could not extract readable content from {url}"
        return f"Content from {url}:\n\n" + "\n\n".join(parts[:60])
    except requests.exceptions.Timeout:
        return f"Error: Request timed out for {url}"
    except requests.exceptions.HTTPError as e:
        return f"Error: HTTP {e.response.status_code} for {url}"
    except Exception as e:
        return f"Error fetching {url}: {str(e)}"

def read_file(filepath: str) -> str:
    if any(b in filepath for b in BLOCKED_PATHS):
        return "Error: Access to that path is not allowed."
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        return (f"File: '{filepath}'\n"
                f"Lines: {content.count(chr(10))+1} | Chars: {len(content)}\n"
                f"{'─'*40}\n{content}")
    except FileNotFoundError:
        return f"Error: File '{filepath}' not found."
    except Exception as e:
        return f"Error: {str(e)}"

def write_file(filepath: str, content: str) -> str:
    if any(b in filepath for b in BLOCKED_PATHS):
        return "Error: Writing to that path is not allowed."
    try:
        d = os.path.dirname(filepath)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        return (f"Written to '{filepath}' — "
                f"{os.path.getsize(filepath)} bytes, "
                f"{content.count(chr(10))+1} lines")
    except Exception as e:
        return f"Error: {str(e)}"

def summarise_text(text: str, style: str = "concise") -> str:
    if len(text.strip()) < 100:
        return "Text too short to summarise:\n\n" + text
    style_map = {
        "concise":  "Summarise in 3-5 clear sentences.",
        "bullets":  "Summarise as 5-7 bullet points.",
        "detailed": "Write a comprehensive 2-paragraph summary."
    }
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": style_map.get(style, style_map["concise"])},
                {"role": "user",   "content": f"Summarise this:\n\n{text[:6000]}"}
            ],
            max_tokens=512,
            temperature=0.3
        )
        return f"Summary ({style}):\n\n{resp.choices[0].message.content}"
    except Exception as e:
        return f"Summarisation error: {str(e)}"

TOOL_MAP = {
    "search_web":     search_web,
    "fetch_webpage":  fetch_webpage,
    "read_file":      read_file,
    "write_file":     write_file,
    "summarise_text": summarise_text
}

# ── Tool definitions ─────────────────────────────────────────
TOOL_DEFINITIONS = [
    {"type":"function","function":{
        "name":"search_web",
        "description":"Search the internet using DuckDuckGo. Use for current events, news, facts. Do NOT use if user gives a specific URL.",
        "parameters":{"type":"object","properties":{
            "query":{"type":"string"},
            "max_results":{"type":"integer","default":5}
        },"required":["query"]}}},
    {"type":"function","function":{
        "name":"fetch_webpage",
        "description":"Fetch and extract content from a specific URL. Use when user provides a link. Do NOT use for general searches.",
        "parameters":{"type":"object","properties":{
            "url":{"type":"string"}
        },"required":["url"]}}},
    {"type":"function","function":{
        "name":"read_file",
        "description":"Read a local file and return its contents with metadata.",
        "parameters":{"type":"object","properties":{
            "filepath":{"type":"string"}
        },"required":["filepath"]}}},
    {"type":"function","function":{
        "name":"write_file",
        "description":"Write content to a local file. Use when user wants to save something.",
        "parameters":{"type":"object","properties":{
            "filepath":{"type":"string"},
            "content":{"type":"string"}
        },"required":["filepath","content"]}}},
    {"type":"function","function":{
        "name":"summarise_text",
        "description":"Compress long text into a summary. Use after fetch_webpage for long pages. Chain: fetch then summarise. Styles: concise, bullets, detailed.",
        "parameters":{"type":"object","properties":{
            "text":{"type":"string"},
            "style":{"type":"string","enum":["concise","bullets","detailed"],"default":"concise"}
        },"required":["text"]}}}
]

# FIX: using full SYSTEM_PROMPT from cell 7 and triple single quotes
SYSTEM_PROMPT = '''You are an intelligent AI assistant with access to 5 powerful tools.

TOOLS AVAILABLE:
- search_web: search the internet for live, current information
- fetch_webpage: read and extract content from a specific URL
- read_file: read a local file and return its contents
- write_file: write or save content to a local file
- summarise_text: compress long text into a concise summary

TOOL USAGE RULES:
1. Always use a tool when it would genuinely improve your answer
2. For URLs provided by the user — always use fetch_webpage, never search_web
3. For general knowledge questions you already know — answer directly, no tool needed
4. Chain tools when useful: fetch_webpage then summarise_text for long pages
5. If a tool returns an error, explain what happened and try an alternative approach

OUTPUT RULES:
- Be concise and direct
- Format answers clearly using markdown when helpful
- Always cite sources (URLs) when using search or fetch results
- If writing a file, confirm what was saved and where'''

# ── Context manager ──────────────────────────────────────────
def trim_history(history):
    if len(history) <= MAX_HISTORY_TURNS * 2:
        return history
    trimmed = history[-(MAX_HISTORY_TURNS * 2):]
    while trimmed and trimmed[0]["role"] != "user":
        trimmed = trimmed[1:]
    return trimmed

def build_messages(user_message, history):
    return (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + trim_history(history)
        + [{"role": "user", "content": user_message}]
    )

# ── Agent loop with streaming ────────────────────────────────
def run_agent_streaming(user_message, history, tool_callback=None):
    messages = build_messages(user_message, history)
    tool_calls_count = 0

    while True:
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    tools=TOOL_DEFINITIONS,
                    tool_choice="auto",
                    max_tokens=1024,
                    temperature=0.5
                )
                break
            except Exception as e:
                if "rate limit" in str(e).lower() and attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                yield f"\n\nAPI error: {str(e)}"
                return

        msg = response.choices[0].message
        assistant_msg = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            assistant_msg["tool_calls"] = [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ]
        messages.append(assistant_msg)

        if not msg.tool_calls:
            try:
                stream = client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    tools=TOOL_DEFINITIONS,
                    tool_choice="none",
                    max_tokens=1024,
                    temperature=0.5,
                    stream=True
                )
                full_reply = ""
                for chunk in stream:
                    delta = chunk.choices[0].delta.content or ""
                    full_reply += delta
                    yield delta
                history.clear()
                history.extend(messages[1:])
                history.append({"role": "assistant", "content": full_reply})
            except Exception:
                yield msg.content or "No response."
                history.clear()
                history.extend(messages[1:])
            return

        tool_calls_count += len(msg.tool_calls)
        if tool_calls_count > MAX_TOOL_CALLS:
            yield "\n\nTool call limit reached."
            history.clear()
            history.extend(messages[1:])
            return

        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments)
            except Exception:
                args = {}

            if tool_callback:
                tool_callback(name, args)

            icon = TOOL_ICONS.get(name, "🔧")
            arg_val = str(list(args.values())[0])[:50] if args else ""
            yield f"\n{icon} **{name}** — `{arg_val}`\n"

            try:
                result = TOOL_MAP[name](**args) if name in TOOL_MAP else f"Tool '{name}' not found"
            except Exception as e:
                result = f"Tool error: {str(e)}"

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": str(result)
            })

# ── Session state ────────────────────────────────────────────
if "messages"  not in st.session_state: st.session_state.messages  = []
if "history"   not in st.session_state: st.session_state.history   = []
if "tool_log"  not in st.session_state: st.session_state.tool_log  = []
if "msg_count" not in st.session_state: st.session_state.msg_count = 0

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.title("🔧 MCP Tools Server")
    st.caption(f"Model: `{MODEL}`")
    st.divider()

    st.subheader("📊 Session Stats")
    col1, col2 = st.columns(2)
    col1.metric("Messages", st.session_state.msg_count)
    col2.metric("Tools Used", len(st.session_state.tool_log))
    st.divider()

    st.subheader("🔧 Tool Activity")
    if st.session_state.tool_log:
        for entry in reversed(st.session_state.tool_log[-10:]):
            st.markdown(f"`{entry['time']}` **{entry['icon']} {entry['name']}**")
            st.caption(f"   {entry['arg']}")
    else:
        st.caption("No tools used yet.")
    st.divider()

    st.subheader("🛠️ Available Tools")
    for name, icon in TOOL_ICONS.items():
        st.markdown(f"{icon} `{name}`")
    st.divider()

    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages  = []
        st.session_state.history   = []
        st.session_state.tool_log  = []
        st.session_state.msg_count = 0
        st.rerun()

# ── Main chat ────────────────────────────────────────────────
st.header("💬 Chat")

if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown(
            "👋 Hello! I'm your MCP Tools assistant.\n\n"
            "I have **5 tools** available:\n"
            "- 🔍 **search_web** — search the internet live\n"
            "- 🌐 **fetch_webpage** — read any URL\n"
            "- 📖 **read_file** — read local files\n"
            "- 📝 **write_file** — save files\n"
            "- ✂️ **summarise_text** — compress long content\n\n"
            "What would you like to know?"
        )

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask me anything..."):
    st.session_state.msg_count += 1
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        def log_tool(name, args):
            icon = TOOL_ICONS.get(name, "🔧")
            arg_val = str(list(args.values())[0])[:40] if args else ""
            st.session_state.tool_log.append({
                "time": datetime.datetime.now().strftime("%H:%M:%S"),
                "icon": icon,
                "name": name,
                "arg":  arg_val
            })

        for chunk in run_agent_streaming(
            prompt,
            st.session_state.history,
            tool_callback=log_tool
        ):
            full_response += chunk
            placeholder.markdown(full_response + "▌")

        placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
    st.rerun()

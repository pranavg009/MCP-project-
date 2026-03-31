import streamlit as st
from groq import Groq, RateLimitError
import os, json, requests, time
from duckduckgo_search import DDGS
from bs4 import BeautifulSoup

client = Groq(api_key=os.environ["GROQ_API_KEY"])

st.set_page_config(page_title="MCP AI Agent", page_icon="⚡")

st.title("⚡ MCP AI Agent")
st.caption("Fast • Stable • Tool-Enabled AI")

# Tools
def search_web(query):
    try:
        with DDGS() as d:
            results = list(d.text(query, max_results=5))
        return "\n".join(f"{r['title']}\n{r['href']}" for r in results)
    except Exception as e:
        return str(e)

def fetch_webpage(url):
    try:
        r = requests.get(url, timeout=5)
        soup = BeautifulSoup(r.text, "html.parser")
        return soup.get_text()[:1500]
    except Exception as e:
        return str(e)

def read_file(filepath):
    try:
        with open(filepath, "r") as f:
            return f.read()
    except Exception as e:
        return str(e)

def write_file(filepath, content):
    try:
        with open(filepath, "w") as f:
            f.write(content)
        return "File written"
    except Exception as e:
        return str(e)

TOOL_MAP = {
    "search_web": search_web,
    "fetch_webpage": fetch_webpage,
    "read_file": read_file,
    "write_file": write_file
}

TOOLS = [
    {"type":"function","function":{"name":"search_web","parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}}},
    {"type":"function","function":{"name":"fetch_webpage","parameters":{"type":"object","properties":{"url":{"type":"string"}},"required":["url"]}}},
    {"type":"function","function":{"name":"read_file","parameters":{"type":"object","properties":{"filepath":{"type":"string"}},"required":["filepath"]}}},
    {"type":"function","function":{"name":"write_file","parameters":{"type":"object","properties":{"filepath":{"type":"string"},"content":{"type":"string"}},"required":["filepath","content"]}}}
]

SYSTEM_PROMPT = "You are an AI assistant with tool access. Use tools when needed."

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask anything..."):

    st.session_state.messages.append({"role":"user","content":prompt})

    with st.chat_message("assistant"):

        status = st.empty()
        status.markdown("🤔 Thinking...")

        messages = [{"role":"system","content":SYSTEM_PROMPT}] + st.session_state.messages

        reply = ""
        response = None

        # ✅ SAFE retry block
        for attempt in range(2):
            try:
                time.sleep(0.8)
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=messages,
                    tools=TOOLS,
                    tool_choice="auto",
                    max_tokens=1024
                )
                break

            except RateLimitError:
                if attempt == 0:
                    status.markdown("⏳ Retrying...")
                    time.sleep(3)
                else:
                    status.empty()
                    reply = "⚠️ Rate limit reached. Try again in 10 seconds."
                    response = None

            except Exception as e:
                status.empty()
                reply = f"❌ Error: {str(e)}"
                response = None

        # 🚨 STOP if failed
        if response is None:
            st.markdown(reply)
            st.session_state.messages.append({"role":"assistant","content":reply})
            st.stop()

        msg = response.choices[0].message

        if msg.tool_calls:
            for tc in msg.tool_calls:
                name = tc.function.name
                args = json.loads(tc.function.arguments)
                result = TOOL_MAP[name](**args)

                messages.append({
                    "role":"tool",
                    "content":result,
                    "tool_call_id":tc.id
                })

            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                max_tokens=1024
            )

            reply = response.choices[0].message.content
        else:
            reply = msg.content

        if not reply:
            reply = "⚠️ No response generated."

        status.empty()
        st.markdown(reply)

    st.session_state.messages.append({"role":"assistant","content":reply})

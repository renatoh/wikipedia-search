"""
Tiny web UI for the agentic Wikipedia search.

Each browser session gets its OWN `AgenticSearch` instance (held in a
per-session gr.State), so concurrent users don't share the message list
or trip over each other's tool-call ids. Provider (openai / ollama) is
picked via `active=` in secrets.properties, or an optional CLI arg.

Usage:
    python scripts/web/app.py               # uses active provider
    python scripts/web/app.py ollama        # override for this run
    # then open http://localhost:7860
"""

import sys
from pathlib import Path

# Make `scripts/` importable so `from agents import AgenticSearch` works
# when this file is run directly (right-click Run in PyCharm, python app.py).
SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import gradio as gr  # noqa: E402

from agents import AgenticSearch  # noqa: E402


def format_trace(trace: list[dict]) -> str:
    """Render the agent's tool-call trace as a collapsible markdown block."""
    if not trace:
        return ""
    lines = []
    for evt in trace:
        if evt["kind"] == "tool":
            args = evt.get("args") or {}
            lines.append(f"- **{evt['name']}**(`{args}`)")
            if evt.get("summary"):
                lines.append(f"  - ↳ {evt['summary']}")
        elif evt["kind"] == "guard":
            lines.append(f"- ⚠️ {evt['text']}")
    body = "\n".join(lines)
    return (
        "\n\n<details><summary>🔧 tool trace "
        f"({sum(1 for e in trace if e['kind'] == 'tool')} call(s))"
        f"</summary>\n\n{body}\n\n</details>"
    )


def main() -> None:
    provider = sys.argv[1] if len(sys.argv) > 1 else None
    # A "template" agent, only used at startup to print the banner and
    # expose the model name in the UI header. Actual per-session agents
    # are created lazily inside `chat()` below.
    banner_agent = AgenticSearch(provider=provider)

    def chat(message: str, _history, agent):
        # Lazy-init one agent per browser session. Gradio persists `agent`
        # across turns in the same session via gr.State.
        if agent is None:
            agent = AgenticSearch(provider=provider)
        answer = agent.ask(message, verbose=True)
        return answer + format_trace(agent.last_trace), agent

    def reset(_agent):
        # Return a brand-new agent (equivalent to `agent.reset()` but also
        # replaces the State value) and clear the visible chat.
        return AgenticSearch(provider=provider), []

    with gr.Blocks(title="Wikipedia Agentic Search") as demo:
        gr.Markdown(f"# Wikipedia Agentic Search\n**Model:** `{banner_agent.model}`")
        # Per-session agent instance. `None` initially; created on first message.
        agent_state = gr.State(value=None)
        chatbot = gr.Chatbot(height=700)
        gr.ChatInterface(
            fn=chat,
            chatbot=chatbot,
            additional_inputs=[agent_state],
            additional_outputs=[agent_state],
            textbox=gr.Textbox(
                placeholder="Ask a question...",
                lines=3,
                max_lines=10,
            ),
            # When additional_inputs is set, each example must be a list
            # matching [message, *additional_inputs]. The agent_state value
            # of None means "no session yet" — same as a fresh visit.
            examples=[
                ["Who is the most successful tennis player in the world?", None],
                ["Who won the NBA title 2026", None],
                ["For how long has the war between USA and Iran going on?", None],
                ["Compare the population of the 5 largest cities in Switzerland.", None],
            ],
        )
        gr.Button("Reset agent memory").click(
            fn=reset, inputs=agent_state, outputs=[agent_state, chatbot]
        )

        # Dedicated stateless API endpoint for scripted / automated use.
        # Each call spins up a fresh AgenticSearch, asks one question,
        # returns {answer, trace}. Simpler contract than the ChatInterface
        # auto-API. Called via gradio_client with api_name="/ask".
        def ask_api(question: str) -> dict:
            a = AgenticSearch(provider=provider)
            answer = a.ask(question, verbose=False)
            return {"answer": answer, "trace": a.last_trace}

        gr.api(fn=ask_api, api_name="ask")

    # `queue()` allows Gradio to handle multiple concurrent requests.
    # `show_error=True` surfaces server-side exceptions in the client
    # response instead of hiding them — useful for API/testing sessions.
    demo.queue(default_concurrency_limit=4).launch(show_error=True)


if __name__ == "__main__":
    main()

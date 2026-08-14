"""
Agentic wikipedia search — a local LLM (via Ollama) drives a tool-calling
loop, calling `wikipedia_search` (backed by our hybrid SearchService) as many
times as it wants, then produces a grounded answer.

Fully offline: the LLM runs on a local Ollama server, so no API key and no
internet access are required.

Chat history is preserved across turns so follow-up questions work.

`base_url` and `model` are read from `config.properties` at the project root.

Usage:
    python3 -m agents.agentic_search
    # or just right-click → Run in PyCharm
"""

import json
import sys
from datetime import date
from pathlib import Path

# Make `scripts/` importable so `from services import ...` works when the
# file is run directly (e.g. right-click Run in PyCharm), not only when
# launched via `python -m agents.agentic_search`.
SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

PROJECT_ROOT = SCRIPTS_DIR.parent
CONFIG_PATH = PROJECT_ROOT / "config.properties"

from openai import OpenAI  # noqa: E402  (imported after sys.path fix)

from services import SearchService  # noqa: E402

SYSTEM_PROMPT = """You are a Wikipedia research assistant. Answer only \
from the results of your tool calls, not from your own knowledge.

Tools:
- `wikipedia_search(query, top_k)` — returns {id, title, snippet}. The \
snippet is just the article's first ~300 words.
- `wikipedia_get(id)` — returns the full article text.

Workflow:
- Search first, then fetch the most relevant article with wikipedia_get \
before answering. Snippets alone are rarely enough.
- If the retrieved text does not contain the answer, say so — don't guess.

Citations:
- Reference articles inline as [Article Title].
- End the answer with a "Sources:" section listing each article you used:
  Sources:
  - [Article Title] (id: 12345)"""

def load_properties(path: Path) -> dict[str, str]:
    """Parse a simple key=value properties file. Blank lines and lines
    starting with '#' are ignored. Values are not unquoted or unescaped."""
    props: dict[str, str] = {}
    if not path.exists():
        return props
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        props[key.strip()] = value.strip()
    return props


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "wikipedia_search",
            "description": (
                "Hybrid (BM25 + kNN) search over the Wikipedia index. "
                "Returns the top-K articles as {id, title, snippet}. "
                "IMPORTANT: the snippet is only the first ~300 words. "
                "Wikipedia article leads are often stale (e.g. written "
                "before an event happened). If a hit's title matches your "
                "topic, call `wikipedia_get(id)` for the full article "
                "before answering — do not rely on the snippet alone."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural-language search query.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "How many results to return (1-10).",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wikipedia_get",
            "description": (
                "Fetch the FULL text of one Wikipedia article by its id "
                "(the `id` field from a wikipedia_search hit). Use this "
                "when the snippet is not enough to answer the question."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "Document id from a search hit.",
                    },
                },
                "required": ["id"],
            },
        },
    },
]


class AgenticSearch:
    def __init__(
        self,
        search_service: SearchService | None = None,
        client: OpenAI | None = None,
        model: str | None = None,
        max_iterations: int = 8,
        snippet_words: int = 300,
    ):
        self.search = search_service or SearchService()
        if client is None:
            client, resolved_model = _build_client()
            model = model or resolved_model
        self.client = client
        self.model = model
        self.max_iterations = max_iterations
        self.snippet_words = snippet_words
        self.last_trace: list[dict] = []
        # Print on init so the active model is visible no matter which
        # entry point (CLI, web UI, or programmatic use) constructed us.
        base_url = getattr(self.client, "base_url", None)
        print(f"[AgenticSearch] model={self.model} base_url={base_url}")
        # Prepend today's date so the model can reason about "current",
        # "recent", "upcoming", etc. without falling back to its training
        # cutoff for temporal grounding.
        system_prompt = f"Today's date is {date.today().isoformat()}.\n\n{SYSTEM_PROMPT}"
        self.messages: list[dict] = [{"role": "system", "content": system_prompt}]

    def reset(self) -> None:
        """Drop conversation history, keep the system prompt."""
        self.messages = self.messages[:1]

    def _first_words(self, text: str, n: int) -> str:
        """Return the first n whitespace-delimited words of `text`. Cheap
        approximation of the article's lead section — 300 words is roughly
        the median Wikipedia lead length."""
        words = (text or "").split()
        return " ".join(words[:n])

    def _run_tool(self, name: str, arguments: dict) -> str:
        # Compatibility: some models (e.g. Command R via Ollama's OpenAI
        # shim) emit args in Cohere's native shape:
        #   {"tool_name": "...", "parameters": {"query": "..."}}
        # instead of the flat OpenAI shape {"query": "..."}. Unwrap.
        if isinstance(arguments.get("parameters"), dict):
            arguments = arguments["parameters"]

        if name == "wikipedia_search":
            return self._tool_search(arguments)
        if name == "wikipedia_get":
            return self._tool_get(arguments)
        return json.dumps({"error": f"unknown tool: {name}"})

    def _tool_search(self, arguments: dict) -> str:
        query = arguments.get("query", "")
        top_k = min(max(int(arguments.get("top_k", 5)), 1), 10)
        try:
            # OpenSearch embeds the query server-side via the neural clause.
            result = self.search.hybrid_search(query, top_k=top_k)
        except Exception as e:
            return json.dumps({"error": str(e)})

        hits = [
            {
                "id": hit["_source"].get("id"),
                "title": hit["_source"].get("title"),
                "snippet": self._first_words(hit["_source"].get("text"), self.snippet_words),
            }
            for hit in result["hits"]["hits"]
        ]
        return json.dumps({"query": query, "hits": hits}, ensure_ascii=False)

    def _tool_get(self, arguments: dict) -> str:
        raw_id = arguments.get("id")
        # Guard against JSON null / missing / literal "None" / "null" —
        # some models emit these when they lose track of the real id.
        if raw_id is None or str(raw_id).strip().lower() in ("", "none", "null"):
            return json.dumps(
                {
                    "error": (
                        "'id' must be a non-empty document id from a "
                        "wikipedia_search hit (e.g. \"17742072\"). "
                        "Call wikipedia_search first and copy the `id` "
                        "field of the hit you want."
                    )
                }
            )
        doc_id = str(raw_id).strip()
        try:
            source = self.search.get(doc_id)
        except Exception as e:
            return json.dumps({"error": str(e)})
        if not source:
            return json.dumps({"error": f"no document with id={doc_id}"})
        # Cap article text at a size that comfortably fits in num_ctx=32k
        # even with a couple of drill-ins per turn, but doesn't chop off
        # the actually-important sections (recent tournament results,
        # match-by-match tables, references — which usually sit at the
        # END of chronological articles).
        MAX_ARTICLE_CHARS = 30000
        text = source.get("text", "") or ""
        if len(text) > MAX_ARTICLE_CHARS:
            text = text[:MAX_ARTICLE_CHARS] + "\n\n[...truncated...]"
        return json.dumps(
            {
                "id": source.get("id"),
                "title": source.get("title"),
                "text": text,
            },
            ensure_ascii=False,
        )

    def ask(self, user_message: str, verbose: bool = False) -> str:
        self.messages.append({"role": "user", "content": user_message})
        # Track whether the model actually drilled in during this turn.
        # If it tries to finalize after only running searches, we inject a
        # nudge and force another loop iteration.
        did_search = False
        did_get = False
        nudged_already = False
        # Fresh trace for this turn — the caller (e.g. the web UI) can
        # inspect `agent.last_trace` after ask() returns to render it.
        self.last_trace: list[dict] = []

        for i in range(self.max_iterations):
            # Force a tool call on the very first iteration so the model
            # can't skip retrieval and answer from parametric memory. After
            # that, "auto" lets it synthesize once it has enough evidence.
            tool_choice = "required" if i == 0 else "auto"
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=TOOLS,
                tool_choice=tool_choice,
                # Bump Ollama's context window — the default (4096) silently
                # truncates once we've pulled a couple of full articles into
                # the message list, causing empty/garbled synthesis turns.
                extra_body={"options": {"num_ctx": 32768}},
            )
            message = response.choices[0].message
            # Append the assistant turn verbatim so tool_call ids match on the
            # next request — required by the Chat Completions tool protocol.
            self.messages.append(message.model_dump(exclude_none=True))

            tool_calls = message.tool_calls or []
            if not tool_calls:
                # Mechanical guard: don't let the model finalize an answer
                # after only searching. Force one drill-in attempt.
                if did_search and not did_get and not nudged_already:
                    self.last_trace.append({"kind": "guard", "text": (
                        "guard: model tried to finalize without wikipedia_get "
                        "— injecting nudge and looping"
                    )})
                    if verbose:
                        print(
                            "  · guard: model tried to finalize without "
                            "wikipedia_get — injecting nudge and looping"
                        )
                    self.messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Stop. You searched but never called "
                                "wikipedia_get. Snippets alone are not "
                                "enough — they can be stale or contain "
                                "data from a prior edition of the event. "
                                "Call wikipedia_get on the most relevant "
                                "hit before answering. If you truly do "
                                "not need more detail, explicitly say so "
                                "and justify why the snippet is enough."
                            ),
                        }
                    )
                    nudged_already = True
                    continue
                if not message.content:
                    return (
                        "(no answer from model — likely context overflow, "
                        "a timeout, or an upstream error. Try 'Reset agent "
                        "memory' and rephrase the question.)"
                    )
                return message.content

            for call in tool_calls:
                try:
                    arguments = json.loads(call.function.arguments or "{}")
                except json.JSONDecodeError:
                    arguments = {}
                if verbose:
                    print(f"  · tool: {call.function.name}({arguments})")
                if call.function.name == "wikipedia_search":
                    did_search = True
                elif call.function.name == "wikipedia_get":
                    did_get = True
                tool_result = self._run_tool(call.function.name, arguments)
                # Build a summary line for both stdout and the UI trace.
                summary = ""
                try:
                    parsed = json.loads(tool_result)
                    if "error" in parsed:
                        summary = f"ERROR: {parsed['error']}"
                    elif "hits" in parsed:
                        titles = [h.get("title") for h in parsed["hits"][:3]]
                        summary = f"{len(parsed['hits'])} hit(s): {titles}"
                    elif "title" in parsed:
                        summary = (
                            f"got: {parsed['title']} "
                            f"({len(parsed.get('text', ''))} chars)"
                        )
                except Exception:
                    pass
                self.last_trace.append(
                    {
                        "kind": "tool",
                        "name": call.function.name,
                        "args": arguments,
                        "summary": summary,
                    }
                )
                if verbose and summary:
                    print(f"    ↳ {summary}")
                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": tool_result,
                    }
                )

        # Budget exhausted — force a final answer from whatever we gathered
        # instead of returning a stub. tool_choice="none" disables further
        # tool calls, so the model must produce prose from the tool results
        # already in self.messages.
        if verbose:
            print(f"  · max_iterations ({self.max_iterations}) reached — forcing final answer")
        self.messages.append(
            {
                "role": "user",
                "content": (
                    "You've hit the tool-call budget. Do not call any more "
                    "tools. Answer the original question using only the "
                    "search results already gathered above. If the evidence "
                    "is incomplete, say what you can conclude and what "
                    "remains uncertain."
                ),
            }
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            tools=TOOLS,
            tool_choice="none",
            extra_body={"options": {"num_ctx": 32768}},
        )
        message = response.choices[0].message
        self.messages.append(message.model_dump(exclude_none=True))
        return message.content or "(no answer produced)"


def _build_client() -> tuple[OpenAI, str]:
    """Build an OpenAI-compatible client pointed at a local Ollama server.

    `base_url` and `model` are read from config.properties (the single
    source of truth). Ollama ignores the API key, so no real key (and no
    internet) is needed.
    """
    props = load_properties(CONFIG_PATH)
    base_url = props.get("base_url", "").strip()
    model = props.get("model", "").strip()
    if not base_url or not model:
        raise RuntimeError(
            f"Set both 'base_url' and 'model' in {CONFIG_PATH}."
        )
    return OpenAI(api_key="not-needed", base_url=base_url), model


def _repl() -> None:
    agent = AgenticSearch()
    print(f"agentic wikipedia search — model: {agent.model}")
    print("type a question, 'reset' to clear history, or 'exit'/'quit' to stop\n")

    while True:
        try:
            question = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            break
        if question.lower() == "reset":
            agent.reset()
            print("(history cleared)\n")
            continue

        try:
            answer = agent.ask(question, verbose=True)
            print(f"\nassistant> {answer}\n")
        except Exception as e:
            print(f"error: {e}\n")


if __name__ == "__main__":
    _repl()

"""
Agentic wikipedia search — o4-mini drives a tool-calling loop, calling
`wikipedia_search` (backed by our hybrid SearchService) as many times as it
wants, then produces a grounded answer.

Chat history is preserved across turns so follow-up questions work.

The OpenAI API key is read from `secrets.properties` at the project root
(key: `openai_api_key`). That file is git-ignored.

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
SECRETS_PATH = PROJECT_ROOT / "secrets.properties"

from openai import OpenAI  # noqa: E402  (imported after sys.path fix)

from services import SearchService  # noqa: E402

DEFAULT_MODEL = "o4-mini"

SYSTEM_PROMPT = """You are a Wikipedia research assistant.

Hard rules:
- ONLY use information returned by your tool calls to answer. Treat your \
own prior knowledge, memory, and training data as unavailable — as if you \
had none. Every fact in your answer must come from a tool result in this \
conversation.
- You MUST call `wikipedia_search` at least once before answering any \
factual question. Do not answer from memory even if you think you know.
- Never mention your training data, knowledge cutoff, that you are an AI/ \
language model, or any date about yourself. The user does not want meta- \
commentary — only the answer, grounded in the search results.
- If the retrieved evidence is insufficient or contradictory, say exactly \
that. Do NOT fall back to prior knowledge to fill gaps.

Tools:
- `wikipedia_search(query, top_k)` — hybrid BM25 + kNN search. Returns \
{id, title, snippet}. The snippet is only the article's first ~300 words, \
so the answer may live deeper in the article.
- `wikipedia_get(id)` — fetch the FULL text of one article by its id (from \
a search hit).

Workflow:
1. Call `wikipedia_search` with a focused query.
2. Look at the returned titles. If ANY hit's title closely matches the \
subject of the question, you MUST call `wikipedia_get` on its id before \
answering. Do not trust the snippet — it is only the first ~300 words of \
the article, and Wikipedia leads are often written before the event and \
not updated (e.g. "the tournament will be held..." even after it's over). \
The current facts almost always live deeper in the article body.
3. Only hedge, refuse, or give a "not yet happened / scheduled" answer \
AFTER you have retrieved the FULL text of the most relevant candidate(s) \
via `wikipedia_get` and confirmed the fact truly isn't there.
4. Prefer several focused searches over one broad one. Cite article titles \
inline like [Title] when you use a fact from a result."""

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
        provider: str | None = None,
        max_iterations: int = 8,
        snippet_words: int = 300,
    ):
        self.search = search_service or SearchService()
        if client is None:
            client, resolved_model = _build_client(provider)
            model = model or resolved_model
        self.client = client
        self.model = model or DEFAULT_MODEL
        self.max_iterations = max_iterations
        self.snippet_words = snippet_words
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
        if name == "wikipedia_search":
            return self._tool_search(arguments)
        if name == "wikipedia_get":
            return self._tool_get(arguments)
        return json.dumps({"error": f"unknown tool: {name}"})

    def _tool_search(self, arguments: dict) -> str:
        query = arguments.get("query", "")
        top_k = min(max(int(arguments.get("top_k", 5)), 1), 10)
        try:
            vector = self.search.embed(query)
            result = self.search.hybrid_search(query, vector, top_k=top_k)
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
        doc_id = str(arguments.get("id", "")).strip()
        if not doc_id:
            return json.dumps({"error": "missing 'id'"})
        try:
            source = self.search.get(doc_id)
        except Exception as e:
            return json.dumps({"error": str(e)})
        if not source:
            return json.dumps({"error": f"no document with id={doc_id}"})
        return json.dumps(
            {
                "id": source.get("id"),
                "title": source.get("title"),
                "text": source.get("text", ""),
            },
            ensure_ascii=False,
        )

    def ask(self, user_message: str, verbose: bool = False) -> str:
        self.messages.append({"role": "user", "content": user_message})

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
            )
            message = response.choices[0].message
            # Append the assistant turn verbatim so tool_call ids match on the
            # next request — required by the Chat Completions tool protocol.
            self.messages.append(message.model_dump(exclude_none=True))

            tool_calls = message.tool_calls or []
            if not tool_calls:
                return message.content or ""

            for call in tool_calls:
                try:
                    arguments = json.loads(call.function.arguments or "{}")
                except json.JSONDecodeError:
                    arguments = {}
                if verbose:
                    print(f"  · tool: {call.function.name}({arguments})")
                tool_result = self._run_tool(call.function.name, arguments)
                if verbose:
                    # Surface tool errors and hit counts so infra issues
                    # (e.g. embed server down) don't look like model bugs.
                    try:
                        parsed = json.loads(tool_result)
                        if "error" in parsed:
                            print(f"    ↳ ERROR: {parsed['error']}")
                        elif "hits" in parsed:
                            titles = [h.get("title") for h in parsed["hits"][:3]]
                            print(f"    ↳ {len(parsed['hits'])} hit(s): {titles}")
                        elif "title" in parsed:
                            print(f"    ↳ got: {parsed['title']} ({len(parsed.get('text', ''))} chars)")
                    except Exception:
                        pass
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
        )
        message = response.choices[0].message
        self.messages.append(message.model_dump(exclude_none=True))
        return message.content or "(no answer produced)"


def _build_client(provider: str | None = None) -> tuple[OpenAI, str]:
    """Build an OpenAI-compatible client from secrets.properties.

    Providers are named profiles in the properties file, e.g. `openai.*`
    and `ollama.*`. If `provider` is None, the `active=` key selects the
    default. Recognized per-profile keys:

      <name>.base_url  — optional (empty = real OpenAI)
      <name>.model     — required
      <name>.api_key   — required for real OpenAI; ignored (any value ok)
                         for local servers like Ollama.
    """
    props = load_properties(SECRETS_PATH)
    provider = (provider or props.get("active", "openai")).strip()

    prefix = f"{provider}."
    base_url = props.get(prefix + "base_url", "").strip() or None
    model = props.get(prefix + "model", "").strip() or DEFAULT_MODEL
    api_key = props.get(prefix + "api_key", "").strip()

    if base_url:
        # Local / OpenAI-compatible server — SDK requires *some* string.
        api_key = api_key or "not-needed"
    else:
        if not api_key or api_key == "REPLACE_ME":
            raise RuntimeError(
                f"'{provider}.api_key' not set in {SECRETS_PATH}. "
                f"Either set it, or run with a different provider "
                f"(e.g. `python agentic_search.py ollama`)."
            )

    return OpenAI(api_key=api_key, base_url=base_url), model


def _repl() -> None:
    # Optional positional arg: provider name (matches a *.model prefix in
    # secrets.properties). If omitted, falls back to `active=` in the file.
    provider = sys.argv[1] if len(sys.argv) > 1 else None
    agent = AgenticSearch(provider=provider)
    print(f"agentic wikipedia search — provider: {provider or 'active'}, model: {agent.model}")
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

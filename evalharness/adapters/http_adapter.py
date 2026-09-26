"""HTTP adapter: evaluate any chatbot that exposes an API.

The harness sends each question to the target URL and converts the reply
into a PipelineOutput. By default it expects:

    {"answer": "...", "sources": [{"doc_id": "...", "text": "...", "score": 0.8}]}

Real APIs all differ, so every field name is configurable with a dot path.
A chatbot replying {"data": {"reply": "..."}} works with answer_field="data.reply".

Safety rules for public deployments:
  - URL validated against SSRF before any request
  - redirects refused, so a public URL can't bounce us to an internal one
  - response bodies capped at 1 MB
  - hard timeout per request
"""

import json
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx

from evalharness.adapters.security import validate_target_url
from evalharness.schemas import PipelineOutput, RetrievedChunk

MAX_RESPONSE_BYTES = 1_000_000  # 1 MB

# Common names chatbots use for a source's id and text. We try the configured
# name first, then these, so most APIs work without any configuration.
ID_FALLBACKS = ("doc_id", "id", "source", "file", "filename", "title")
TEXT_FALLBACKS = ("text", "content", "chunk", "page_content")


class TargetError(RuntimeError):
    """The target chatbot failed or returned something we can't use."""


# ------------------------------------------------------------ helpers

def get_path(data: Any, path: str) -> Any:
    """Read a value by dot path. 'a.b' reads data['a']['b']; 'items.0' reads a list index."""
    cur = data
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        elif isinstance(cur, list) and part.isdigit() and int(part) < len(cur):
            cur = cur[int(part)]
        else:
            return None
    return cur


def set_path(data: dict, path: str, value: Any) -> None:
    """Write a value by dot path, creating nested dicts as needed."""
    parts = path.split(".")
    cur = data
    for part in parts[:-1]:
        cur = cur.setdefault(part, {})
    cur[parts[-1]] = value


def _first_key(item: dict, preferred: str, fallbacks: tuple[str, ...]) -> Any:
    if preferred in item:
        return item[preferred]
    for key in fallbacks:
        if key in item:
            return item[key]
    return None


def parse_sources(raw: Any, id_field: str = "doc_id", text_field: str = "text") -> list[RetrievedChunk]:
    """Turn whatever the chatbot returned as sources into RetrievedChunks.

    Accepts a list of dicts or a plain list of strings. Anything else means
    the chatbot doesn't expose sources, so retrieval metrics are skipped.
    """
    if not isinstance(raw, list):
        return []

    chunks: list[RetrievedChunk] = []
    for item in raw:
        position = len(chunks) + 1
        if isinstance(item, str):
            doc_id, text, score = f"source_{position}", item, 0.0
        elif isinstance(item, dict):
            doc_id = _first_key(item, id_field, ID_FALLBACKS) or f"source_{position}"
            text = _first_key(item, text_field, TEXT_FALLBACKS) or ""
            score = item.get("score", 0.0)
        else:
            continue

        try:
            score = float(score)
        except (TypeError, ValueError):
            score = 0.0

        chunks.append(RetrievedChunk(doc_id=str(doc_id), text=str(text), score=score, rank=position))
    return chunks


# ------------------------------------------------------------ adapter

@dataclass
class HttpAdapter:
    url: str
    question_field: str = "question"
    answer_field: str = "answer"
    sources_field: str = "sources"
    source_id_field: str = "doc_id"
    source_text_field: str = "text"
    headers: dict[str, str] = field(default_factory=dict)
    extra_body: dict[str, Any] = field(default_factory=dict)
    timeout_s: float = 30.0
    allow_private: Optional[bool] = None

    def __post_init__(self) -> None:
        # Fails fast: an unsafe URL never reaches the network.
        self.url = validate_target_url(self.url, self.allow_private)
        self._client = httpx.Client(
            timeout=self.timeout_s,
            follow_redirects=False,
            headers={"Content-Type": "application/json", **self.headers},
        )

    def answer(self, question: str, test_case_id: str = "adhoc") -> PipelineOutput:
        body = json.loads(json.dumps(self.extra_body))  # deep copy, never mutate config
        set_path(body, self.question_field, question)

        start = time.perf_counter()
        try:
            with self._client.stream("POST", self.url, json=body) as resp:
                if resp.is_redirect:
                    raise TargetError(
                        f"Target redirected (HTTP {resp.status_code}); redirects are blocked for safety"
                    )
                if resp.status_code >= 400:
                    raise TargetError(f"Target returned HTTP {resp.status_code}")

                raw = bytearray()
                for chunk in resp.iter_bytes():
                    raw.extend(chunk)
                    if len(raw) > MAX_RESPONSE_BYTES:
                        raise TargetError("Response larger than 1 MB; refused")

        except httpx.TimeoutException:
            raise TargetError(f"Target did not respond within {self.timeout_s:.0f}s")
        except httpx.RequestError as e:
            raise TargetError(f"Could not reach target ({e.__class__.__name__})")

        elapsed_ms = (time.perf_counter() - start) * 1000

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            raise TargetError("Target did not return valid JSON")

        answer = get_path(data, self.answer_field)
        if answer is None:
            raise TargetError(f"Response has no '{self.answer_field}' field")

        retrieved = parse_sources(
            get_path(data, self.sources_field),
            id_field=self.source_id_field,
            text_field=self.source_text_field,
        )

        return PipelineOutput(
            test_case_id=test_case_id,
            answer=str(answer).strip(),
            retrieved=retrieved,
            latency_ms=elapsed_ms,
        )

    def close(self) -> None:
        self._client.close()


# ------------------------------------------------------------ self-test

if __name__ == "__main__":
    from evalharness.adapters.security import UnsafeTargetError

    passed = failed = 0

    def check(label: str, condition: bool) -> None:
        global passed, failed
        if condition:
            passed += 1
            print(f"  OK    {label}")
        else:
            failed += 1
            print(f"  FAIL  {label}")

    print("Dot paths:")
    check("reads nested field", get_path({"data": {"reply": "hi"}}, "data.reply") == "hi")
    check("reads list index", get_path({"items": [{"t": "x"}]}, "items.0.t") == "x")
    check("missing path returns None", get_path({"a": 1}, "a.b.c") is None)
    body: dict = {}
    set_path(body, "input.query", "q")
    check("writes nested field", body == {"input": {"query": "q"}})

    print("\nSource parsing:")
    s = parse_sources([{"doc_id": "hr.pdf", "text": "leave", "score": 0.9}])
    check("standard shape", s[0].doc_id == "hr.pdf" and s[0].score == 0.9 and s[0].rank == 1)
    s = parse_sources([{"source": "it.pdf", "content": "vpn"}])
    check("fallback field names", s[0].doc_id == "it.pdf" and s[0].text == "vpn")
    s = parse_sources(["first chunk", "second chunk"])
    check("plain strings", len(s) == 2 and s[1].doc_id == "source_2" and s[1].rank == 2)
    s = parse_sources([{"doc_id": "a"}, 42, {"doc_id": "b"}])
    check("skips junk, ranks stay 1,2", [c.rank for c in s] == [1, 2])
    check("no sources -> empty list", parse_sources(None) == [])

    print("\nSafety:")
    try:
        HttpAdapter(url="http://localhost:8000/chat", allow_private=False)
        check("blocks localhost in public mode", False)
    except UnsafeTargetError:
        check("blocks localhost in public mode", True)

    print(f"\n{passed} passed, {failed} failed")
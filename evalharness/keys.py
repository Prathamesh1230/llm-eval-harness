"""API key resolution: server keys, or the visitor's own.

A public deployment can't hand out unlimited free API calls, so visitors
may supply their own Gemini and Groq keys. Those are used for that one
request and then discarded.

Rules for user-supplied keys:
  - never written to disk, never logged, never stored in the database
  - held only for the life of the request
  - validated for shape before use, so a typo fails fast with a clear
    message instead of a confusing 401 from the provider

The dataclass overrides __repr__ so a key can't leak into a stack trace
or a debug print, which is the most common way secrets end up in logs.
"""

import os
import re
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

# Shape checks only. We can't verify a key is valid without calling the
# provider, but catching obvious typos early gives a much better error.
GEMINI_PATTERN = re.compile(r"^AIza[A-Za-z0-9_\-]{30,}$")
GROQ_PATTERN = re.compile(r"^gsk_[A-Za-z0-9]{20,}$")


class KeyError_(ValueError):
    """Raised when a key is missing or obviously malformed."""


@dataclass
class ApiKeys:
    """Keys for one evaluation run."""

    gemini: str
    groq: str
    source: str = "server"  # "server" or "user"

    def __repr__(self) -> str:
        # Never let a real key reach a log line or traceback.
        return f"ApiKeys(source={self.source!r}, gemini='***', groq='***')"

    __str__ = __repr__

    @property
    def is_user_supplied(self) -> bool:
        return self.source == "user"


def _clean(value: Optional[str]) -> str:
    return (value or "").strip()


def validate_gemini(key: str) -> str:
    key = _clean(key)
    if not key:
        raise KeyError_("Gemini API key is empty")
    if not GEMINI_PATTERN.match(key):
        raise KeyError_("That doesn't look like a Gemini API key (they start with 'AIza')")
    return key


def validate_groq(key: str) -> str:
    key = _clean(key)
    if not key:
        raise KeyError_("Groq API key is empty")
    if not GROQ_PATTERN.match(key):
        raise KeyError_("That doesn't look like a Groq API key (they start with 'gsk_')")
    return key


def server_keys() -> ApiKeys:
    """The deployment's own keys, from the environment."""
    gemini = _clean(os.getenv("GEMINI_API_KEY"))
    groq = _clean(os.getenv("GROQ_API_KEY"))

    if not gemini or not groq:
        missing = []
        if not gemini:
            missing.append("GEMINI_API_KEY")
        if not groq:
            missing.append("GROQ_API_KEY")
        raise KeyError_(f"Server is missing {', '.join(missing)} in its environment")

    return ApiKeys(gemini=gemini, groq=groq, source="server")


def resolve(
    user_gemini: Optional[str] = None,
    user_groq: Optional[str] = None,
    require_user_keys: bool = False,
) -> ApiKeys:
    """Pick which keys this run uses.

    Both user keys must be supplied together — a half-configured run would
    silently spend the server's budget on one provider.

    require_user_keys is set when the visitor has used up their free runs.
    """
    gemini = _clean(user_gemini)
    groq = _clean(user_groq)

    if gemini or groq:
        if not (gemini and groq):
            missing = "Groq" if gemini else "Gemini"
            raise KeyError_(f"Both keys are needed; the {missing} key is missing")
        return ApiKeys(
            gemini=validate_gemini(gemini),
            groq=validate_groq(groq),
            source="user",
        )

    if require_user_keys:
        raise KeyError_(
            "You've used your free runs. Add your own Gemini and Groq API keys to continue."
        )

    return server_keys()


def server_keys_available() -> bool:
    """Whether the free tier can be offered at all. Shown in the UI."""
    try:
        server_keys()
        return True
    except KeyError_:
        return False


if __name__ == "__main__":
    passed = failed = 0

    def check(label: str, condition: bool) -> None:
        global passed, failed
        if condition:
            passed += 1
            print(f"  OK    {label}")
        else:
            failed += 1
            print(f"  FAIL  {label}")

    def expect_error(label: str, fn, fragment: str = "") -> None:
        global passed, failed
        try:
            fn()
            failed += 1
            print(f"  FAIL  {label} (should have raised)")
        except KeyError_ as e:
            if fragment and fragment.lower() not in str(e).lower():
                failed += 1
                print(f"  FAIL  {label} (wrong message: {e})")
            else:
                passed += 1
                print(f"  OK    {label}  ({e})")

    FAKE_GEMINI = "AIza" + "x" * 35
    FAKE_GROQ = "gsk_" + "y" * 40

    print("Key shape validation:")
    check("accepts valid gemini", validate_gemini(FAKE_GEMINI) == FAKE_GEMINI)
    check("accepts valid groq", validate_groq(FAKE_GROQ) == FAKE_GROQ)
    check("trims whitespace", validate_gemini(f"  {FAKE_GEMINI}  ") == FAKE_GEMINI)
    expect_error("rejects wrong gemini prefix", lambda: validate_gemini("sk-abc123xyz"), "AIza")
    expect_error("rejects wrong groq prefix", lambda: validate_groq("AIzaSomething"), "gsk_")
    expect_error("rejects empty gemini", lambda: validate_gemini(""), "empty")
    expect_error("rejects too short", lambda: validate_gemini("AIzaShort"), "doesn't look like")

    print("\nResolution:")
    keys = resolve(FAKE_GEMINI, FAKE_GROQ)
    check("user keys used when given", keys.is_user_supplied)
    check("source labelled", keys.source == "user")

    expect_error("rejects gemini alone", lambda: resolve(FAKE_GEMINI, None), "Groq")
    expect_error("rejects groq alone", lambda: resolve(None, FAKE_GROQ), "Gemini")
    expect_error(
        "demands keys when free runs spent",
        lambda: resolve(None, None, require_user_keys=True),
        "free runs",
    )

    if server_keys_available():
        k = resolve(None, None)
        check("falls back to server keys", k.source == "server")
    else:
        print("  SKIP  server key fallback (no keys in .env)")

    print("\nLeak protection:")
    keys = resolve(FAKE_GEMINI, FAKE_GROQ)
    check("repr hides keys", FAKE_GEMINI not in repr(keys))
    check("str hides keys", FAKE_GEMINI not in str(keys))
    check("f-string hides keys", FAKE_GROQ not in f"{keys}")
    check("still usable internally", keys.gemini == FAKE_GEMINI)

    print(f"\n{passed} passed, {failed} failed")
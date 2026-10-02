"""
One chat-completion call to the planner VLM, with the latency accounting COMRES needs.

Plans are released ceil(L_s / dt) steps after the query, so L_s must be the latency of
a call at single-request load, not of a call queued behind other simulations' calls.
Every process therefore takes one cross-process `fcntl.flock` around the HTTP request
(and only around it: image rendering and prompt building run outside), and reports the
time inside the lock (`call_seconds`, what the simulation is charged) separately from
the time spent waiting for it (`lock_wait_seconds`, never charged).

The endpoint and model come from the environment (COMRES_VLM_BASE_URL,
COMRES_VLM_MODEL; defaults: serve_vlm.sh's port and served name), so a second model can be
served on another port and measured with the same prompt, image and schema.

Thinking is off (Qwen3 chat template, enable_thinking False): a plan is a few hundred tokens.

COMRES_VLM_PROVIDER picks who answers: "local" (default; the vLLM server above), "anthropic"
(Claude API, key from ANTHROPIC_API_KEY or an `ant auth login` profile) or "openai" (OpenAI
API, key from OPENAI_API_KEY); COMRES_VLM_MODEL names the model there. Same messages (the
picture in full: OpenAI's detail "high", which our pictures fit unscaled), thinking off
(Claude: thinking "between_tools", which with no tools means none; OpenAI: reasoning effort
"none"), no temperature (Claude's current models reject one), no fallback model, the same lock
and timing. A temporary API failure (network error, timeout, rate limit, server error) is tried
again, up to API_RETRIES times after API_BACKOFF seconds, inside the lock; only the attempt that
answered is charged (call_seconds), the failed attempts and the waits are recorded apart
(api_retries, api_retry_seconds), like the lock wait. A refused request (bad key, bad request)
is not retried. The local server keeps no retries. The reply schema goes to the API without the four keywords only our
server's grammar enforces (minItems, maxItems, minimum, maximum: one piece per robot, chances
in [0, 1]); parse_reply already takes a robot's first valid piece and normalises the chances,
so a reply that breaks them is read the same way, and validate() can still report it.

COMRES_FAKE_VLM=1 answers without a server: a random reply that satisfies the schema,
seeded by a hash of the messages (so a replayed query gets the same reply), with
call_seconds 0.5. COMRES_FAKE_VLM=empty returns an empty reply, which parses to no plan.
"""

import contextlib
import fcntl
import hashlib
import json
import os
import random
import time
from typing import Dict, List, Optional, Tuple

EXTRA_BODY = {"chat_template_kwargs": {"enable_thinking": False}}
MAX_TOKENS = 1024                      # a plan is a few hundred tokens
DEFAULT_BASE_URL = "http://127.0.0.1:18765/v1"
DEFAULT_MODEL = "qwen3.8-27b"
DEFAULT_LOCK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp", "comres_vlm.lock")
FAKE_CALL_SECONDS = 0.5

PROVIDERS = ("local", "anthropic", "openai")
API_URLS = {"anthropic": "https://api.anthropic.com", "openai": "https://api.openai.com/v1"}
API_DROPPED_KEYWORDS = ("minItems", "maxItems", "minimum", "maximum")
API_RETRIES = 5                        # further attempts after a temporary API failure
API_BACKOFF = (1.0, 2.0, 4.0, 8.0, 16.0)   # seconds before each of them
API_TIMEOUT = 120.0                    # s per API attempt (COMRES_VLM_API_TIMEOUT); replies take a few s

_clients: Dict[Tuple[str, float], object] = {}


def provider() -> str:
    p = os.environ.get("COMRES_VLM_PROVIDER", "").strip().lower() or "local"
    if p not in PROVIDERS:
        raise ValueError(f"COMRES_VLM_PROVIDER must be one of {PROVIDERS}, got {p!r}")
    return p


def default_base_url() -> str:
    p = provider()
    return API_URLS[p] if p != "local" else os.environ.get("COMRES_VLM_BASE_URL", DEFAULT_BASE_URL)


def model_name() -> str:
    return os.environ.get("COMRES_VLM_MODEL", DEFAULT_MODEL)


def fake_mode() -> str:
    """"" (real server), "1" (random valid reply) or "empty" (no reply)."""
    v = os.environ.get("COMRES_FAKE_VLM", "").strip().lower()
    return "" if v in ("", "0", "false", "no") else v


def _client(url: str):
    timeout = float(os.environ.get("COMRES_VLM_TIMEOUT", "1800"))
    key = (url, timeout)
    if key not in _clients:
        from openai import OpenAI
        _clients[key] = OpenAI(base_url=url, api_key=os.environ.get("COMRES_VLM_API_KEY", "EMPTY"),
                               timeout=timeout, max_retries=0)
    return _clients[key]


def _api_client(prov: str):
    """The Claude or OpenAI API client: credentials from the environment, no retries."""
    timeout = float(os.environ.get("COMRES_VLM_API_TIMEOUT", API_TIMEOUT))
    key = (prov, timeout)
    if key not in _clients:
        if prov == "anthropic":
            import anthropic
            _clients[key] = anthropic.Anthropic(timeout=timeout, max_retries=0)
        else:
            from openai import OpenAI
            _clients[key] = OpenAI(timeout=timeout, max_retries=0)
    return _clients[key]


def temporary_failure(e: Exception) -> bool:
    """Worth another attempt: a network error or timeout, or HTTP 408, 409, 429 or 5xx (both SDKs)."""
    status = getattr(e, "status_code", None)
    if isinstance(status, int):
        return status in (408, 409, 429) or status >= 500
    names = {c.__name__ for c in type(e).__mro__}
    return bool(names & {"APIConnectionError", "APITimeoutError"}) or isinstance(e, (ConnectionError, TimeoutError))


def api_schema(schema: Dict) -> Dict:
    """The reply schema without API_DROPPED_KEYWORDS (see the module docstring)."""
    if isinstance(schema, dict):
        return {k: api_schema(v) for k, v in schema.items() if k not in API_DROPPED_KEYWORDS}
    if isinstance(schema, list):
        return [api_schema(v) for v in schema]
    return schema


def anthropic_request(messages: List[Dict], schema: Dict, model: str) -> Dict:
    """Messages API request: the same parts in the same order (data-URL images as base64 blocks)."""
    system, out = [], []
    for m in messages:
        parts = m["content"] if isinstance(m["content"], list) else [{"type": "text", "text": m["content"]}]
        blocks = []
        for part in parts:
            if part["type"] == "text":
                blocks.append({"type": "text", "text": part["text"]})
            elif part["type"] == "image_url":
                head, data = part["image_url"]["url"].split(",", 1)
                blocks.append({"type": "image", "source": {"type": "base64", "data": data,
                                                           "media_type": head[len("data:"):].split(";")[0]}})
            else:
                raise ValueError(f"unknown message part {part['type']!r}")
        if m["role"] == "system":
            system += blocks
        else:
            out.append({"role": m["role"], "content": blocks})
    req = dict(model=model, max_tokens=MAX_TOKENS, messages=out, thinking={"type": "between_tools"},
               output_config={"format": {"type": "json_schema", "schema": api_schema(schema)}})
    if system:
        req["system"] = system
    return req


def openai_request(messages: List[Dict], schema: Dict, model: str) -> Dict:
    """Chat Completions request: the same messages, pictures at detail "high"."""
    msgs = []
    for m in messages:
        if isinstance(m["content"], list):
            m = dict(m, content=[dict(p, image_url=dict(p["image_url"], detail="high")) if p["type"] == "image_url" else p
                                 for p in m["content"]])
        msgs.append(m)
    return dict(model=model, messages=msgs, max_completion_tokens=MAX_TOKENS, reasoning_effort="none",
                response_format={"type": "json_schema",
                                 "json_schema": {"name": "plan", "schema": api_schema(schema), "strict": True}})


def _read_reply(prov: str, resp) -> Tuple[str, Dict]:
    """(reply text, the usage fields of call()) of a local / Anthropic / OpenAI response."""
    if prov == "anthropic":
        u = resp.usage
        prompt = sum(int(getattr(u, k, None) or 0) for k in
                     ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"))
        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text").strip()
        return text, dict(prompt_tokens=prompt, output_tokens=getattr(u, "output_tokens", None),
                          reasoning_tokens=None, finish_reason=resp.stop_reason)
    choice = resp.choices[0]
    u = resp.usage
    details = getattr(u, "completion_tokens_details", None) if u is not None else None
    return (choice.message.content or "").strip(), dict(
        prompt_tokens=getattr(u, "prompt_tokens", None), output_tokens=getattr(u, "completion_tokens", None),
        reasoning_tokens=getattr(details, "reasoning_tokens", None), finish_reason=choice.finish_reason)


@contextlib.contextmanager
def server_lock():
    """The cross-process lock every VLM request is made in (see above), shared by COMRES and
    Co-NavGPT: a request never waits at the server behind another run's request, so its
    measured time is its own."""
    lock_path = os.environ.get("COMRES_VLM_LOCK", DEFAULT_LOCK)
    os.makedirs(os.path.dirname(lock_path) or ".", exist_ok=True)
    with open(lock_path, "a+") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def call(messages: List[Dict], schema: Dict, model: Optional[str] = None,
         base_url: Optional[str] = None) -> Tuple[str, Dict]:
    """
    (reply text, usage), thinking off. usage = {call_seconds, lock_wait_seconds, prompt_tokens,
    output_tokens, reasoning_tokens (the server's count: 0 with thinking off), finish_reason,
    model, fake}. Raises on HTTP / server errors; the caller decides what a failed call means.
    """
    model = model or model_name()
    fake = fake_mode()
    if fake:
        return _fake_call(messages, schema, model, fake)

    prov = provider()
    if prov == "anthropic":
        client, kwargs = _api_client(prov).messages, anthropic_request(messages, schema, model)
    elif prov == "openai":
        client, kwargs = _api_client(prov).chat.completions, openai_request(messages, schema, model)
    else:
        client = _client(base_url or default_base_url()).chat.completions
        kwargs = dict(model=model, messages=messages, temperature=0.0, max_tokens=MAX_TOKENS,
                      response_format={"type": "json_schema",
                                       "json_schema": {"name": "plan", "schema": schema}},
                      extra_body=EXTRA_BODY)
    retries = 0
    t0 = time.perf_counter()
    with server_lock():
        t1 = time.perf_counter()
        while True:
            ta = time.perf_counter()
            try:
                resp = client.create(**kwargs)
                break
            except Exception as e:
                if prov == "local" or retries >= API_RETRIES or not temporary_failure(e):
                    raise
                print(f"[WARN] {prov} call failed ({type(e).__name__}: {str(e)[:120]}); "
                      f"retry {retries + 1}/{API_RETRIES} in {API_BACKOFF[retries]:.0f} s")
                time.sleep(API_BACKOFF[retries])
                retries += 1
        t2 = time.perf_counter()

    text, read = _read_reply(prov, resp)
    usage = dict(call_seconds=round(t2 - ta, 4), lock_wait_seconds=round(t1 - t0, 4), **read,
                 model=model, fake=False)
    if prov != "local":
        usage.update(api_retries=retries, api_retry_seconds=round(ta - t1, 4))
    return text, usage


# ---------------------------------------------------------------------------
# Schema check (offline scoring and tests; the planner itself never repairs)
# ---------------------------------------------------------------------------

_TYPES = {"object": dict, "array": list, "string": str, "boolean": bool}


def validate(value, schema: Dict, path: str = "$") -> List[str]:
    """Violations of the JSON-schema subset the planner uses ([] = valid)."""
    errs: List[str] = []
    if "enum" in schema and value not in schema["enum"]:
        errs.append(f"{path}: {value!r} not in enum")
    t = schema.get("type")
    if t in ("number", "integer"):
        ok = isinstance(value, (int, float)) and not isinstance(value, bool)
        if t == "integer":
            ok = ok and float(value).is_integer()
        if not ok:
            return errs + [f"{path}: expected {t}, got {type(value).__name__}"]
        if "minimum" in schema and value < schema["minimum"]:
            errs.append(f"{path}: {value} < minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errs.append(f"{path}: {value} > maximum {schema['maximum']}")
        return errs
    if t in _TYPES and not isinstance(value, _TYPES[t]):
        return errs + [f"{path}: expected {t}, got {type(value).__name__}"]
    if t == "object":
        props = schema.get("properties", {})
        for k in schema.get("required", []):
            if k not in value:
                errs.append(f"{path}: missing {k!r}")
        for k, v in value.items():
            if k in props:
                errs += validate(v, props[k], f"{path}.{k}")
            elif schema.get("additionalProperties") is False:
                errs.append(f"{path}: unexpected key {k!r}")
    elif t == "array":
        if len(value) < schema.get("minItems", 0):
            errs.append(f"{path}: {len(value)} items < minItems {schema['minItems']}")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errs.append(f"{path}: {len(value)} items > maxItems {schema['maxItems']}")
        for j, v in enumerate(value):
            errs += validate(v, schema.get("items", {}), f"{path}[{j}]")
    return errs


# ---------------------------------------------------------------------------
# Fake replies (tests, dry runs)
# ---------------------------------------------------------------------------

def _messages_seed(messages: List[Dict]) -> int:
    blob = json.dumps(messages, sort_keys=True, default=str).encode()
    return int.from_bytes(hashlib.sha256(blob).digest()[:8], "big")


def random_instance(schema: Dict, rng: random.Random):
    """A random value satisfying the subset of JSON schema the planner uses."""
    if "enum" in schema:
        return rng.choice(schema["enum"])
    t = schema.get("type")
    if t == "object":
        props = schema.get("properties", {})
        return {k: random_instance(v, rng) for k, v in props.items()}
    if t == "array":
        lo = int(schema.get("minItems", 0))
        hi = int(schema.get("maxItems", max(lo, 3)))
        return [random_instance(schema.get("items", {}), rng) for _ in range(rng.randint(lo, hi))]
    if t == "number":
        lo, hi = float(schema.get("minimum", 0.0)), float(schema.get("maximum", 1.0))
        return round(rng.uniform(lo, hi), 3)
    if t == "integer":
        return rng.randint(int(schema.get("minimum", 0)), int(schema.get("maximum", 100)))
    if t == "boolean":
        return rng.random() < 0.5
    if t == "string":
        return "x" * rng.randint(1, 8)
    return None


def _fake_call(messages, schema, model, fake) -> Tuple[str, Dict]:
    if fake == "empty":
        text = ""
    else:
        text = json.dumps(random_instance(schema, random.Random(_messages_seed(messages))))
    usage = dict(call_seconds=FAKE_CALL_SECONDS, lock_wait_seconds=0.0, prompt_tokens=None,
                 output_tokens=None, reasoning_tokens=None,
                 finish_reason="fake" if text else "fake_empty", model=model, fake=True)
    return text, usage

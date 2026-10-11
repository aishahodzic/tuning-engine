"""Tuning Engine: one reusable fine-tuning backend for every Möbius app.

Other apps call it at /api/services/tuning-engine/v1/<route> with their own app
token; agents call the same functions as tools. The engine owns the W&B /
CoreWeave Forge connection, the live provider catalogue (methods, models,
serverless flags), dataset upload and validation, and the checkpointed training
loop (start, advance, stop-and-keep-latest-checkpoint, metrics, inference).

Approval policy (owner decision, 2026-10-11): an app that asks the engine to
start a run is trusted to have asked its user first; runs bill the W&B account
connected to this engine install. The engine never returns the stored key.
"""
import base64
import concurrent.futures
import fcntl
import hashlib
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path

APP_ID = os.environ.get("APP_ID", "")
APP_SLUG = os.environ.get("APP_SLUG", "tuning-engine")
API_BASE = os.environ.get("API_BASE_URL", "").rstrip("/")
APP_TOKEN = os.environ.get("APP_TOKEN", "")
STORE = Path(os.environ.get("APP_STORAGE_DIR") or f"/data/apps/{APP_ID}")
SECRET_NAME = "wandb-api-key"
UA = "TuningEngine/1.0"

FORGE_URL = "https://forge.coreweave.com/api/training/v1"
WANDB_GRAPHQL_URL = "https://api.wandb.ai/graphql"
INFERENCE_MODELS_URL = "https://api.inference.wandb.ai/v1/models"
DOCS = "https://docs.coreweave.com/products"
TRAINING_DOCS = f"{DOCS}/post-training/serverless-training"
INFERENCE_MODELS_DOC = f"{DOCS}/inference/serverless/models.md"
LORA_SERVING_DOC = f"{DOCS}/inference/serverless/lora.md"

CATALOGUE_TTL = 6 * 3600
CAPABILITIES_TTL = 3600
EXAMPLES_PER_MINUTE = 150          # measured Forge SFT throughput (packed), used to size checkpoint chunks
DEFAULT_LR = 5e-5                  # ART's SFT default when the caller leaves it blank
MAX_ROWS_PER_APPEND = 20000
MAX_DATASET_ROWS = 2_000_000
BUDGET_S = 11.0                    # stay inside the platform's 15 s service limit
ACTIVE = {"QUEUED", "STARTING", "PENDING", "RUNNING"}
ENDED = {"COMPLETED", "STOPPED", "FAILED", "CANCELLED"}
ROLES = {"system", "user", "assistant", "tool"}


class Problem(Exception):
    def __init__(self, status, message):
        self.status = status
        self.message = message


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _iso_seconds(value):
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


# ---------------------------------------------------------------- storage

def _path(*parts):
    return STORE.joinpath(*parts)


def _read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return default


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=1))
    tmp.replace(path)


def _safe_id(value, prefix):
    value = str(value or "")
    if not re.fullmatch(prefix + r"_[a-f0-9]{12}", value):
        raise Problem(404, "Not found.")
    return value


class _Locked:
    """Exclusive per-record lock. nonblocking=True returns None when someone else holds it."""

    def __init__(self, path, nonblocking=False):
        self.path, self.nonblocking, self.handle = Path(path), nonblocking, None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = open(self.path.with_name(self.path.name + ".lock"), "w")
        try:
            fcntl.flock(self.handle, fcntl.LOCK_EX | (fcntl.LOCK_NB if self.nonblocking else 0))
        except OSError:
            self.handle.close()
            self.handle = None
            return None
        return self

    def __exit__(self, *exc):
        if self.handle:
            self.handle.close()


# ---------------------------------------------------------------- secrets / HTTP

def platform_secret():
    if not API_BASE or not APP_TOKEN or not APP_ID.isdigit():
        raise Problem(503, "The engine's private connection store is unavailable.")
    req = urllib.request.Request(f"{API_BASE}/api/apps/{APP_ID}/secrets/{SECRET_NAME}",
                                 headers={"Authorization": f"Bearer {APP_TOKEN}"})
    try:
        with urllib.request.urlopen(req, timeout=4) as response:
            value = response.read(8193).decode("utf-8")
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise Problem(502, "The engine could not read its encrypted connection.")
    except Exception:
        raise Problem(502, "The engine could not read its encrypted connection.")
    return value.strip() if value and len(value) <= 8192 else None


def require_key():
    key = platform_secret()
    if not key:
        raise Problem(409, "The Tuning Engine is not connected to W&B yet. Open the Tuning Engine app and connect it.")
    return key


def platform_store_secret(value):
    req = urllib.request.Request(f"{API_BASE}/api/apps/{APP_ID}/secrets/{SECRET_NAME}",
                                 data=json.dumps({"value": value}).encode(), method="PUT",
                                 headers={"Authorization": f"Bearer {APP_TOKEN}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=4) as response:
            response.read(1)
    except Exception:
        raise Problem(502, "The engine could not save the encrypted W&B connection.")


def _http(url, *, key=None, method="GET", body=None, basic=False, raw=None, headers=None, limit=4 * 1024 * 1024, timeout=12):
    h = {"Accept": "application/json", "User-Agent": UA, **(headers or {})}
    if key:
        h["Authorization"] = ("Basic " + base64.b64encode(f"api:{key}".encode()).decode()) if basic else f"Bearer {key}"
    data = raw
    if body is not None:
        data = json.dumps(body).encode()
        h["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, method=method, headers=h)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            payload = response.read(limit + 1)
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = json.loads(exc.read(4096) or b"{}").get("detail") or ""
        except Exception:
            pass
        if isinstance(detail, list):
            detail = "; ".join(str(d.get("msg", d)) if isinstance(d, dict) else str(d) for d in detail)
        if exc.code in (401, 403):
            raise Problem(401, "W&B rejected the engine's API key.")
        if exc.code == 404:
            raise Problem(404, "The provider could not find that resource.")
        msg = f"The provider returned status {exc.code}" + (f": {str(detail)[:300]}" if detail else ".")
        if key:
            msg = msg.replace(key, "[key]")
        raise Problem(502 if exc.code >= 500 else 400, msg)
    except (urllib.error.URLError, TimeoutError):
        raise Problem(502, "The provider could not be reached.")
    if len(payload) > limit:
        raise Problem(502, "The provider returned too much data.")
    if not payload:
        return {}
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return payload.decode("utf-8", "replace")


def _gql(key, query, variables=None):
    out = _http(WANDB_GRAPHQL_URL, key=key, basic=True, method="POST", body={"query": query, "variables": variables or {}})
    if not isinstance(out, dict) or out.get("errors"):
        errors = (out or {}).get("errors") if isinstance(out, dict) else None
        raise Problem(502, "W&B: " + "; ".join(e.get("message", "?") for e in errors or [])[:300])
    return out.get("data") or {}


def _put(url, data, header_list, timeout=60):
    headers = {}
    for item in header_list or []:
        name, _, value = str(item).partition(":")
        headers[name.strip()] = value.strip()
    _http(url, method="PUT", raw=data, headers=headers, timeout=timeout)


def _forge(key, path, method="GET", body=None):
    return _http(f"{FORGE_URL}{path}", key=key, method=method, body=body)


def _entity(key):
    cached = _read(_path("account.json"), {})
    fingerprint = hashlib.sha256(key.encode()).hexdigest()[:16]
    if cached.get("fingerprint") == fingerprint and cached.get("entity"):
        return cached["entity"]
    viewer = _gql(key, "query{viewer{entity username}}").get("viewer") or {}
    if not viewer.get("entity"):
        raise Problem(401, "W&B did not return an account for the engine's key.")
    _write(_path("account.json"), {"fingerprint": fingerprint, "entity": viewer["entity"], "username": viewer.get("username")})
    return viewer["entity"]


# ---------------------------------------------------------------- W&B artifacts (dependency-free, bounded)

def _ensure_wandb_run(key, entity, project, run_name, display_name):
    _gql(key, "mutation($input: UpsertModelInput!){upsertModel(input:$input){project{id}}}",
         {"input": {"name": project, "entityName": entity}})
    v = {"input": {"name": run_name, "modelName": project, "entityName": entity}}
    if display_name:
        v["input"]["displayName"] = display_name[:128]
    _gql(key, "mutation($input: UpsertBucketInput!){upsertBucket(input:$input){bucket{id}}}", v)


def _upload_artifact(key, entity, project, run_name, name, files, *, type_="dataset", metadata=None):
    """Upload {filename: bytes} as a committed W&B artifact; returns its wandb-artifact:/// URI."""
    entries = {f: {"b64": base64.b64encode(hashlib.md5(d).digest()).decode(), "size": len(d), "data": d}
               for f, d in files.items()}
    hasher = hashlib.md5(b"wandb-artifact-manifest-v1\n")
    for f in sorted(entries):
        hasher.update(f"{f}:{entries[f]['b64']}\n".encode())
    created = _gql(key, "mutation($input: CreateArtifactInput!){createArtifact(input:$input){artifact{id state versionIndex}}}",
                   {"input": {"artifactTypeName": type_, "artifactCollectionNames": [name], "entityName": entity,
                              "projectName": project, "runName": run_name, "digest": hasher.hexdigest(),
                              "digestAlgorithm": "MANIFEST_MD5",
                              "aliases": [{"artifactCollectionName": name, "alias": "latest"}],
                              "metadata": json.dumps(metadata or {}), "clientID": str(uuid.uuid4()),
                              "sequenceClientID": str(uuid.uuid4()), "enableDigestDeduplication": True}})
    art = created["createArtifact"]["artifact"]
    if art["state"] != "COMMITTED":
        manifest = {"version": 1, "storagePolicy": "wandb-storage-policy-v1",
                    "storagePolicyConfig": {"storageLayout": "V2"},
                    "contents": {f: {"digest": e["b64"], "birthArtifactID": art["id"], "size": e["size"]}
                                 for f, e in entries.items()}}
        body = json.dumps(manifest, indent=4, sort_keys=True).encode()
        m = _gql(key, "mutation($input: CreateArtifactManifestInput!){createArtifactManifest(input:$input){artifactManifest{file{uploadUrl uploadHeaders}}}}",
                 {"input": {"name": "wandb_manifest.json", "digest": base64.b64encode(hashlib.md5(body).digest()).decode(),
                            "artifactID": art["id"], "entityName": entity, "projectName": project,
                            "runName": run_name, "includeUpload": True, "type": "FULL"}})
        f = m["createArtifactManifest"]["artifactManifest"]["file"]
        _put(f["uploadUrl"], body, f.get("uploadHeaders"))
        nodes = _gql(key, "mutation($input: CreateArtifactFilesInput!){createArtifactFiles(input:$input){files{edges{node{uploadUrl uploadHeaders}}}}}",
                     {"input": {"storageLayout": "V2", "artifactFiles": [
                         {"artifactID": art["id"], "name": f, "md5": e["b64"]} for f, e in entries.items()]}})
        for (fname, e), edge in zip(entries.items(), nodes["createArtifactFiles"]["files"]["edges"]):
            if edge["node"].get("uploadUrl"):
                _put(edge["node"]["uploadUrl"], e["data"], edge["node"].get("uploadHeaders"))
        _gql(key, "mutation($input: CommitArtifactInput!){commitArtifact(input:$input){artifact{id}}}",
             {"input": {"artifactID": art["id"]}})
    version = art.get("versionIndex")
    if version is None:
        version = (_gql(key, "query($id: ID!){artifact(id:$id){versionIndex}}", {"id": art["id"]}).get("artifact") or {}).get("versionIndex", 0)
    return f"wandb-artifact:///{entity}/{project}/{name}:v{version}"


# ---------------------------------------------------------------- catalogue

def _public_text(url):
    out = _http(url, headers={"Accept": "text/markdown,text/plain"}, limit=1024 * 1024)
    return out if isinstance(out, str) else json.dumps(out)


def _doc_model_ids(markdown, section):
    start = markdown.find(section)
    if start < 0:
        return []
    part = markdown[start:]
    nxt = re.search(r"\n## ", part[len(section):])
    if nxt:
        part = part[:len(section) + nxt.start()]
    return re.findall(r"`([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)`", part)


def _cached(name, ttl, build, refresh=False):
    path = _path("cache", name)
    if not refresh and path.exists() and time.time() - path.stat().st_mtime < ttl:
        value = _read(path)
        if value:
            return value
    try:
        value = build()
    except Problem:
        stale = _read(path)
        if stale:
            return {**stale, "stale": True}
        raise
    _write(path, value)
    return value


def _serving(refresh=False):
    def build():
        serverless, source = [], None
        try:
            key = platform_secret()
            if key:
                data = _http(INFERENCE_MODELS_URL, key=key, timeout=8)
                serverless = [m.get("id") for m in (data or {}).get("data", []) if isinstance(m, dict) and m.get("id")]
                source = "CoreWeave Serverless Inference API"
        except Problem:
            serverless = []
        if not serverless:
            try:
                serverless = _doc_model_ids(_public_text(INFERENCE_MODELS_DOC), "## Generally available models")
                source = "CoreWeave Serverless Inference docs"
            except Problem:
                serverless = []
        try:
            lora = _doc_model_ids(_public_text(LORA_SERVING_DOC), "## Supported base models")
        except Problem:
            lora = []
        if not serverless and not lora:
            raise Problem(502, "The serverless catalogue could not be read.")
        return {"serverless": serverless, "lora_serving": lora, "serverless_source": source,
                "lora_source": "CoreWeave Serverless Inference LoRA docs" if lora else None, "refreshed_at": now_iso()}
    return _cached("serving.json", CATALOGUE_TTL, build, refresh)


def _same_model(trainable, served):
    a, b = trainable.casefold(), served.casefold()
    return a == b or a.startswith(b + "-")


def list_models(refresh=False):
    def build():
        raw = _http(f"{FORGE_URL}/preview/base-models")
        raw = raw.get("data", raw) if isinstance(raw, dict) else raw
        ids = sorted({(m.get("id") if isinstance(m, dict) else m).strip() for m in raw or []
                      if isinstance(m, str) or (isinstance(m, dict) and isinstance(m.get("id"), str))}, key=str.casefold)
        if not ids:
            raise Problem(502, "CoreWeave returned an empty model catalogue.")
        return {"ids": ids, "refreshed_at": now_iso()}
    trainable = _cached("base-models.json", CAPABILITIES_TTL, build, refresh)
    try:
        serving = _serving(refresh)
    except Problem:
        serving = None
    models = []
    for model_id in trainable["ids"]:
        item = {"id": model_id, "provider": "coreweave-forge", "adapter": "lora", "trainable": True}
        if serving:
            item["serverless"] = any(_same_model(model_id, s) for s in serving["serverless"])
            item["serverless_after_tuning"] = any(_same_model(model_id, s) for s in serving["lora_serving"])
        models.append(item)
    return {"models": models, "sources": {
        "trainable": "CoreWeave Forge API (GET /preview/base-models)",
        "serverless": serving and serving.get("serverless_source"),
        "serverless_after_tuning": serving and serving.get("lora_source"),
        "checked_at": serving and serving.get("refreshed_at")}}


def _schema_parameters(markdown, schema_name):
    marker = f"    {schema_name}:"
    start = markdown.find(marker)
    if start < 0:
        return []
    section = markdown[start + len(marker):]
    nxt = re.search(r"\n    [A-Za-z][A-Za-z0-9_]+:\n", section)
    if nxt:
        section = section[:nxt.start()]
    props = section.find("      properties:")
    if props < 0:
        return []
    section = section[props + len("      properties:"):]
    blocks = list(re.finditer(r"(?m)^        ([a-z][a-z0-9_]*):\s*$", section))
    result = []
    for i, match in enumerate(blocks):
        block = section[match.end():blocks[i + 1].start() if i + 1 < len(blocks) else len(section)]
        types = re.findall(r"(?m)^\s+(?:- )?type: (number|integer|boolean|string|object)\s*$", block)
        if not types:
            continue
        item = {"id": match.group(1), "type": types[0]}
        title = re.search(r"(?m)^\s+title: (.+?)\s*$", block)
        item["name"] = title.group(1).strip() if title else match.group(1).replace("_", " ").title()
        for field, pattern in (("min", r"(?:exclusiveMinimum|minimum)"), ("max", r"(?:exclusiveMaximum|maximum)")):
            found = re.search(rf"(?m)^\s+{pattern}: ([^\s]+)\s*$", block)
            if found:
                try:
                    item[field] = float(found.group(1))
                except ValueError:
                    pass
        enum = re.search(r"(?ms)^\s+enum:\s*\n((?:\s+- .+\n?)+)", block)
        if enum:
            item["options"] = [v for v in re.findall(r"(?m)^\s+- ([^\s].*?)\s*$", enum.group(1)) if not v.startswith("type:")]
        default = re.search(r"(?m)^\s+default: (.+?)\s*$", block)
        if default:
            item["default"] = default.group(1).strip()
        if re.search(r"(?m)^\s+const: auto\s*$", block):
            item["allow_auto"] = True
        result.append(item)
    return result


SFT_SPEC = {
    "format": "JSONL, one conversation per line",
    "row": {"messages": [{"role": "system|user|assistant", "content": "text"}]},
    "rules": [
        "messages is a list of {role, content}; roles are system (optional), user, assistant",
        "the last message is the assistant answer; loss is computed on assistant tokens only",
        "at least ~50 examples; hundreds to thousands work best",
        "keep ~2% held out for evaluation, with no prompt overlap",
    ],
    "accepted_shapes": ["{messages:[...]}", "{prompt, completion}", "{instruction, input, output} (Alpaca)"],
    "example": [
        {"messages": [{"role": "user", "content": "Summarise this paragraph in one sentence: ..."},
                      {"role": "assistant", "content": "The paragraph argues that ..."}]},
        {"messages": [{"role": "system", "content": "You answer in a warm, concise tone."},
                      {"role": "user", "content": "Why do leaves change colour?"},
                      {"role": "assistant", "content": "As days shorten, trees stop making chlorophyll ..."}]},
    ],
}
RL_SPEC = {
    "format": "JSONL prompts, each with a reference answer or metadata the reward can check",
    "row": {"messages": [{"role": "user", "content": "task prompt"}], "metadata": {"answer": "reference"}},
    "rules": [
        "messages holds the prompt; a final assistant message (or metadata.answer) is used as the reference, never shown to the model",
        "each step samples several answers per prompt from the current checkpoint and scores them 0 to 1",
        "rewards: exact match, JSON field match, contains, numeric answer, regex, similarity to the reference, or scores sent by your app",
        "tens to hundreds of distinct prompts; prompts where every sample scores the same teach nothing",
    ],
    "example": [
        {"messages": [{"role": "user", "content": "Plan a 3-stop route from A to D using the map below ..."}],
         "metadata": {"optimal_cost": 42}},
    ],
}
METHOD_INFO = {
    "sft": {"name": "SFT", "summary": "Supervised fine-tuning: learns to copy ideal answers from example conversations.",
            "use_when": "You can show the ideal answer for each prompt (style, format, domain knowledge, a voice).",
            "you_need": ["prompts with ideal assistant answers"],
            "spec": SFT_SPEC, "launchable": True, "schema": "SFTTrainingConfig",
            "doc": f"{TRAINING_DOCS}/api-reference/training-jobs/create-sft-training-job.md"},
    "rl": {"name": "RL", "summary": "Reinforcement learning: improves answers using a reward signal.",
           "use_when": "You can score an answer automatically but cannot write the ideal one (math, code that must pass tests, games, tool use).",
           "you_need": ["task prompts with a checkable answer", "a reward: built-in (exact match, JSON field, number, regex, similarity) or scored by your app"],
           "spec": RL_SPEC, "launchable": True,
           "rewards": ["similarity", "exact_match", "json_field", "contains", "numeric", "regex", "external"],
           "schema": "ExperimentalTrainingConfig",
           "doc": f"{TRAINING_DOCS}/api-reference/training-jobs/create-rl-training-job.md"},
}


def recommend_method(body):
    """Suggest a method from what the caller has: sample rows and whether it can score answers."""
    rows = body.get("rows") if isinstance(body.get("rows"), list) else []
    sft_ok = sum(1 for r in rows[:200] if _normalise_row(r, "sft")[1] is None)
    prompts_only = sum(1 for r in rows[:200] if _normalise_row(r, "sft")[1] and _normalise_row(r, "rl")[1] is None)
    has_reward = bool(body.get("has_reward_function"))
    if sft_ok and sft_ok >= prompts_only:
        pick, why = "sft", f"{sft_ok} of {min(len(rows), 200)} sample rows already contain an ideal assistant answer."
    elif prompts_only and has_reward:
        pick, why = "rl", "The rows are prompts without answers and you can score answers, which is what RL needs."
    elif prompts_only:
        pick, why = None, "The rows are prompts without answers: write ideal answers for SFT, or add a reward function for RL."
    else:
        pick, why = None, "No sample rows matched a method's data shape; see each method's data_spec."
    launchable = bool(pick and METHOD_INFO[pick]["launchable"])
    return {"method": pick, "reason": why, "launchable": launchable,
            "note": None if launchable or not pick else METHOD_INFO[pick].get("not_launchable_reason")}


def list_methods(refresh=False):
    def build():
        reference = _public_text(f"{TRAINING_DOCS}/api-reference.md")
        published = {"sft" if e == "sft-training-jobs" else "rl"
                     for e in re.findall(r"POST `?/v1/preview/(sft-training-jobs|training-jobs)`?", reference)}
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            pages = {m: pool.submit(_public_text, METHOD_INFO[m]["doc"]) for m in published}
            params = {m: _schema_parameters(f.result(), METHOD_INFO[m]["schema"]) for m, f in pages.items()}
        if not params:
            raise Problem(502, "CoreWeave did not publish any training methods.")
        return {"params": params, "refreshed_at": now_iso()}
    caps = _cached("methods.json", CAPABILITIES_TTL, build, refresh)
    methods = []
    for method_id in ("sft", "rl"):
        if method_id not in caps["params"]:
            continue
        info = METHOD_INFO[method_id]
        methods.append({"id": method_id, "name": info["name"], "summary": info["summary"], "use_when": info["use_when"],
                        "you_need": info["you_need"], "adapter": "lora",
                        "launchable": info["launchable"], "not_launchable_reason": info.get("not_launchable_reason"),
                        "data_spec": info["spec"], "parameters": caps["params"][method_id],
                        "rewards": info.get("rewards"), "loop_settings": RL_LOOP_SETTINGS if method_id == "rl" else None})
    return {"methods": methods, "source": "CoreWeave Forge API reference", "refreshed_at": caps.get("refreshed_at")}


# ---------------------------------------------------------------- callers

def _caller(actor):
    scope = actor.get("scope")
    if scope not in {"owner", "app"}:
        raise Problem(403, "Only Möbius apps and agents can use the Tuning Engine.")
    app_id = actor.get("app_id")
    own = app_id is None or str(app_id) == APP_ID
    return {"app_id": app_id, "app_slug": actor.get("app_slug") or (APP_SLUG if own else None),
            "is_owner": own, "can_write": actor.get("access", "write") != "read",
            "delegated": bool(actor.get("delegated"))}


def _visible(record, caller):
    return caller["is_owner"] or str(record.get("caller", {}).get("app_id")) == str(caller["app_id"])


def _need_write(caller):
    if not caller["can_write"]:
        raise Problem(403, "This caller has read-only access.")


# ---------------------------------------------------------------- datasets

def _normalise_row(row, method):
    """Return (row, error). Converts common shapes to {messages:[...]}."""
    if not isinstance(row, dict):
        return None, "row is not an object"
    if "messages" not in row:
        if isinstance(row.get("prompt"), str) and isinstance(row.get("completion"), str):
            row = {"messages": [{"role": "user", "content": row["prompt"]}, {"role": "assistant", "content": row["completion"]}]}
        elif isinstance(row.get("instruction"), str) and isinstance(row.get("output"), str):
            user = row["instruction"] + (f"\n\n{row['input']}" if str(row.get("input") or "").strip() else "")
            row = {"messages": [{"role": "user", "content": user}, {"role": "assistant", "content": row["output"]}]}
        else:
            return None, "missing messages (or prompt/completion, or instruction/output)"
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        return None, "messages must be a non-empty list"
    for m in messages:
        if not isinstance(m, dict) or m.get("role") not in ROLES or not isinstance(m.get("content"), str):
            return None, "each message needs a role (system/user/assistant/tool) and string content"
    if not any(m["role"] == "user" for m in messages):
        return None, "no user message"
    if method == "sft":
        if messages[-1]["role"] != "assistant" or not messages[-1]["content"].strip():
            return None, "the last message must be a non-empty assistant answer"
    clean = {"messages": [{"role": m["role"], "content": m["content"]} for m in messages]}
    if method == "rl" and isinstance(row.get("metadata"), dict):
        clean["metadata"] = row["metadata"]
    return clean, None


def dataset_create(body, caller):
    _need_write(caller)
    method = str(body.get("method") or "sft")
    if method not in METHOD_INFO:
        raise Problem(400, "Unknown method. Use one listed by GET v1/methods.")
    ds_id = "ds_" + uuid.uuid4().hex[:12]
    meta = {"id": ds_id, "name": str(body.get("name") or ds_id)[:120], "method": method,
            "source": str(body.get("source") or "")[:300], "license": str(body.get("license") or "")[:80],
            "caller": caller, "rows": 0, "rejected": 0, "errors": [], "sealed": False,
            "created_at": now_iso(), "updated_at": now_iso()}
    _write(_path("datasets", ds_id, "meta.json"), meta)
    return _public_dataset(meta)


def _dataset(ds_id, caller):
    ds_id = _safe_id(ds_id, "ds")
    meta = _read(_path("datasets", ds_id, "meta.json"))
    if not meta or not _visible(meta, caller):
        raise Problem(404, "Dataset not found.")
    return meta


def _public_dataset(meta):
    ready = meta["rows"] >= 10 and meta["method"] in METHOD_INFO
    return {k: meta.get(k) for k in ("id", "name", "method", "source", "license", "rows", "rejected", "sealed",
                                      "created_at", "updated_at")} | {
        "errors": meta.get("errors", [])[:10], "ready": ready and meta.get("sealed", False),
        "hint": None if meta["rows"] >= 10 else "Append at least 10 valid rows, then seal the dataset."}


def dataset_append(ds_id, body, caller):
    _need_write(caller)
    meta = _dataset(ds_id, caller)
    rows = body.get("rows")
    if isinstance(body.get("jsonl"), str):
        rows = []
        for line in body["jsonl"].splitlines():
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    rows.append(None)
    if not isinstance(rows, list) or not rows:
        raise Problem(400, "Send rows: [...] or jsonl: '...'.")
    if len(rows) > MAX_ROWS_PER_APPEND:
        raise Problem(413, f"Send at most {MAX_ROWS_PER_APPEND} rows per call (and under 8 MB).")
    with _Locked(_path("datasets", meta["id"], "meta.json")):
        meta = _read(_path("datasets", meta["id"], "meta.json"))
        if meta.get("sealed"):
            raise Problem(409, "This dataset is sealed; create a new one to change it.")
        if meta["rows"] + len(rows) > MAX_DATASET_ROWS:
            raise Problem(413, "The dataset is too large for the engine.")
        good, errors = [], []
        for i, row in enumerate(rows):
            clean, error = _normalise_row(row, meta["method"])
            if error:
                errors.append({"row": meta["rows"] + meta["rejected"] + i, "error": error})
            else:
                good.append(json.dumps(clean, ensure_ascii=False))
        if good:
            with open(_path("datasets", meta["id"], "rows.jsonl"), "a", encoding="utf-8") as handle:
                handle.write("\n".join(good) + "\n")
        meta.update(rows=meta["rows"] + len(good), rejected=meta["rejected"] + len(errors),
                    errors=(meta.get("errors", []) + errors)[:50], updated_at=now_iso())
        _write(_path("datasets", meta["id"], "meta.json"), meta)
    return {**_public_dataset(meta), "accepted": len(good), "rejected_now": len(errors), "row_errors": errors[:10]}


def dataset_seal(ds_id, caller):
    _need_write(caller)
    meta = _dataset(ds_id, caller)
    with _Locked(_path("datasets", meta["id"], "meta.json")):
        meta = _read(_path("datasets", meta["id"], "meta.json"))
        if meta["rows"] < 10:
            raise Problem(409, "A dataset needs at least 10 valid rows before it can be sealed.")
        digest = hashlib.sha256()
        with open(_path("datasets", meta["id"], "rows.jsonl"), "rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        meta.update(sealed=True, sha256=digest.hexdigest(), updated_at=now_iso())
        _write(_path("datasets", meta["id"], "meta.json"), meta)
    return {**_public_dataset(meta), "sha256": meta["sha256"]}


def dataset_validate(body):
    """Check rows against a method's spec without storing them (for previews)."""
    method = str(body.get("method") or "sft")
    if method not in METHOD_INFO:
        raise Problem(400, "Unknown method.")
    rows = body.get("rows")
    if not isinstance(rows, list) or not rows:
        raise Problem(400, "Send rows: [...].")
    rows = rows[:2000]
    errors, preview = [], []
    for i, row in enumerate(rows):
        clean, error = _normalise_row(row, method)
        if error:
            errors.append({"row": i, "error": error})
        elif len(preview) < 3:
            preview.append(clean)
    return {"method": method, "checked": len(rows), "valid": len(rows) - len(errors), "errors": errors[:20],
            "normalised_preview": preview, "spec": METHOD_INFO[method]["spec"]}


def dataset_list(caller):
    out = []
    root = _path("datasets")
    for meta_path in sorted(root.glob("ds_*/meta.json")) if root.exists() else []:
        meta = _read(meta_path)
        if meta and _visible(meta, caller):
            out.append(_public_dataset(meta))
    return {"datasets": sorted(out, key=lambda d: d["created_at"], reverse=True)[:100]}


# ---------------------------------------------------------------- runs

RUN_FIELDS = ("id", "label", "status", "method", "adapter", "base_model", "dataset_id", "project", "entity",
              "model_name", "model_id", "job_id", "config", "checkpoint_minutes", "train_rows", "created_at",
              "started_at", "ended_at", "updated_at", "error", "stop_requested", "stopped_note", "wandb_url")


def _run_path(run_id):
    return _path("runs", f"{run_id}.json")


def _public_run(record):
    out = {k: record.get(k) for k in RUN_FIELDS}
    out["caller"] = {"app_slug": (record.get("caller") or {}).get("app_slug")}
    chunks = record.get("chunks") or []
    jobs = record.get("chunk_jobs") or []
    done = [j for j in jobs if j.get("status") == "COMPLETED"]
    durations = [(_iso_seconds(j["finished_at"]) or 0) - (_iso_seconds(j["started_at"]) or 0)
                 for j in done if j.get("finished_at") and j.get("started_at")]
    per = sum(durations) / len(durations) if durations else (record.get("checkpoint_minutes") or 5) * 60 + 120
    latest = record.get("kept_step") if record.get("status") == "STOPPED" else len(done)
    trained = sum(c.get("rows", 0) for c in chunks[:len(done)])
    total = sum(c.get("rows", 0) for c in chunks) or record.get("train_rows") or 0
    out.update(
        progress=round(trained / total, 4) if total else 0.0,
        checkpoints_saved=len(done), checkpoints_total=len(chunks), latest_checkpoint_step=latest or None,
        latest_checkpoint_model=(f"wandb-artifact:///{record['entity']}/{record['project']}/{record['model_name']}:step{latest}"
                                 if latest and record.get("entity") else None),
        trained_rows=trained, total_rows=total, seconds_per_checkpoint=round(per),
        eta_seconds=round(max(0, len(chunks) - len(done)) * per) if record.get("status") in ACTIVE and chunks else None,
        events=(record.get("events") or [])[-10:])
    if record.get("method") == "rl":
        batch = record.get("rl_batch") or {}
        hist = record.get("rl_history") or []
        out.update(rl={k: record["rl"].get(k) for k in ("reward", "steps", "prompts_per_step", "samples_per_prompt",
                                                         "temperature", "max_tokens")},
                   rl_phase=("awaiting_rewards" if batch.get("awaiting_rewards") else "sampling" if batch else None),
                   rl_samples=f"{sum(len(g['samples']) for g in batch.get('groups', []))}/{len(batch.get('groups', [])) * record['rl']['samples_per_prompt']}" if batch else None,
                   mean_reward=hist[-1]["reward_mean"] if hist else None, steps_done=len(done),
                   total_rows=None, trained_rows=None, progress=round(len(done) / max(1, len(chunks)), 4))
    return out


def _load_run(run_id, caller):
    run_id = _safe_id(run_id, "run")
    record = _read(_run_path(run_id))
    if not record or not _visible(record, caller):
        raise Problem(404, "Run not found.")
    return record


def _coerce_config(method_id, params):
    allowed = {p["id"]: p for m in list_methods()["methods"] if m["id"] == method_id for p in m["parameters"]}
    config = {}
    for name, value in (params or {}).items():
        if value in (None, ""):
            continue
        spec = allowed.get(name)
        if not spec:
            raise Problem(400, f"'{name}' is not a CoreWeave Forge {method_id.upper()} setting. See GET v1/methods.")
        if spec.get("allow_auto") and str(value) == "auto":
            config[name] = "auto"
            continue
        try:
            if spec["type"] == "integer":
                value = int(value)
            elif spec["type"] == "number":
                value = float(value)
            elif spec["type"] == "boolean":
                value = value if isinstance(value, bool) else str(value).lower() == "true"
            elif spec["type"] == "object" and not isinstance(value, dict):
                raise ValueError
        except (TypeError, ValueError):
            raise Problem(400, f"'{name}' must be {spec['type']}.")
        if spec.get("options") and str(value) not in spec["options"]:
            raise Problem(400, f"'{name}' must be one of {', '.join(spec['options'])}.")
        config[name] = value
    config.setdefault("learning_rate", DEFAULT_LR if method_id == "sft" else RL_DEFAULT_LR)
    if method_id == "sft":
        config.setdefault("metric_logging", {"enabled": True})
    return config


def run_create(body, caller):
    _need_write(caller)
    meta = _dataset(body.get("dataset_id"), caller)
    if not meta.get("sealed"):
        raise Problem(409, "Seal the dataset first (POST v1/datasets/<id>/seal).")
    method = str(body.get("method") or meta["method"])
    if method != meta["method"] and not (method == "rl" and meta["method"] == "sft"):
        raise Problem(400, f"This dataset was shaped for {meta['method'].upper()}, not {method.upper()}.")
    info = METHOD_INFO.get(method)
    if not info or not info["launchable"]:
        raise Problem(400, (info or {}).get("not_launchable_reason") or "Unknown method.")
    model = str(body.get("model") or "")
    if model not in {m["id"] for m in list_models()["models"]}:
        raise Problem(400, "That base model is not in CoreWeave Forge's current catalogue (GET v1/models).")
    key = require_key()
    for other in _all_runs():
        if other.get("status") in ACTIVE and str((other.get("caller") or {}).get("app_id")) == str(caller["app_id"]):
            raise Problem(409, f"This app already has an active run ({other['id']}). Stop it first.")
    minutes = max(1, min(120, int(body.get("checkpoint_minutes") or 5)))
    label = str(body.get("label") or "").strip()[:80]
    project = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(body.get("project") or caller.get("app_slug") or "tuning-engine"))[:60] or "tuning-engine"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    slug = re.sub(r"[^a-z0-9]+", "-", (label or model.split("/")[-1]).lower()).strip("-")[:36] or "run"
    run_id = "run_" + uuid.uuid4().hex[:12]
    record = {"id": run_id, "label": label or f"{slug}-{stamp}", "status": "QUEUED", "method": method, "adapter": "lora",
              "base_model": model, "dataset_id": meta["id"], "dataset_sha256": meta.get("sha256"), "project": project,
              "entity": _entity(key), "model_name": f"te-{slug}-{stamp}", "config": _coerce_config(method, body.get("parameters")),
              "checkpoint_minutes": minutes, "chunk_rows": minutes * EXAMPLES_PER_MINUTE, "train_rows": meta["rows"],
              "caller": caller, "created_at": now_iso(), "updated_at": now_iso(), "events": []}
    if method == "rl":
        record["rl"] = _rl_options(body.get("rl") or {})
        record["chunks"] = [{"index": i, "rows": record["rl"]["prompts_per_step"]} for i in range(record["rl"]["steps"])]
        record["checkpoint_minutes"] = None
    else:
        size = record["chunk_rows"]
        record["chunks"] = [{"index": i, "start": s, "rows": min(size, meta["rows"] - s)} for i, s in enumerate(range(0, meta["rows"], size))]
    _write(_run_path(run_id), record)
    advanced = _advance_locked(run_id, key, budget=5.0)
    return _public_run(advanced or record)


def _all_runs():
    root = _path("runs")
    return [r for r in (_read(p) for p in sorted(root.glob("run_*.json"))) if r] if root.exists() else []


def _note(record, text):
    record.setdefault("events", []).append({"at": now_iso(), "message": text})
    record["events"] = record["events"][-40:]


def _chunk_bytes(record, index):
    chunk = record["chunks"][index]
    out = []
    with open(_path("datasets", record["dataset_id"], "rows.jsonl"), encoding="utf-8") as handle:
        for position, line in enumerate(handle):
            if position >= chunk["start"] + chunk["rows"]:
                break
            if position >= chunk["start"]:
                out.append(line.rstrip("\n"))
    return ("\n".join(out) + "\n").encode()


def _advance(record, key, deadline):
    """Move one run forward as far as the time budget allows. Returns True if the record changed."""
    changed = False
    while time.time() < deadline:
        status = record["status"]
        if status == "QUEUED":
            model = _forge(key, "/preview/models", "POST", {"entity": record["entity"], "project": record["project"],
                                                              "name": record["model_name"], "base_model": record["base_model"]})
            record.update(model_id=model["id"], wandb_run=model.get("run_id") or record["model_name"],
                          entity=model.get("entity") or record["entity"], status="STARTING", chunk_index=0,
                          chunk_jobs=[], started_at=now_iso(),
                          wandb_url=f"https://wandb.ai/{model.get('entity') or record['entity']}/{record['project']}")
            try:
                _ensure_wandb_run(key, record["entity"], record["project"], record["wandb_run"], record["label"])
            except Problem:
                pass
            _note(record, "Created the Forge model.")
            changed = True
        elif status == "STARTING":
            if record.get("stop_requested"):
                record.update(status="STOPPED", ended_at=now_iso())
                return True
            index = record["chunk_index"]
            if record["method"] == "rl":
                return _rl_step(record, key, deadline) or changed
            if deadline - time.time() < 5:
                return changed
            url = (record["chunks"][index].get("url")
                   or _upload_artifact(key, record["entity"], record["project"], record["wandb_run"],
                                       f"{record['model_name']}-data-c{index:03d}", {"train.jsonl": _chunk_bytes(record, index)},
                                       metadata={"format": "jsonl", "num_trajectories": record["chunks"][index]["rows"],
                                                 "chunk": index, "dataset": record["dataset_id"]}))
            record["chunks"][index]["url"] = url
            job = _forge(key, "/preview/sft-training-jobs", "POST",
                         {"model_id": record["model_id"], "training_data_url": url, "config": record["config"]})
            record["chunk_jobs"].append({"index": index, "job_id": job["id"], "started_at": now_iso()})
            record.update(job_id=job["id"], status=job.get("status") or "PENDING")
            _note(record, f"Started checkpoint {index + 1} of {len(record['chunks'])}.")
            return True
        elif status in ("PENDING", "RUNNING"):
            job = _forge(key, f"/preview/training-jobs/{urllib.parse.quote(record['job_id'])}")
            job_status = job.get("status") or status
            current = record["chunk_jobs"][-1]
            if job_status in ("PENDING", "RUNNING"):
                if job_status != status:
                    record["status"] = job_status
                    changed = True
                return changed
            current.update(status=job_status, finished_at=now_iso())
            changed = True
            if record["method"] == "rl" and job_status == "COMPLETED":
                record["rl"]["policy_step"] = record["rl"].get("policy_step", 0) + 1
            if job_status != "COMPLETED":
                record.update(status="FAILED", ended_at=now_iso(),
                              error=job.get("error_message") or f"Checkpoint {current['index'] + 1} {job_status.lower()}.")
                return True
            nxt = current["index"] + 1
            if nxt >= len(record["chunks"]):
                record.update(status="COMPLETED", ended_at=now_iso())
                _note(record, "Training finished.")
                return True
            record.update(chunk_index=nxt, status="STARTING")
            if record["method"] == "rl":
                record["rl_batch"] = None
        else:
            return changed
    return changed


# ---------------------------------------------------------------- RL loop
# One RL step = sample K answers for B prompts from the current checkpoint, score them with the
# reward, send the scored trajectory groups as one Forge training job, wait for it, repeat.
# Sampling is resumable across calls so each service call stays inside the 15 s limit.

RL_DEFAULT_LR = 1e-5
RL_LOOP_SETTINGS = [
    {"id": "steps", "name": "Training steps", "type": "integer", "min": 1, "max": 500, "default": "20"},
    {"id": "prompts_per_step", "name": "Prompts per step", "type": "integer", "min": 1, "max": 64, "default": "8"},
    {"id": "samples_per_prompt", "name": "Samples per prompt", "type": "integer", "min": 2, "max": 16, "default": "4"},
    {"id": "temperature", "name": "Sampling temperature", "type": "number", "min": 0.1, "max": 2, "default": "1.0"},
    {"id": "max_tokens", "name": "Max answer tokens", "type": "integer", "min": 8, "max": 2048, "default": "256"},
]
REWARD_TYPES = {"similarity", "exact_match", "json_field", "contains", "numeric", "regex", "external"}


def _rl_options(raw):
    def num(name, kind, lo, hi, default):
        try:
            value = kind(raw.get(name) if raw.get(name) not in (None, "") else default)
        except (TypeError, ValueError):
            raise Problem(400, f"rl.{name} must be a number.")
        if not lo <= value <= hi:
            raise Problem(400, f"rl.{name} must be between {lo} and {hi}.")
        return value
    reward = raw.get("reward") or {"type": "similarity"}
    if isinstance(reward, str):
        reward = {"type": reward}
    if reward.get("type") not in REWARD_TYPES:
        raise Problem(400, f"rl.reward.type must be one of {', '.join(sorted(REWARD_TYPES))}.")
    if reward["type"] == "json_field" and not str(reward.get("field") or "").strip():
        raise Problem(400, "A json_field reward needs reward.field (the key to compare).")
    if reward["type"] == "regex":
        try:
            re.compile(str(reward.get("pattern") or ""))
        except re.error:
            raise Problem(400, "reward.pattern is not a valid regular expression.")
        if not reward.get("pattern"):
            raise Problem(400, "A regex reward needs reward.pattern.")
    return {"reward": {k: reward.get(k) for k in ("type", "field", "pattern") if reward.get(k) is not None},
            "steps": num("steps", int, 1, 500, 20), "prompts_per_step": num("prompts_per_step", int, 1, 64, 8),
            "samples_per_prompt": num("samples_per_prompt", int, 2, 16, 4),
            "temperature": num("temperature", float, 0.1, 2.0, 1.0), "max_tokens": num("max_tokens", int, 8, 2048, 256),
            "policy_step": 0, "round": 0}


def _norm_text(text):
    return re.sub(r"\s+", " ", str(text or "").strip().lower()).strip(" .!")


def _first_json(text):
    text = str(text or "")
    start = text.find("{")
    while start >= 0:
        depth = 0
        for i in range(start, len(text)):
            depth += text[i] == "{"
            depth -= text[i] == "}"
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    break
        start = text.find("{", start + 1)
    return None


def _last_number(text):
    found = re.findall(r"-?\d+(?:[.,]\d+)?", str(text or "").replace(",", ""))
    try:
        return float(found[-1]) if found else None
    except ValueError:
        return None


def _lcs_f1(a, b):
    x, y = _norm_text(a).split()[:300], _norm_text(b).split()[:300]
    if not x or not y:
        return 0.0
    prev = [0] * (len(y) + 1)
    for i in range(1, len(x) + 1):
        cur = [0] * (len(y) + 1)
        for j in range(1, len(y) + 1):
            cur[j] = prev[j - 1] + 1 if x[i - 1] == y[j - 1] else max(prev[j], cur[j - 1])
        prev = cur
    lcs = prev[-1]
    if not lcs:
        return 0.0
    p, r = lcs / len(x), lcs / len(y)
    return 2 * p * r / (p + r)


def score_answer(reward, answer, reference, metadata=None):
    """Built-in rewards, 0..1. reference is the dataset's answer (or metadata.answer)."""
    kind = reward.get("type")
    if kind == "exact_match":
        ref_json, out_json = _first_json(reference), _first_json(answer)
        if ref_json is not None:
            return 1.0 if out_json == ref_json else 0.0
        return 1.0 if _norm_text(answer) == _norm_text(reference) else 0.0
    if kind == "json_field":
        field = reward["field"]
        out = _first_json(answer)
        want = (metadata or {}).get(field)
        if want is None:
            ref = _first_json(reference)
            want = ref.get(field) if isinstance(ref, dict) else reference
        return 1.0 if isinstance(out, dict) and _norm_text(out.get(field)) == _norm_text(want) else 0.0
    if kind == "contains":
        return 1.0 if reference and _norm_text(reference) in _norm_text(answer) else 0.0
    if kind == "numeric":
        a, b = _last_number(answer), _last_number(reference)
        return 1.0 if a is not None and b is not None and abs(a - b) <= max(1e-6, abs(b) * 1e-3) else 0.0
    if kind == "regex":
        return 1.0 if re.search(reward["pattern"], str(answer or ""), re.S) else 0.0
    return round(_lcs_f1(answer, reference), 4)


def _rl_prompts(record, round_index):
    """Prompt groups for one round: (messages_without_answer, reference, metadata), cycling through the dataset."""
    path = _path("datasets", record["dataset_id"], "rows.jsonl")
    total = record["train_rows"]
    size = record["rl"]["prompts_per_step"]
    wanted = {(round_index * size + i) % total for i in range(size)}
    out = []
    with open(path, encoding="utf-8") as handle:
        for position, line in enumerate(handle):
            if position in wanted:
                row = json.loads(line)
                msgs = row["messages"]
                meta = row.get("metadata") or {}
                reference = meta.get("answer") or meta.get("reference")
                if msgs and msgs[-1]["role"] == "assistant":
                    reference = reference if reference is not None else msgs[-1]["content"]
                    msgs = msgs[:-1]
                out.append({"messages": msgs, "reference": reference if reference is not None else "", "metadata": meta,
                            "samples": []})
                if len(out) == len(wanted):
                    break
    return out


def _rl_sample_one(key, record, messages):
    rl = record["rl"]
    body = {"model": f"wandb-artifact:///{record['entity']}/{record['project']}/{record['model_name']}:step{rl.get('policy_step', 0)}",
            "messages": messages, "temperature": rl["temperature"], "max_tokens": rl["max_tokens"]}
    if not rl.get("no_logprobs"):
        body["logprobs"] = True
    try:
        reply = _http(f"{FORGE_URL}/chat/completions", key=key, method="POST", timeout=10, body=body)
    except Problem as exc:
        if body.get("logprobs") and exc.status == 400:
            rl["no_logprobs"] = True  # this deployment rejects logprobs; train without them
            body.pop("logprobs")
            reply = _http(f"{FORGE_URL}/chat/completions", key=key, method="POST", timeout=10, body=body)
        else:
            raise
    choice = (reply.get("choices") or [None])[0] if isinstance(reply, dict) else None
    if not isinstance(choice, dict):
        raise Problem(502, "Forge returned no answer while sampling.")
    choice.setdefault("index", 0)
    return choice


def _rl_step(record, key, deadline):
    """Fill the current batch (resumable), score it, then submit one RL training job."""
    rl = record["rl"]
    batch = record.get("rl_batch")
    if not batch:
        batch = record["rl_batch"] = {"round": rl.get("round", 0), "groups": _rl_prompts(record, rl.get("round", 0))}
        _note(record, f"Sampling step {record['chunk_index'] + 1}: {len(batch['groups'])} prompts × {rl['samples_per_prompt']} answers.")
    if batch.get("awaiting_rewards"):
        return True  # the calling app scores samples via POST v1/runs/<id>/rewards
    todo = [(gi, g) for gi, g in enumerate(batch["groups"]) for _ in range(rl["samples_per_prompt"] - len(g["samples"]))]
    if todo:
        budget = deadline - time.time() - 1.5
        if budget < 3:
            return True
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futures = {pool.submit(_rl_sample_one, key, record, g["messages"]): gi for gi, g in todo}
            try:
                for future in concurrent.futures.as_completed(futures, timeout=budget):
                    gi = futures[future]
                    try:
                        batch["groups"][gi]["samples"].append({"choice": future.result()})
                    except Problem as exc:
                        record["last_error"] = exc.message
            except concurrent.futures.TimeoutError:
                for f in futures:
                    f.cancel()
        if any(len(g["samples"]) < rl["samples_per_prompt"] for g in batch["groups"]):
            return True
    reward = rl["reward"]
    if reward["type"] == "external" and not batch.get("scored"):
        batch["awaiting_rewards"] = True
        _note(record, "Samples ready; waiting for the calling app to send rewards.")
        return True
    if not batch.get("scored"):
        for g in batch["groups"]:
            for s in g["samples"]:
                answer = ((s["choice"].get("message") or {}).get("content") or "")
                s["reward"] = score_answer(reward, answer, g["reference"], g["metadata"])
        batch["scored"] = True
    rewards = [s["reward"] for g in batch["groups"] for s in g["samples"]]
    informative = [g for g in batch["groups"] if len({s["reward"] for s in g["samples"]}) > 1]
    record.setdefault("rl_history", []).append({
        "step": record["chunk_index"] + 1, "round": batch["round"], "t": time.time(),
        "reward_mean": round(sum(rewards) / len(rewards), 4) if rewards else 0.0,
        "reward_max": max(rewards) if rewards else 0.0, "informative_groups": len(informative)})
    record["rl_history"] = record["rl_history"][-2000:]
    rl["round"] = batch["round"] + 1
    if not informative:
        _note(record, "Every answer in this batch scored the same, so there is nothing to learn from; sampling new prompts.")
        record["rl_batch"] = None
        return True
    no_logprobs = any(s["choice"].get("logprobs") is None for g in informative for s in g["samples"])
    groups = [{"trajectories": [{"messages_and_choices": g["messages"] + [s["choice"]], "reward": s["reward"]}
                                for s in g["samples"]], "metadata": {"prompt_index": i}} for i, g in enumerate(informative)]
    config = dict(record["config"])
    if no_logprobs:
        config["allow_training_without_logprobs"] = True
    try:
        _forge(key, f"/preview/models/{urllib.parse.quote(record['model_id'])}/log", "POST", {"split": "train", "trajectory_groups": groups})
    except Problem:
        pass  # metrics logging is best-effort; training still proceeds
    job = _forge(key, "/preview/training-jobs", "POST", {"model_id": record["model_id"], "trajectory_groups": groups,
                                                         "experimental_config": config})
    record["chunk_jobs"].append({"index": record["chunk_index"], "job_id": job["id"], "started_at": now_iso()})
    record.update(job_id=job["id"], status=job.get("status") or "PENDING", rl_batch=None)
    _note(record, f"Step {record['chunk_index'] + 1}: mean reward {record['rl_history'][-1]['reward_mean']}, training on {len(informative)} groups.")
    return True


def rl_rollouts(run_id, caller):
    """For reward type 'external': the samples waiting for the calling app's scores."""
    record = _load_run(run_id, caller)
    batch = record.get("rl_batch") or {}
    if not batch.get("awaiting_rewards"):
        return {"awaiting_rewards": False, "groups": []}
    return {"awaiting_rewards": True, "step": record["chunk_index"] + 1, "groups": [
        {"prompt": g["messages"], "reference": g["reference"], "metadata": g["metadata"],
         "answers": [((s["choice"].get("message") or {}).get("content") or "") for s in g["samples"]]}
        for g in batch["groups"]]}


def rl_rewards(run_id, body, caller):
    """Accept scores [[r, ...], ...] (one list per group, one number 0..1 per answer) for the waiting batch."""
    _need_write(caller)
    record = _load_run(run_id, caller)
    with _Locked(_run_path(record["id"])):
        record = _read(_run_path(record["id"]))
        batch = record.get("rl_batch") or {}
        if not batch.get("awaiting_rewards"):
            raise Problem(409, "No samples are waiting for rewards.")
        scores = body.get("rewards")
        if not isinstance(scores, list) or len(scores) != len(batch["groups"]):
            raise Problem(400, "Send rewards as one list per group, in the order returned by rollouts.")
        for g, row in zip(batch["groups"], scores):
            if not isinstance(row, list) or len(row) != len(g["samples"]) or not all(_finite(x) for x in row):
                raise Problem(400, "Each group needs one numeric reward per answer.")
            for s, value in zip(g["samples"], row):
                s["reward"] = float(value)
        batch.update(awaiting_rewards=False, scored=True)
        record["updated_at"] = now_iso()
        _write(_run_path(record["id"]), record)
    return {"accepted": True}


def _discard_cleanup(record, key):
    pending = record.get("discard") or {}
    if not pending.get("job_id"):
        return False
    job = _forge(key, f"/preview/training-jobs/{urllib.parse.quote(pending['job_id'])}")
    if job.get("status") in ("PENDING", "RUNNING"):
        return False
    note = "Removed the checkpoint that was in flight when the run was stopped."
    if job.get("status") == "COMPLETED":
        try:
            _forge(key, f"/preview/models/{urllib.parse.quote(record['model_id'])}/checkpoints", "DELETE",
                   {"steps": [int(pending["step"])]})
        except Problem as exc:
            if "last checkpoint" not in exc.message.lower():
                raise
            # Forge never deletes a model's newest checkpoint; the engine hides it instead
            # (checkpoint lists and inference stop at kept_step).
            note = f"Forge keeps its newest checkpoint (step {pending['step']}); the engine hides it and uses step {record.get('kept_step')}."
    record["discard"] = None
    record.pop("last_error", None)
    _note(record, note)
    return True


def _advance_locked(run_id, key, budget=8.0):
    deadline = time.time() + budget
    with _Locked(_run_path(run_id), nonblocking=True) as held:
        if not held:
            return None
        record = _read(_run_path(run_id))
        if not record:
            return None
        try:
            changed = _advance(record, key, deadline) if record["status"] in ACTIVE else False
            if record.get("discard"):
                changed = _discard_cleanup(record, key) or changed
        except Problem as exc:
            record["last_error"] = exc.message
            _note(record, f"Provider error (will retry): {exc.message}")
            if record["status"] == "QUEUED" and exc.status == 400:
                record.update(status="FAILED", error=exc.message, ended_at=now_iso())
            changed = True
        if changed:
            record["updated_at"] = now_iso()
            _write(_run_path(run_id), record)
        return record


def run_get(run_id, caller, refresh=True):
    record = _load_run(run_id, caller)
    if refresh and (record["status"] in ACTIVE or record.get("discard")):
        key = platform_secret()
        if key:
            record = _advance_locked(record["id"], key, budget=6.0) or record
    return {"run": _public_run(record)}


def run_list(caller):
    runs = [_public_run(r) for r in _all_runs() if _visible(r, caller)]
    return {"runs": sorted(runs, key=lambda r: r["created_at"] or "", reverse=True)[:100]}


def run_stop(run_id, caller):
    """Stop now and keep the latest saved checkpoint. Forge cannot cancel a job, so a checkpoint
    still in flight is deleted when it lands; nothing after the stop is kept."""
    _need_write(caller)
    record = _load_run(run_id, caller)
    with _Locked(_run_path(record["id"])):
        record = _read(_run_path(record["id"]))
        if record["status"] not in ACTIVE:
            raise Problem(409, f"The run is already {record['status'].lower()}.")
        jobs = record.get("chunk_jobs") or []
        kept = len([j for j in jobs if j.get("status") == "COMPLETED"])
        in_flight = jobs[-1] if jobs and not jobs[-1].get("status") else None
        record.update(status="STOPPED", stop_requested=True, ended_at=now_iso(), kept_step=kept,
                      stopped_note=f"Stopped; keeping checkpoint step {kept}." if kept else "Stopped before the first checkpoint.")
        if in_flight:
            in_flight["status"] = "DISCARDED"
            record["discard"] = {"job_id": in_flight["job_id"], "step": kept + 1}
        record["rl_batch"] = None
        _note(record, record["stopped_note"])
        record["updated_at"] = now_iso()
        _write(_run_path(record["id"]), record)
    return {"run": _public_run(record)}


def run_rename(run_id, body, caller):
    _need_write(caller)
    label = str(body.get("label") or "").strip()[:80]
    if not label:
        raise Problem(400, "Send a label.")
    record = _load_run(run_id, caller)
    with _Locked(_run_path(record["id"])):
        record = _read(_run_path(record["id"]))
        record.update(label=label, updated_at=now_iso())
        _write(_run_path(record["id"]), record)
    key = platform_secret()
    if key and record.get("wandb_run"):
        try:
            _ensure_wandb_run(key, record["entity"], record["project"], record["wandb_run"], label)
        except Problem:
            pass
    return {"run": _public_run(record)}


def run_checkpoints(run_id, caller):
    record = _load_run(run_id, caller)
    if not record.get("model_id"):
        return {"checkpoints": []}
    key = require_key()
    page = _forge(key, f"/preview/models/{urllib.parse.quote(record['model_id'])}/checkpoints?limit=100")
    kept = record.get("kept_step") if record["status"] == "STOPPED" else None
    out = []
    for item in (page.get("data") if isinstance(page, dict) else None) or []:
        step = item.get("step") if isinstance(item, dict) else None
        if isinstance(step, int) and step > 0 and (kept is None or step <= kept):
            out.append({"step": step, "model": f"wandb-artifact:///{record['entity']}/{record['project']}/{record['model_name']}:step{step}",
                        "created_at": item.get("created_at")})
    return {"checkpoints": sorted(out, key=lambda c: c["step"]), "base_model_lora": (
        f"wandb-artifact:///{record['entity']}/{record['project']}/{record['model_name']}:step0")}


METRICS_CACHE_VERSION = 2


def _metric(data, *names):
    """Forge has logged the same metric under different keys over time (e.g. loss/train, then loss)."""
    for name in names:
        if _finite(data.get(name)):
            return data[name]
    return None


def run_metrics(run_id, caller):
    """Per-step training metrics from Forge job events, cached so each call fetches only new events."""
    record = _load_run(run_id, caller)
    jobs = record.get("chunk_jobs") or []
    if record.get("method") == "rl":
        return {"points": [{"step": h["step"], "reward": h["reward_mean"], "reward_max": h["reward_max"], "t": h["t"]}
                           for h in record.get("rl_history") or []], "status": record["status"], "method": "rl"}
    if not jobs:
        return {"points": [], "status": record["status"]}
    key = require_key()
    cache_path = _path("metrics", f"{record['id']}.json")
    cache = _read(cache_path, {})
    if cache.get("_v") != METRICS_CACHE_VERSION:
        cache = {"_v": METRICS_CACHE_VERSION}
    started = time.time()
    for job in jobs:
        entry = cache.setdefault(job["job_id"], {"after": "", "rows": [], "done": False})
        while not entry["done"] and time.time() - started < 8:
            query = "limit=100" + (f"&after={urllib.parse.quote(entry['after'])}" if entry["after"] else "")
            try:
                page = _forge(key, f"/preview/training-jobs/{urllib.parse.quote(job['job_id'])}/events?{query}")
            except Problem:
                break
            for event in page.get("data") or []:
                data = event.get("data") if isinstance(event, dict) and isinstance(event.get("data"), dict) else {}
                loss = _metric(data, "loss/train", "loss", "train/loss")
                if event.get("type") == "gradient_step" and _finite(loss):
                    entry["rows"].append({"local": data.get("gradient_step") or data.get("step") or len(entry["rows"]) + 1,
                                          "loss": loss, "grad_norm": _metric(data, "loss/grad_norm", "grad_norm"),
                                          "learning_rate": _metric(data, "loss/learning_rate", "learning_rate"),
                                          "t": _iso_seconds(event.get("created_at")) or time.time()})
            if page.get("last_id"):
                entry["after"] = page["last_id"]
            if not page.get("has_more"):
                entry["done"] = job.get("status") in ("COMPLETED", "FAILED", "CANCELLED", "DISCARDED")
                break
    _write(cache_path, cache)
    points, offset = [], 0
    for job in jobs:
        if job.get("status") == "DISCARDED":
            continue
        rows = (cache.get(job["job_id"]) or {}).get("rows") or []
        for row in rows:
            point = {"step": offset + int(row["local"]), "loss": row["loss"], "t": row["t"]}
            for k in ("grad_norm", "learning_rate"):
                if _finite(row.get(k)):
                    point[k] = row[k]
            points.append(point)
        offset += len(rows)
    if len(points) > 1500:
        stride = len(points) / 1500
        points = [points[int(i * stride)] for i in range(1500)] + [points[-1]]
    return {"points": points, "status": record["status"]}


def infer(body, caller):
    """One chat completion against a run's checkpoint (step 0 = base model through the same LoRA path)."""
    record = _load_run(body.get("run_id"), caller)
    if not record.get("model_name"):
        raise Problem(409, "This run has no model yet.")
    step = body.get("step")
    if step in (None, "latest"):
        step = _public_run(record)["latest_checkpoint_step"] or 0
    try:
        step = int(step)
    except (TypeError, ValueError):
        raise Problem(400, "step must be a number or 'latest'.")
    if record["status"] == "STOPPED" and step > (record.get("kept_step") or 0):
        raise Problem(404, "That checkpoint was discarded when the run was stopped.")
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise Problem(400, "Send messages: [{role, content}, ...].")
    for m in messages:
        if not isinstance(m, dict) or m.get("role") not in ROLES or not isinstance(m.get("content"), str):
            raise Problem(400, "Each message needs a role and string content.")
    key = require_key()
    started = time.time()
    reply = _http(f"{FORGE_URL}/chat/completions", key=key, method="POST", timeout=13, body={
        "model": f"wandb-artifact:///{record['entity']}/{record['project']}/{record['model_name']}:step{step}",
        "messages": messages, "temperature": float(body.get("temperature", 0)),
        "max_tokens": max(1, min(2048, int(body.get("max_tokens") or 256)))})
    choice = (reply.get("choices") or [{}])[0] if isinstance(reply, dict) else {}
    return {"run_id": record["id"], "step": step, "output": ((choice.get("message") or {}).get("content") or "").strip(),
            "finish_reason": choice.get("finish_reason"), "usage": reply.get("usage") if isinstance(reply, dict) else None,
            "latency_s": round(time.time() - started, 2)}


def tick():
    """Advance every active run (called each minute by the engine's job)."""
    key = platform_secret()
    if not key:
        return {"advanced": 0}
    deadline = time.time() + BUDGET_S
    count = 0
    for record in _all_runs():
        if time.time() > deadline - 3:
            break
        if record.get("status") in ACTIVE or record.get("discard"):
            if _advance_locked(record["id"], key, budget=max(2.0, deadline - time.time())):
                count += 1
    return {"advanced": count}


# ---------------------------------------------------------------- connection

def status(caller):
    key = platform_secret()
    out = {"engine": "tuning-engine", "api_version": 1, "provider": "coreweave-forge", "connected": False,
           "billing": "Runs bill the W&B account connected to this Tuning Engine.",
           "caller": caller.get("app_slug") or ("owner" if caller["is_owner"] else None)}
    if not key:
        return out
    try:
        out.update(connected=True, entity=_entity(key))
    except Problem as exc:
        out["error"] = exc.message
    return out


def connect(body, caller):
    # The user may connect from the engine itself or from inside a calling app (e.g. TuningKit's
    # Provider step). The key is stored only in the engine's encrypted store; agents/helpers cannot set it.
    if caller.get("delegated") or not caller["can_write"]:
        raise Problem(403, "Agents cannot change the Tuning Engine's W&B connection.")
    value = str(body.get("api_key") or "").strip()
    if not 16 <= len(value) <= 8192:
        raise Problem(400, "Enter a valid W&B API key.")
    try:
        viewer = _gql(value, "query{viewer{entity}}").get("viewer") or {}
    except Problem:
        viewer = {}
    if not viewer.get("entity"):
        raise Problem(400, "W&B did not accept this API key.")
    platform_store_secret(value)
    _write(_path("account.json"), {"fingerprint": hashlib.sha256(value.encode()).hexdigest()[:16], "entity": viewer["entity"],
                                   "connected_via": caller.get("app_slug") or "tuning-engine", "connected_at": now_iso()})
    return {"connected": True, "entity": viewer["entity"]}


# ---------------------------------------------------------------- agent tools

TOOLS = {
    "status": lambda a, c: status(c),
    "list_methods": lambda a, c: list_methods(bool(a.get("refresh"))),
    "list_models": lambda a, c: list_models(bool(a.get("refresh"))),
    "upload_dataset": lambda a, c: _tool_upload(a, c),
    "start_run": lambda a, c: run_create(a, c),
    "run_status": lambda a, c: run_get(a.get("run_id"), c) if a.get("run_id") else run_list(c),
    "stop_run": lambda a, c: run_stop(a.get("run_id"), c),
    "infer": lambda a, c: infer(a, c),
}


def _tool_upload(args, caller):
    """Agents pass a JSONL file path (under /data) or inline rows; the engine creates, fills and seals a dataset."""
    ds = dataset_create({"name": args.get("name"), "method": args.get("method") or "sft",
                         "source": args.get("source") or args.get("path") or "agent"}, caller)
    path = args.get("path")
    if path:
        p = Path(str(path)).resolve()
        if not str(p).startswith("/data/") or not p.is_file():
            raise Problem(400, "path must be an existing JSONL file under /data.")
        batch = []
        with p.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    try:
                        batch.append(json.loads(line))
                    except json.JSONDecodeError:
                        batch.append(None)
                if len(batch) >= MAX_ROWS_PER_APPEND:
                    dataset_append(ds["id"], {"rows": batch}, caller)
                    batch = []
        if batch:
            dataset_append(ds["id"], {"rows": batch}, caller)
    elif isinstance(args.get("rows"), list):
        dataset_append(ds["id"], {"rows": args["rows"]}, caller)
    else:
        raise Problem(400, "Pass path (a JSONL file) or rows.")
    return dataset_seal(ds["id"], caller)


# ---------------------------------------------------------------- router

def route(method, path, query, body, caller):
    parts = [p for p in path.strip("/").split("/") if p]
    if parts and parts[0] == "tools":
        name = parts[1] if len(parts) > 1 else ""
        if name not in TOOLS:
            raise Problem(404, "Unknown tool.")
        return TOOLS[name]((body or {}).get("arguments") or {}, caller)
    if parts[:1] == ["internal"] and parts[1:] == ["tick"]:
        if not caller["is_owner"]:
            raise Problem(403, "Not allowed.")
        return tick()
    if parts[:1] != ["v1"]:
        raise Problem(404, "Use /v1/... routes.")
    raw_refresh = (query or {}).get("refresh", "")
    if isinstance(raw_refresh, list):
        raw_refresh = raw_refresh[0] if raw_refresh else ""
    p, refresh = parts[1:], str(raw_refresh) in ("1", "true")
    if method == "GET" and p == ["status"]:
        return status(caller)
    if method == "POST" and p == ["connect"]:
        return connect(body, caller)
    if method == "GET" and p == ["methods"]:
        return list_methods(refresh)
    if method == "POST" and p == ["methods", "recommend"]:
        return recommend_method(body)
    if method == "GET" and p == ["models"]:
        return list_models(refresh)
    if method == "POST" and p == ["data", "validate"]:
        return dataset_validate(body)
    if p[:1] == ["datasets"]:
        if method == "GET" and len(p) == 1:
            return dataset_list(caller)
        if method == "POST" and len(p) == 1:
            return dataset_create(body, caller)
        if method == "GET" and len(p) == 2:
            return _public_dataset(_dataset(p[1], caller))
        if method == "POST" and len(p) == 3 and p[2] == "rows":
            return dataset_append(p[1], body, caller)
        if method == "POST" and len(p) == 3 and p[2] == "seal":
            return dataset_seal(p[1], caller)
    if p[:1] == ["runs"]:
        if method == "GET" and len(p) == 1:
            return run_list(caller)
        if method == "POST" and len(p) == 1:
            return run_create(body, caller)
        if method == "GET" and len(p) == 2:
            return run_get(p[1], caller)
        if len(p) == 3:
            action = p[2]
            if method == "POST" and action == "stop":
                return run_stop(p[1], caller)
            if method == "POST" and action == "rename":
                return run_rename(p[1], body, caller)
            if method == "GET" and action == "checkpoints":
                return run_checkpoints(p[1], caller)
            if method == "GET" and action == "metrics":
                return run_metrics(p[1], caller)
            if method == "GET" and action == "rollouts":
                return rl_rollouts(p[1], caller)
            if method == "POST" and action == "rewards":
                return rl_rewards(p[1], body, caller)
    if method == "POST" and p == ["infer"]:
        return infer(body, caller)
    raise Problem(404, "Not found. See the Tuning Engine API guide.")


def main(req):
    if not isinstance(req, dict) or req.get("schema") != 1:
        raise Problem(400, "Invalid service request.")
    caller = _caller(req.get("actor") or {})
    body = req.get("body") if isinstance(req.get("body"), dict) else {}
    return route(str(req.get("method") or "GET").upper(), str(req.get("path") or ""), req.get("query") or {}, body, caller)


if __name__ == "__main__":
    try:
        result = main(json.load(sys.stdin))
        print(json.dumps({"status": 200, "body": result}))
    except Problem as exc:
        print(json.dumps({"status": exc.status, "body": {"error": exc.message}}))
    except Exception as exc:
        print(type(exc).__name__, file=sys.stderr)
        print(json.dumps({"status": 500, "body": {"error": "The Tuning Engine hit an unexpected error."}}))

# Tuning Engine — fine-tuning as a service for Möbius apps and agents

Use this when an app or agent needs to fine-tune a model (CoreWeave Forge /
W&B Serverless Training) without writing its own provider code. The engine owns
the W&B connection, the live method/model catalogue, dataset upload, the
checkpointed training loop, Stop, metrics and inference.

## Who pays and who approves
- Runs bill the W&B account connected **in this install's Tuning Engine app**
  (each Möbius user connects their own). The engine never returns the key.
- Approval policy (owner decision 2026-10-11): if an app asks the engine to start
  a run, the engine runs it. **The calling app must ask its user first** (show
  model, data size, method and cost, and require an explicit "Start training").
- Agents using the `tuning-engine_start_run` tool still need the owner's exact
  approval in chat before calling it.

## Calling it from an app
Any app can call `/api/services/tuning-engine/v1/<route>` with its own app token
(`Authorization: Bearer <token>`). Each app only sees its own datasets and runs.
Every call must finish in 15 s; training is advanced by the engine's own
once-a-minute job, so poll status instead of waiting.

| Route | Purpose |
| - | - |
| `GET v1/status` | `{connected, entity, billing}` — show a "connect the Tuning Engine" hint if not connected |
| `POST v1/connect` `{api_key}` | store the user's W&B key in the engine's encrypted store (call from your app's own key form; never store or log the key yourself) |
| `GET v1/methods` | methods with `use_when`, `you_need`, `data_spec` (format, rules, example rows), `parameters` (accepted hyperparameters) and `launchable` |
| `POST v1/methods/recommend` `{rows, has_reward_function?}` | suggests a method from sample rows, with the reason and whether it can launch now |
| `GET v1/models` | trainable base models with `serverless` and `serverless_after_tuning` flags (live; `?refresh=1` to bypass the 6 h cache) |
| `POST v1/data/validate` `{method, rows}` | check up to 2000 rows against the spec without storing them; returns errors and normalised preview |
| `POST v1/datasets` `{name, method, source?, license?}` | create a dataset → `{id}` |
| `POST v1/datasets/<id>/rows` `{rows:[...]}` or `{jsonl:"..."}` | append ≤ 20 000 rows and < 8 MB per call; invalid rows are reported and skipped |
| `POST v1/datasets/<id>/seal` | freeze the dataset (≥ 10 rows) → `{sha256, ready}` |
| `POST v1/runs` | start training (see below) → `{id, status}` |
| `GET v1/runs` / `GET v1/runs/<id>` | status, `progress` 0–1, `latest_checkpoint_step`, `latest_checkpoint_model`, `eta_seconds`, `events` |
| `GET v1/runs/<id>/metrics` | `points: [{step, loss, grad_norm, learning_rate, t}]` (real time, from Forge events) |
| `GET v1/runs/<id>/checkpoints` | saved steps with their `wandb-artifact:///…:stepN` model URIs (`step0` = base model through the LoRA path) |
| `POST v1/runs/<id>/stop` | stop now, keep the latest saved checkpoint (an in-flight checkpoint is discarded) |
| `POST v1/runs/<id>/rename` `{label}` | rename (also the W&B run display name) |
| `POST v1/infer` `{run_id, step|'latest'|0, messages, max_tokens?, temperature?}` | one chat completion from a checkpoint (paid inference) |

Accepted SFT row shapes: `{messages:[{role,content},…]}` ending with the
assistant answer, `{prompt, completion}`, or Alpaca `{instruction, input,
output}` (converted to messages). Keep a held-out split in the calling app;
upload only the training rows.

`POST v1/runs` body:

```json
{"dataset_id": "ds_…", "model": "Qwen/Qwen3.8-27B", "method": "sft",
 "label": "aisha-voice v1", "project": "write-like-me",
 "checkpoint_minutes": 5, "parameters": {"learning_rate": 5e-5}}
```

`parameters` must be fields from `GET v1/methods` (unknown fields are
rejected); `learning_rate` defaults to 5e-5. Training is LoRA-only (rank set by
Forge per model). One active run per calling app. A checkpoint is saved every
`checkpoint_minutes` (each adds ~2 min start-up), so Stop loses at most that
much work.

### RL runs
`POST v1/runs` with `"method": "rl"` and an `rl` object:
`{"reward": {"type": "json_field", "field": "route"}, "steps": 20, "prompts_per_step": 8, "samples_per_prompt": 4, "temperature": 1.0, "max_tokens": 256}`.
Each step samples `samples_per_prompt` answers for `prompts_per_step` prompts from the current checkpoint,
scores them, and submits one Forge RL training job (one policy update). The reference answer is the row's
final assistant message or `metadata.answer`; it is never shown to the model. Built-in rewards:
`similarity` (word-overlap F1, default), `exact_match` (JSON-aware), `json_field`, `contains`, `numeric`, `regex`.
With `"type": "external"` your app scores answers: poll `GET v1/runs/<id>/rollouts`, then send
`POST v1/runs/<id>/rewards {"rewards": [[0.0..1.0 per answer], ...per group]}`. Batches where every answer
scores the same are skipped (no learning signal). `GET v1/runs/<id>/metrics` returns `{step, reward, reward_max}` points.
Sampling is billed at Serverless Inference rates (steps × prompts × samples × tokens).

Status values: `QUEUED → STARTING → PENDING → RUNNING → (STARTING …) →
COMPLETED | STOPPED | FAILED`.

Minimal app flow:

```js
const api = (p, o = {}) => fetch(`/api/services/tuning-engine/v1/${p}`, {...o,
  headers: {Authorization: `Bearer ${token}`, 'Content-Type': 'application/json'}}).then(r => r.json());
const ds = await api('datasets', {method: 'POST', body: JSON.stringify({name: 'samples', method: 'sft'})});
await api(`datasets/${ds.id}/rows`, {method: 'POST', body: JSON.stringify({rows})});
await api(`datasets/${ds.id}/seal`, {method: 'POST'});
// after the user confirms:
const run = await api('runs', {method: 'POST', body: JSON.stringify({dataset_id: ds.id, model, label})});
// poll api(`runs/${run.id}`) every few seconds while active
```

## Agent tools
`tuning-engine_status`, `_list_methods`, `_list_models`, `_upload_dataset`
(path to a JSONL file under /data, or rows), `_start_run` (paid — needs owner
approval), `_run_status`, `_stop_run`, `_infer`.

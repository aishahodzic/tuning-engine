# Tuning Engine

Tuning Engine is the shared training service for Möbius. Connect it once, then
apps such as **TuningKit** and **Write Like Me** can use it to prepare data,
fine-tune models, follow runs, save checkpoints, and test the result.

You do not need to understand the infrastructure to begin. The shortest path
is:

```text
Install Tuning Engine → Connect W&B → Open TuningKit or a use-case app → Train
```

## Start here

### 1. Open Tuning Engine and press Connect

**Connect** opens W&B's authorization page. Sign in or create a W&B account,
copy the API key shown there, return to Tuning Engine, and store it in the
secure key field.

The key is encrypted inside this Tuning Engine installation. Other apps can
ask the engine to train, but they cannot read the key.

You do **not** need a separate CoreWeave account or CoreWeave API key. Training
runs through W&B on CoreWeave Forge. Paid training and inference are billed to
the W&B account connected here.

### 2. Choose where to build the run

Tuning Engine is the machinery, not a step-by-step project builder.

- Use **TuningKit** for a general fine-tuning project: add or create a dataset,
  choose a current model, choose SFT or RL, adjust settings, review, and train.
- Use an app such as **Write Like Me** when the app already understands the
  specific dataset and evaluation for its use case.
- Agents and other apps can call Tuning Engine directly using the contract
  below.

All of these paths use the same connection, model catalogue, datasets, runs,
checkpoints, and metrics owned by Tuning Engine.

### 3. Review before starting

Preparing and validating a dataset does not start training. Before a paid run,
the calling app should show you:

- the dataset and number of training examples;
- the held-out examples that will not be used for training;
- the base model and method;
- the training settings and expected charge; and
- a deliberate **Start training** button.

Models and methods are discovered from the connected service rather than kept
as a fixed list in the client app.

### 4. Follow or stop the run

Return to Tuning Engine to see every run started by your apps. Active runs show
their model, method, source app, progress, estimated time, and saved
checkpoints. **Stop · keep latest checkpoint** ends further training while
preserving the most recent completed checkpoint.

Closing the app does not stop a run. Tuning Engine continues checking it in the
background.

## What each part does

```text
TuningKit ───────┐
                 ├── Tuning Engine ── W&B ── CoreWeave Forge
Use-case apps ───┤
                 └── Agent tools
```

- **Tuning Engine** owns the secure W&B connection, live catalogues, validated
  datasets, training runs, checkpoints, metrics, stopping, and inference.
- **TuningKit** is the general-purpose visual workflow built on the engine.
- **Use-case apps** define the experience for a particular goal and send their
  reviewed training data to the engine.
- **W&B / CoreWeave Forge** runs and bills the managed training and inference.

Each calling app sees only the datasets and runs it created. A Write Like Me
dataset, for example, is not exposed to an unrelated app.

## For agents

Installing Tuning Engine exposes these tools:

- `tuning-engine_status` — check whether W&B is connected;
- `tuning-engine_list_models` — read the current trainable model catalogue;
- `tuning-engine_list_methods` — read current methods, data rules, and settings;
- `tuning-engine_upload_dataset` — validate and seal training rows;
- `tuning-engine_start_run` — start a paid run after exact owner approval;
- `tuning-engine_run_status` — list runs or inspect one run;
- `tuning-engine_stop_run` — stop and retain the latest checkpoint; and
- `tuning-engine_infer` — test a base model or saved checkpoint.

Recommended agent order:

1. Call `status`. If disconnected, ask the owner to open Tuning Engine and use
   **Connect**. Never ask them to paste a live key into ordinary chat.
2. Call `list_methods` and `list_models`; do not rely on a remembered catalogue.
3. Keep a held-out evaluation split outside the uploaded training rows.
4. Call `upload_dataset` and report any rejected rows.
5. Show the exact dataset, model, method, settings, and expected cost.
6. Obtain the owner's approval for that exact paid run.
7. Call `start_run`, then use `run_status`; do not hold a request open.
8. Use saved checkpoints for evaluation and inference.

The complete agent contract—including SFT row formats, RL rewards, accepted
settings, statuses, and checkpoint behavior—is in
[`tuning-engine.md`](tuning-engine.md). Agents should treat that installed skill
as the canonical reference.

## For app builders

An installed app calls the shared service at:

```text
/api/services/tuning-engine/v1/…
```

Authenticate with the calling app's own app token. Tuning Engine isolates all
created datasets and runs by that app identity. Never copy the engine's W&B key
into the calling app.

### Minimal SFT flow

```js
const engine = async (path, options = {}) => {
  const response = await fetch(`/api/services/tuning-engine/v1/${path}`, {
    ...options,
    headers: {
      Authorization: `Bearer ${token}`,
      ...(options.body ? {'Content-Type': 'application/json'} : {}),
    },
  })
  const body = await response.json()
  if (!response.ok) throw new Error(body.error || 'Tuning Engine request failed')
  return body
}

const status = await engine('status')
if (!status.connected) throw new Error('Connect W&B in Tuning Engine first')

const {models} = await engine('models')
const {methods} = await engine('methods')

const dataset = await engine('datasets', {
  method: 'POST',
  body: JSON.stringify({
    name: 'Support examples',
    method: 'sft',
    source: 'Reviewed support conversations',
  }),
})

await engine(`datasets/${dataset.id}/rows`, {
  method: 'POST',
  body: JSON.stringify({rows}),
})

await engine(`datasets/${dataset.id}/seal`, {method: 'POST'})

// Only after the UI shows the exact run and the user presses Start training:
const run = await engine('runs', {
  method: 'POST',
  body: JSON.stringify({
    dataset_id: dataset.id,
    model: selectedModel,
    method: 'sft',
    label: 'Support assistant v1',
    checkpoint_minutes: 5,
  }),
})
```

Poll `GET v1/runs/<id>` while a run is active. Tuning Engine's scheduled worker
advances long-running work independently, so a client should never wait for a
training request to finish.

Common routes:

| Route | Use |
| --- | --- |
| `GET v1/status` | Connection and billing account |
| `GET v1/models` | Current trainable models |
| `GET v1/methods` | Current methods, data rules, and settings |
| `POST v1/data/validate` | Validate rows without storing them |
| `POST v1/datasets` | Create an app-isolated dataset |
| `POST v1/datasets/<id>/rows` | Append training rows |
| `POST v1/datasets/<id>/seal` | Freeze the validated dataset |
| `POST v1/runs` | Start an approved paid run |
| `GET v1/runs/<id>` | Progress, events, ETA, and latest checkpoint |
| `GET v1/runs/<id>/metrics` | Training metrics |
| `GET v1/runs/<id>/checkpoints` | Saved checkpoints |
| `POST v1/runs/<id>/stop` | Stop and retain the latest checkpoint |
| `POST v1/infer` | Paid inference from a base model or checkpoint |

Use [`tuning-engine.md`](tuning-engine.md) for the full request and response
contract. Keep the service, tool definitions, and that skill synchronized.

## Data and safety rules

- Upload only training rows; keep evaluation examples in the calling app.
- A dataset must be validated and sealed before it can train.
- Starting training and inference can incur charges.
- Never start a paid run from an agent without the owner's exact approval.
- Never commit API keys, copied credential stores, private datasets, or run
  output to the app source.

## Development

Validate changes before applying or publishing the app:

```bash
python3 /data/platform/backend/scripts/validate-app.py /data/apps/tuning-engine
```

Tuning Engine is an installable infrastructure app in the App Store. It is not
an Integration: Möbius uses Integrations for remote MCP connections.

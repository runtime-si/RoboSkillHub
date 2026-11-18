---
name: vlm
description: Free-form and yes/no visual question answering against a hosted
  vision-language model (OpenRouter API by default; Vertex AI Gemini, Azure
  OpenAI, a Responses gateway such as Astra, or Gemini's OpenAI-compatible
  endpoint selectable by config). Use when a workflow needs scene descriptions,
  semantic checks ("is the drawer open?"), or LLM-judged verification of a
  camera frame — no GPU required.
license: MIT
compatibility: requires gap>=0.1
metadata:
  category: perception
  tags: [perception, vlm, api]
  hub:
    stage: perceive
    embodiments: [any]
    version: 0.3.0
    provenance:
      code: https://github.com/graph-robots/open-robot-skills/tree/752dfbd96968d4d08379f8111765efbf6457800b/tools/vlm
      upstream_license: MIT
gap:
  # The vlm bundle inherits provider/model/project/region from GAP_LLM_*
  # and GOOGLE_CLOUD_* when its own GAP_VLM_* knobs are unset (see
  # tools.py:_resolve_provider). env_any lists the inheritance sources
  # so a shell configured solely for the agent LLM doesn't trip this
  # readiness gate. `gap check` (and `gap check --probe`) report the
  # RESOLVED config — use them to verify the bundle will dispatch where
  # you expect.
  requires: {env_any: [
    OPENROUTER_API_KEY, GAP_VLM_API_KEY, GAP_VLM_BASE_URL,
    GAP_VLM_PROJECT_ID, GOOGLE_CLOUD_PROJECT,
    GAP_LLM_PROVIDER, AZURE_OPENAI_API_KEY,
    GAP_VLM_ENV_FILE, GAP_VLM_ASTRA_ENV, GAP_VLM_GEMINI_ENV,
    ASTRA_API_KEY, GEMINI_API_KEY, GEMINI_API_KEYS,
  ]}
  serving:
    command: ["python", "-m", "gap_core.rpc.server", "--bundle", "vlm"]
    protocol: stdio-msgpack
  tools:
    - vlm.query: Free-form visual question answering via a hosted VLM.
    - vlm.query_yes_no: Yes/no visual question answering; coerces the model reply to a bool.
---

# vlm

API-backed vision-language Q&A. Zero GPU: every provider is a remote
endpoint. Images are gap-native `uint8 [H, W, 3]` numpy arrays, PNG-encoded
on the wire.

## Providers

Selected by `GAP_VLM_PROVIDER` (default `openrouter`); each tool also accepts
a per-call `provider=` override.

| Provider     | Backend                                             | Config (env)                                              |
|--------------|-----------------------------------------------------|-----------------------------------------------------------|
| `openrouter` | OpenRouter's OpenAI-compatible chat-completions API | `OPENROUTER_API_KEY` (or `GAP_VLM_API_KEY`); `GAP_VLM_MODEL` (default `gemini-3.1-flash-lite-preview`, see `DEFAULT_MODEL` in `tools.py`); set `GAP_VLM_BASE_URL` to point at another OpenAI-compatible server (e.g. a local vLLM) |
| `vertex`     | Vertex AI via google-genai (Gemini models)          | `GAP_VLM_MODEL`, `GAP_VLM_PROJECT_ID`, `GAP_VLM_REGION`   |
| `azure`      | Azure OpenAI v1 endpoint, Responses API (no temperature; works with reasoning deployments) | `AZURE_OPENAI_ENDPOINT` (`https://<resource>.services.ai.azure.com/openai/v1`), `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_DEPLOYMENT` (the deployment name is the model); `GAP_VLM_BASE_URL` / `GAP_VLM_API_KEY` / `GAP_VLM_MODEL` override them |
| `astra`      | a Responses API endpoint behind a gateway (Microsoft's Astra, an Azure API Management front) | from the environment or the env file (below): `ASTRA_API_KEY` (or `OPENAI_API_KEY`, in the file only), `ASTRA_URL` (https; a trailing `/responses` and query string are dropped), `ASTRA_AUTH` (`azure`, the default, sends an `api-key` header; `bearer` a bearer token), `ASTRA_MODEL` (overridden by `model=`) |
| `gemini`     | Gemini through Google's OpenAI-compatible chat endpoint | `GEMINI_API_KEY`, or the first of `GEMINI_API_KEYS` (comma-separated), from the environment or the env file (below); model from `model=` (default `DEFAULT_MODEL`) |

### Keys in an env file on the host

`astra` and `gemini` read what the environment does not carry from a `NAME=value`
file on the host the bundle runs on, so a key never travels with a job, sits in
a repository, or appears in an error message:

| Variable | Names |
|----------|-------|
| `GAP_VLM_ENV_FILE` | the file both providers read; default `~/.config/viper/vlm.env` |
| `GAP_VLM_ASTRA_ENV` | a file for `astra` alone, read instead of `GAP_VLM_ENV_FILE` (an existing gateway `openai.env` works as it is) |
| `GAP_VLM_GEMINI_ENV` | a file for `gemini` alone, read instead of `GAP_VLM_ENV_FILE` |

```bash
# ~/.config/viper/vlm.env, chmod 600; `export` and quotes are allowed
GEMINI_API_KEYS=key-one,key-two
ASTRA_API_KEY=...
ASTRA_URL=https://<gateway>/openai/v1/responses
ASTRA_AUTH=azure
ASTRA_MODEL=<deployment>
```

A name set in the environment wins over the file. `astra` never reads
`OPENAI_API_KEY` from the environment: that is usually a key for
api.openai.com, and it would be sent to the gateway.

`GAP_VLM_MAX_TOKENS` (default 4096) caps the reply for every provider. A thinking model spends
part of it reasoning: at the earlier 1024, 5 of 56 gemini-3-flash replies stopped before their
answer line.

The `vertex` provider lazy-imports google-genai — install the engine's vertex
extra first: `pip install "graph-as-policy[vertex]"`.

## When to use

- Semantic scene checks and checkpoint verification (`vlm.query_yes_no`).
- Free-form scene descriptions or attribute queries (`vlm.query`).
- Prefer `gemini-er.detect` when you need pixel-space bounding boxes, and
  `molmo.point_prompt` when you need a single click point.

## Notes

- `vlm.query_yes_no` coerces with the source-verbatim rule: answer is true
  iff `"yes"` appears in the lowercased reply.
- Requests carry no system prompt and no temperature knob (mirrors the
  original `vlm.v1` proto); the chat and Vertex providers pin `temperature: 0.0`,
  the Responses ones (`azure`, `astra`) send none, as reasoning deployments reject it.

# Anthropic Messages API Backend — `dev.mcpg.backend.llm.anthropic`

> class `backend` · `native` · package `mcpg-plugin-backend-llm-anthropic` · artifact `libmcpg_plugin_backend_llm_anthropic.so` · Apache-2.0

Exposes the Anthropic Messages API as an MCP capability. A binding pins one
model, one system/user prompt pair and one execution policy; the plugin renders
the prompt from the caller's tool arguments, calls `POST {base_url}/messages`,
and returns free-form text or JSON validated against the binding's schema. It
also drives an agentic loop — the model may call other MCP tools you explicitly
allowlist, and their results are fed back until it produces a final answer.
Reach for it when Claude should be reachable as a governed, audited tool
instead of an API key handed out to every client.

## What it does
- Registers one backend entity, `dev.mcpg.backend.anthropic.chat`, whose
  `BackendPlugin::kind()` is `anthropic.chat`; bindings select it with
  `backend.kind: anthropic_chat`.
- Renders `prompt.system` and `prompt.user` as MiniJinja templates over
  `input.*` (the caller's tool arguments) and `meta.*` (`backend_name`,
  `request_id`, `session_id`, `timestamp_iso8601`).
- Emulates structured output with the forced-tool pattern: when the binding
  declares an `output_schema`, the adapter synthesises a tool named `respond`
  whose `input_schema` *is* that schema, then unwraps the model's `tool_use`
  block back into the response.
- Runs a bounded agentic loop over child MCP tools named in `tools.allowed`,
  refusing any call the model invents outside that list before it leaves the
  plugin.
- Streams incremental tokens over SSE and accumulates token usage so the
  stream's terminal event carries a full accounting.
- Accepts image and document (PDF) parts in the user turn; audio parts are
  replaced with a visible placeholder because the Messages API does not take
  them.
- Retries rate-limit, 5xx and network failures with exponential backoff, and
  enforces per-binding token and daily-USD budget caps before spending.
- Declares the `network_outbound` capability — required in every mode, since
  every call is an outbound HTTPS request to the Anthropic API.

Anthropic ships no embedding API, so this plugin has no embedding entity. Pair
the binding with any other MCPG embedding backend — `openai_embedding`,
`gemini_embedding`, or `compat_embedding` against an endpoint of your own.

## Configuration

Load the artifact once from the flat top-level `plugins:` list, then declare one
binding per capability under `mcp.capabilities.tools[]` (or `.prompts[]` /
`.resources[]`) with `backend.kind: anthropic_chat`. Everything else inside the
`backend:` block is the plugin's own spec, forwarded verbatim and validated by
the plugin at boot — an invalid value fails gateway startup, not the first call.

```yaml
plugins:
  - id: dev.mcpg.backend.llm.anthropic
    class: backend
    source:
      oci: ghcr.io/mcpg-dev/source-code/plugins/backend-llm-anthropic:protocol-1

mcp:
  capabilities:
    tools:
      - name: incident.summarize
        description: Summarise an incident report into a structured verdict.
        input_schema:
          type: object
          properties:
            report: { type: string }
          required: [report]
        backend:
          kind: anthropic_chat
          api_key: "${env.ANTHROPIC_API_KEY}"
          model: claude-sonnet-4-5
          prompt:
            system: You are a terse incident analyst. Answer only as JSON.
            user: "Summarise this report:\n{{ input.report }}"
          sampling:
            temperature: 0
            max_completion_tokens: 1024
          response_format:
            mode: json_schema
            on_mismatch: retry_once
          # Read by the plugin when `response_format.mode: json_schema`.
          output_schema:
            type: object
            properties:
              severity: { type: string }
              summary:  { type: string }
            required: [severity, summary]
```

### Provider fields

| Field | Type | Default | Description |
|---|---|---|---|
| `api_key` | string | *(required)* | Sent as the `x-api-key` header. Supply `${env.NAME}` or a `scheme://` URI bound to a `secret_provider` plugin (for example `vault://secret/anthropic#key`); the gateway substitutes the literal value at config load. An empty resolved value is rejected. |
| `base_url` | string | `https://api.anthropic.com/v1` | Override only for a forwarding proxy or a test fixture. The adapter appends `/messages`. |

### Execution fields

Shared with every other MCPG chat binding, so switching providers means changing
`kind` and `model` — not relearning the schema.

| Field | Type | Default | Description |
|---|---|---|---|
| `model` | string | *(required)* | Anthropic model id. |
| `prompt.system` | string | *(required)* | System-prompt template. Must be non-empty after trimming. |
| `prompt.user` | string | *(required)* | User-prompt template. Must be non-empty after trimming. |
| `prompt.image_inputs` | string[] | `[]` | Argument names carrying image content (URL, `data:` URL, raw base64, `mcpg-resource://` URI, or an explicit object). An array value fans out to several parts. |
| `prompt.file_inputs` | string[] | `[]` | Argument names carrying documents; object values may set `mime_type` and `filename`. |
| `timeout_ms` | integer | `60000` | Per-iteration wall-clock budget upstream, retries included. |
| `connect_timeout_ms` | integer | `5000` | TCP connect timeout, kept separate so a slow-but-connected upstream is not killed early. |
| `sampling.temperature` | number | *(unset)* | Passed through when set. |
| `sampling.top_p` | number | *(unset)* | Passed through when set. |
| `sampling.max_completion_tokens` | integer | *(unset)* | Maps to Anthropic's mandatory `max_tokens`; the adapter sends `4096` when unset. |
| `sampling.seed` | integer | *(unset)* | Accepted by the shared schema, then dropped — the Messages API has no `seed` parameter. |
| `response_format.mode` | `json_schema` \| `text` | `json_schema` | `text` wraps the reply as `{"text": "…"}` and skips validation. |
| `response_format.strict` | boolean | `true` | Requests provider-side strictness where available; binding-side validation runs either way. |
| `response_format.on_mismatch` | `error` \| `retry_once` \| `return_raw` | `error` | `return_raw` is legal only with `mode: text`. |
| `tools.allowed` | string[] | `[]` | Names of other bindings in this gateway the model may call. Empty means single-shot. |
| `tools.max_iterations` | integer | `1` when `allowed` is empty, else `5` | Maximum model round-trips. Values above `50` are refused at boot. |
| `tools.tool_choice` | `auto` \| `required` \| `none` | `auto` | Provider-level tool-choice hint. |
| `tools.tool_result_max_bytes` | integer | `16384` | Each child result is truncated to this before re-entering the conversation. |
| `tools.on_iteration_exhausted` | `error` \| `return_partial` | `error` | What happens when the loop runs out of iterations. |
| `retry.max_attempts` | integer | `3` | Attempts per upstream call. |
| `retry.initial_backoff_ms` | integer | `500` | First backoff; must not exceed `max_backoff_ms`. |
| `retry.max_backoff_ms` | integer | `8000` | Backoff ceiling. |
| `retry.retry_on` | list of `rate_limited` \| `server` \| `network` | all three | Failure classes worth retrying. |
| `guardrails.max_output_tokens_per_iteration` | integer | *(unset)* | Hard cap that overrides `sampling.max_completion_tokens`. |
| `cache.enabled` | boolean | `false` | Opt-in response cache. Refused at boot together with a non-empty `tools.allowed`. |
| `cache.ttl_seconds` | integer | `3600000` | Per-entry TTL, in seconds. |
| `budget.tokens_per_call_cap` | integer | `0` (uncapped) | Total input + output tokens across all loop iterations of one call. Checked between iterations, never on the first. |
| `budget.usd_daily_cap` | number | `0` (uncapped) | Aggregate spend for this binding per UTC day, checked before each call. |
| `output_schema` | object | *(unset)* | JSON Schema the reply must satisfy under `mode: json_schema`. Read out of this `backend:` block, not the binding-level field. |

## Agentic tool-calling

`tools.allowed` is an explicit allowlist. A tool call the model invents that is
not on the list never leaves the plugin — the model gets an error string back
and the loop continues, so a hallucinated tool name costs one iteration instead
of failing the call. Allowed names are advertised to the model with a permissive
`{"type": "object"}` argument schema; the child binding's own `input_schema` is
not mirrored into the loop, so a malformed argument surfaces as the child's own
error, fed back to the model as that tool's result.

The gateway refuses a child call that targets the initiating binding itself, and
caps child-invocation depth at 8 regardless of what a binding asks for;
`tools.max_iterations` bounds the per-call horizon on top of that.

```yaml
        backend:
          kind: anthropic_chat
          api_key: "${env.ANTHROPIC_API_KEY}"
          model: claude-sonnet-4-5
          prompt:
            system: You investigate incidents. Use the tools available.
            user: "{{ input.question }}"
          tools:
            allowed: [orders.get, metrics.query]
            max_iterations: 6
            tool_choice: auto
```

## Response envelope

Under `response_format.mode: json_schema` the binding returns the validated
object as-is; a reply that is not valid JSON or does not satisfy the schema
either fails the call or earns one corrective round-trip, per
`response_format.on_mismatch`. Under `mode: text` it returns `{"text": "…"}` and
skips validation entirely. Structured-output guarantees are softer during an agentic loop
than on providers with native strict JSON: `respond` is offered alongside the
operator's tools with `tool_choice: auto` so the loop is not trapped on its
first iteration, and binding-side validation is the real contract.

## Security

- The API key is held in a redacting wrapper — `Debug` renders `***`, so it
  cannot leak through logs or error strings. A key that resolves to an empty
  value is rejected at boot rather than producing unauthenticated calls.
- Prompt templates can reference only `input.*` and `meta.*`. There is no
  filesystem loader, no env-var lookup, and the `debug` filter is removed, so a
  template cannot dump gateway state or exfiltrate the context. Undefined
  variables fail loudly instead of rendering empty.
- Child tool calls made inside the agentic loop carry no caller identity, and
  `cred://` credential threading is unsupported on that path. They are ungated
  unless you turn on `governance.child_invoke.enforce_gates`, which makes each
  child call run the same policy chain, trust floor, CEL `allow_if` gate and
  tool-gate chain a direct `tools/call` runs.
- Budget caps fail closed: exceeding `budget.usd_daily_cap` refuses the call
  before any upstream request is made. Models absent from the bundled rate card
  cannot accumulate cost, so a USD cap is inert for them.

## Observability

Every call opens a span (`llm_anthropic.execute`, or
`llm_anthropic.execute_streaming`) and emits a latency histogram
(`mcpg_llm_anthropic_latency_seconds`) plus a call counter
(`mcpg_llm_anthropic_calls_total`), both labelled with a bounded `outcome`
(`ok`, `rate_limited`, `auth_failed`, `model_not_found`, `server_error`,
`client_error`, `timeout`, `transport`) and `model`. When token usage is known
— the streaming path — it also emits
`mcpg_llm_anthropic_input_tokens_total`,
`mcpg_llm_anthropic_output_tokens_total` and
`mcpg_llm_anthropic_cost_usd_micros_total`.

One audit event lands per call at `dev.mcpg.llm.anthropic.completion` or
`dev.mcpg.llm.anthropic.failure`, carrying binding, model, outcome, duration
and — when known — token counts and cost in micro-USD.

## MCP surfaces & composition

### As a child tool

An `anthropic_chat` binding can itself appear in another chat binding's
`tools.allowed`, which is how you build a cheap-router-in-front-of-expensive-
model pattern with no gateway-side orchestration code.

```yaml
        backend:
          kind: openai_chat
          api_key: "${env.OPENAI_API_KEY}"
          model: gpt-4o-mini
          prompt:
            system: Route the request. Delegate hard reasoning to `deep.analyse`.
            user: "{{ input.question }}"
          tools:
            allowed: [deep.analyse]     # a binding backed by anthropic_chat
```

### Schemas & annotations

The binding-level `input_schema` is what clients see in `tools/list` and what
the gateway validates arguments against. The `output_schema` *inside* the
`backend:` block is what the plugin enforces on the model's reply; declare the
binding-level `output_schema` too when you want clients to see the contract.
Mark bindings that only read as side-effect-free:

```yaml
        annotations: { read_only: true, open_world: true }
```

## Build
Default feature set is OFF (avoids `mcpg_plugin_register` linker
collisions in the workspace build); opt in to the cdylib:

```bash
cargo build -p mcpg-plugin-backend-llm-anthropic --features cdylib-export --release   # → target/release/libmcpg_plugin_backend_llm_anthropic.so
```

## Sign & load (production)
Sign the artifact, pin/verify via the entry's `signature:` block, and honour
revocations. See <https://mcpg.dev/docs/security/plugin-security>.

## See also
- Backend binding reference: <https://mcpg.dev/docs/reference/backends>
- Full gateway config schema: <https://mcpg.dev/docs/reference/configuration>
- Provider-agnostic engine and shared config types: `libs/plugins/backend/llms/shared`
- Sibling providers: `libs/plugins/backend/llms/openai`, `libs/plugins/backend/llms/gemini`, `libs/plugins/backend/llms/compat`

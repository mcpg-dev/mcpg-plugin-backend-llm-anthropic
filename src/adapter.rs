//! Anthropic Messages API adapter.
//!
//! Translates the engine's canonical (OpenAI-shaped)
//! [`NormalizedChatRequest`] / [`NormalizedChatResponse`] to/from
//! Anthropic's `/v1/messages` wire format. Two semantic differences
//! drive this adapter's complexity:
//!
//! 1. **System prompt placement**: Anthropic carries the system prompt
//!    in a top-level `system` field, not as a `Role::System` entry in
//!    `messages`. We extract it during encode.
//!
//! 2. **Structured output**: Anthropic has no `response_format`
//!    primitive. We emulate it via a "forced tool" pattern — when the
//!    operator declares an `output_schema`, we synthesize a tool
//!    named `respond` whose `input_schema` *is* the operator's
//!    output schema, and bias the model toward calling it. The model
//!    then emits a `tool_use` block whose `input` is the structured
//!    response. The adapter unwraps this and presents it to the
//!    engine as ordinary content text (a JSON string), so the engine
//!    doesn't need to know about the trick.
//!
//! ## Mixed mode (response_schema + child tools)
//!
//! When both an output schema and child tools are configured, both
//! `respond` and the operator's tools appear in the request. The
//! model picks `respond` to terminate the loop or any operator tool
//! to continue it. We use `tool_choice: { type: "auto" }` rather than
//! forcing `respond` — forcing it would trap the agentic loop on
//! the first iteration. This means structured-output guarantees are
//! *softer* during agentic loops on Anthropic than on OpenAI's
//! `strict: true` json_schema mode; binding-side validation is the
//! contract.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use futures::StreamExt;
use reqwest::Client;
use reqwest::header::{CONTENT_TYPE, HeaderMap, HeaderName, HeaderValue};
use serde_json::{Value, json};

use mcpg_backend_llm_shared::normalized::{
    ContentPart, FileSource, FinishReason, ImageSource, Message, MessageContent,
    NormalizedChatRequest, NormalizedChatResponse, Role, TokenUsage, ToolCall, ToolChoiceWire,
    ToolDef,
};
use mcpg_backend_llm_shared::{
    ChatProviderAdapter, NormalizedStreamEvent, ProviderError, StreamEventReceiver,
};

/// Synthetic tool name used to carry structured output. Operator
/// tool names that collide with this would shadow the trick — we
/// validate at adapter construction (rare in practice).
pub(crate) const RESPOND_TOOL_NAME: &str = "respond";

/// Anthropic API version sent on every request. Bumped only when
/// breaking changes land upstream.
const ANTHROPIC_VERSION: &str = "2023-06-01";

pub struct AnthropicAdapter {
    client: Client,
    /// Includes the API version path segment (`/v1`). The `messages`
    /// suffix is appended at request time.
    base_url: String,
    api_key: Arc<str>,
}

impl AnthropicAdapter {
    pub fn new(
        base_url: impl Into<String>,
        api_key: impl Into<String>,
        connect_timeout: Duration,
    ) -> Result<Self, ProviderError> {
        let client = Client::builder()
            .user_agent("mcpg-plugin-backend-llm-anthropic/1.0")
            .connect_timeout(connect_timeout)
            .build()
            .map_err(|e| ProviderError::Network {
                message: format!("build http client: {e}"),
            })?;
        let base_url = base_url.into();
        if base_url.is_empty() {
            return Err(ProviderError::BadRequest {
                message: "anthropic base_url is empty".into(),
            });
        }
        Ok(Self {
            client,
            base_url: base_url.trim_end_matches('/').to_owned(),
            api_key: Arc::from(api_key.into()),
        })
    }

    fn endpoint_url(&self) -> String {
        format!("{}/messages", self.base_url)
    }

    fn build_headers(&self) -> Result<HeaderMap, ProviderError> {
        let mut h = HeaderMap::new();
        h.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        h.insert(
            HeaderName::from_static("anthropic-version"),
            HeaderValue::from_static(ANTHROPIC_VERSION),
        );
        let key_value = HeaderValue::from_str(self.api_key.as_ref()).map_err(|_| {
            ProviderError::BadRequest {
                message: "api_key contains characters not allowed in HTTP headers".into(),
            }
        })?;
        h.insert(HeaderName::from_static("x-api-key"), key_value);
        Ok(h)
    }
}

#[async_trait]
impl ChatProviderAdapter for AnthropicAdapter {
    fn label(&self) -> &'static str {
        "anthropic"
    }

    async fn chat_completion(
        &self,
        request: &NormalizedChatRequest,
        timeout: Duration,
    ) -> Result<NormalizedChatResponse, ProviderError> {
        let body = encode_request(request)?;
        let headers = self.build_headers()?;
        let url = self.endpoint_url();

        let resp = self
            .client
            .post(&url)
            .headers(headers)
            .timeout(timeout)
            .json(&body)
            .send()
            .await
            .map_err(map_reqwest_error)?;

        let status = resp.status();
        let bytes = resp.bytes().await.map_err(|e| ProviderError::Network {
            message: format!("read response body: {e}"),
        })?;

        if !status.is_success() {
            return Err(map_status_error(status, &bytes));
        }

        let value: Value =
            serde_json::from_slice(&bytes).map_err(|e| ProviderError::Malformed {
                message: format!("response is not JSON: {e}"),
            })?;

        decode_response(&value, request.response_schema.is_some())
    }

    async fn stream_chat_completion(
        &self,
        request: &NormalizedChatRequest,
        timeout: Duration,
    ) -> Result<StreamEventReceiver, ProviderError> {
        let mut body = encode_request(request)?;
        if let Value::Object(obj) = &mut body {
            obj.insert("stream".into(), Value::Bool(true));
        }
        let headers = self.build_headers()?;
        let url = self.endpoint_url();

        let resp = self
            .client
            .post(&url)
            .headers(headers)
            .timeout(timeout)
            .json(&body)
            .send()
            .await
            .map_err(map_reqwest_error)?;

        let status = resp.status();
        if !status.is_success() {
            let bytes = resp.bytes().await.unwrap_or_default();
            return Err(map_status_error(status, &bytes));
        }

        let (tx, rx) =
            tokio::sync::mpsc::channel::<Result<NormalizedStreamEvent, ProviderError>>(32);
        let mut byte_stream = resp.bytes_stream();
        let expects_respond = request.response_schema.is_some();

        tokio::spawn(async move {
            // Anthropic emits one SSE event per content-block boundary
            // and per delta. We track block-by-block state in a small
            // map keyed on `index` (the block index) — Anthropic
            // guarantees ordering within an index.
            #[derive(Default)]
            struct BlockState {
                kind: Option<String>, // "text" | "tool_use"
                tool_id: Option<String>,
                tool_name: Option<String>,
                json_buffer: String,
            }
            let mut blocks: std::collections::BTreeMap<u64, BlockState> =
                std::collections::BTreeMap::new();

            let mut buffer: Vec<u8> = Vec::new();
            let mut final_finish: Option<FinishReason> = None;
            let mut input_tokens: u32 = 0;
            let mut output_tokens: u32 = 0;
            let mut respond_text_emitted = false;
            let mut stream_error: Option<ProviderError> = None;

            'outer: while let Some(chunk_res) = byte_stream.next().await {
                let chunk = match chunk_res {
                    Ok(c) => c,
                    Err(e) => {
                        stream_error = Some(ProviderError::Network {
                            message: format!("read sse chunk: {e}"),
                        });
                        break 'outer;
                    }
                };
                buffer.extend_from_slice(&chunk);

                while let Some(boundary) = find_event_boundary(&buffer) {
                    let event_bytes = buffer.drain(..boundary).collect::<Vec<u8>>();
                    let _ = strip_boundary_prefix(&mut buffer);

                    let event_text = match std::str::from_utf8(&event_bytes) {
                        Ok(s) => s,
                        Err(_) => continue,
                    };

                    // Each Anthropic event has one `data:` line we
                    // care about. The `event: <type>` line is
                    // redundant (the JSON also has `type`), so we
                    // parse data only.
                    let data = match event_text
                        .lines()
                        .find_map(|line| line.strip_prefix("data:"))
                        .map(|s| s.trim_start_matches(' '))
                    {
                        Some(d) => d,
                        None => continue,
                    };
                    let event: Value = match serde_json::from_str(data) {
                        Ok(v) => v,
                        Err(_) => continue,
                    };
                    let etype = event.get("type").and_then(|v| v.as_str()).unwrap_or("");

                    match etype {
                        "message_start" => {
                            if let Some(usage) = event.get("message").and_then(|m| m.get("usage")) {
                                input_tokens = usage
                                    .get("input_tokens")
                                    .and_then(|v| v.as_u64())
                                    .unwrap_or(0)
                                    as u32;
                            }
                        }
                        "content_block_start" => {
                            let idx = event.get("index").and_then(|v| v.as_u64()).unwrap_or(0);
                            let block = event.get("content_block");
                            let entry = blocks.entry(idx).or_default();
                            if let Some(b) = block {
                                let kind = b
                                    .get("type")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("")
                                    .to_owned();
                                entry.kind = Some(kind);
                                if let Some(id) = b.get("id").and_then(|v| v.as_str()) {
                                    entry.tool_id = Some(id.to_owned());
                                }
                                if let Some(name) = b.get("name").and_then(|v| v.as_str()) {
                                    entry.tool_name = Some(name.to_owned());
                                }
                            }
                        }
                        "content_block_delta" => {
                            let idx = event.get("index").and_then(|v| v.as_u64()).unwrap_or(0);
                            let delta = match event.get("delta") {
                                Some(d) => d,
                                None => continue,
                            };
                            let dtype = delta.get("type").and_then(|v| v.as_str()).unwrap_or("");
                            let entry = blocks.entry(idx).or_default();
                            match dtype {
                                "text_delta" => {
                                    if let Some(t) = delta.get("text").and_then(|v| v.as_str())
                                        && tx
                                            .send(Ok(NormalizedStreamEvent::TextDelta(
                                                t.to_owned(),
                                            )))
                                            .await
                                            .is_err()
                                    {
                                        return;
                                    }
                                }
                                "input_json_delta" => {
                                    if let Some(p) =
                                        delta.get("partial_json").and_then(|v| v.as_str())
                                    {
                                        entry.json_buffer.push_str(p);
                                    }
                                }
                                _ => {}
                            }
                        }
                        "content_block_stop" => {
                            let idx = event.get("index").and_then(|v| v.as_u64()).unwrap_or(0);
                            if let Some(state) = blocks.remove(&idx)
                                && state.kind.as_deref() == Some("tool_use")
                            {
                                let name = state.tool_name.unwrap_or_default();
                                let id = state.tool_id.unwrap_or_default();
                                let args: Value = serde_json::from_str(&state.json_buffer)
                                    .unwrap_or(Value::Object(Default::default()));

                                if expects_respond && name == RESPOND_TOOL_NAME {
                                    // Forced-respond completion: the
                                    // synthetic tool's input IS the
                                    // structured response. Emit it as a
                                    // text delta so the engine reads it
                                    // as content. (Not perfectly
                                    // streaming-friendly — the whole
                                    // JSON arrives as one delta — but
                                    // semantically correct.)
                                    let serialized =
                                        serde_json::to_string(&args).unwrap_or_default();
                                    respond_text_emitted = true;
                                    if tx
                                        .send(Ok(NormalizedStreamEvent::TextDelta(serialized)))
                                        .await
                                        .is_err()
                                    {
                                        return;
                                    }
                                } else if tx
                                    .send(Ok(NormalizedStreamEvent::ToolCallReady(ToolCall {
                                        id,
                                        name,
                                        arguments: args,
                                    })))
                                    .await
                                    .is_err()
                                {
                                    return;
                                }
                            }
                        }
                        "message_delta" => {
                            if let Some(stop) = event
                                .get("delta")
                                .and_then(|d| d.get("stop_reason"))
                                .and_then(|v| v.as_str())
                            {
                                final_finish = Some(if respond_text_emitted {
                                    FinishReason::Stop
                                } else {
                                    match stop {
                                        "end_turn" | "stop_sequence" => FinishReason::Stop,
                                        "tool_use" => FinishReason::ToolCalls,
                                        "max_tokens" => FinishReason::Length,
                                        _ => FinishReason::Other,
                                    }
                                });
                            }
                            if let Some(out) = event
                                .get("usage")
                                .and_then(|u| u.get("output_tokens"))
                                .and_then(|v| v.as_u64())
                            {
                                output_tokens = out as u32;
                            }
                        }
                        "message_stop" => {
                            break 'outer;
                        }
                        _ => {}
                    }
                }
            }

            if let Some(err) = stream_error {
                let _ = tx.send(Err(err)).await;
                return;
            }

            let _ = tx
                .send(Ok(NormalizedStreamEvent::Finish {
                    reason: final_finish.unwrap_or(FinishReason::Other),
                    usage: TokenUsage {
                        input_tokens,
                        output_tokens,
                        cached_input_tokens: 0,
                    },
                }))
                .await;
        });

        Ok(rx)
    }
}

/// SSE event boundary helpers — duplicated locally because the OpenAI
/// adapter has its own copy and the trade-off for a third tiny crate
/// or a shared module wasn't worth the import cost at the time. If a
/// fourth provider lands, lift these into `crate::sse`.
fn find_event_boundary(buf: &[u8]) -> Option<usize> {
    if let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
        return Some(pos);
    }
    buf.windows(2).position(|w| w == b"\n\n")
}
fn strip_boundary_prefix(buf: &mut Vec<u8>) -> usize {
    if buf.starts_with(b"\r\n\r\n") {
        buf.drain(..4);
        4
    } else if buf.starts_with(b"\n\n") {
        buf.drain(..2);
        2
    } else {
        0
    }
}

// ---------------------------------------------------------------------------
// Request encoding
// ---------------------------------------------------------------------------

fn encode_request(req: &NormalizedChatRequest) -> Result<Value, ProviderError> {
    // Anthropic requires `max_tokens` on every call. We default to a
    // sane upper bound when the operator hasn't set one, so the
    // adapter doesn't 400-out by omission.
    const DEFAULT_MAX_TOKENS: u32 = 4_096;

    let (system, messages) = encode_messages(&req.messages)?;

    let mut body = serde_json::Map::new();
    body.insert("model".into(), Value::String(req.model.clone()));
    body.insert(
        "max_tokens".into(),
        json!(req.max_completion_tokens.unwrap_or(DEFAULT_MAX_TOKENS)),
    );
    if let Some(s) = system {
        body.insert("system".into(), Value::String(s));
    }
    body.insert("messages".into(), Value::Array(messages));

    // Tools: combine operator tools with the synthetic `respond` tool
    // whenever a response_schema was supplied.
    let mut tools: Vec<Value> = req
        .tools
        .iter()
        .map(encode_tool_def)
        .collect::<Result<Vec<_>, _>>()?;
    if let Some(schema) = &req.response_schema {
        // Refuse if an operator tool happens to be named `respond`
        // — the synthetic would shadow it ambiguously.
        if tools
            .iter()
            .any(|t| t.get("name").and_then(|v| v.as_str()) == Some(RESPOND_TOOL_NAME))
        {
            return Err(ProviderError::BadRequest {
                message: format!(
                    "operator tool named '{RESPOND_TOOL_NAME}' conflicts with the Anthropic structured-output synthetic; rename it"
                ),
            });
        }
        tools.push(json!({
            "name": RESPOND_TOOL_NAME,
            "description": "Return your final structured response. Call this tool exactly once when you have the answer.",
            "input_schema": schema,
        }));
    }
    if !tools.is_empty() {
        body.insert("tools".into(), Value::Array(tools));
    }

    // tool_choice strategy:
    // - response_schema set + no operator tools → force `respond`
    //   (single-shot structured generation; nothing else to do).
    // - response_schema set + operator tools present → `auto` so the
    //   loop can run; the model still has `respond` available to
    //   terminate.
    // - no response_schema, operator tools present → translate the
    //   engine's choice (auto / required / none).
    let tool_choice = match (&req.response_schema, req.tools.is_empty(), req.tool_choice) {
        (Some(_), true, _) => Some(json!({"type": "tool", "name": RESPOND_TOOL_NAME})),
        (Some(_), false, _) => Some(json!({"type": "auto"})),
        (None, true, _) => None,
        (None, false, ToolChoiceWire::Auto) => Some(json!({"type": "auto"})),
        // Anthropic uses `any` (not `required`) to mean "must call a
        // tool but model chooses which".
        (None, false, ToolChoiceWire::Required) => Some(json!({"type": "any"})),
        (None, false, ToolChoiceWire::None) => Some(json!({"type": "none"})),
    };
    if let Some(tc) = tool_choice {
        body.insert("tool_choice".into(), tc);
    }

    if let Some(t) = req.temperature {
        body.insert("temperature".into(), json!(t));
    }
    if let Some(t) = req.top_p {
        body.insert("top_p".into(), json!(t));
    }
    // Anthropic ignores `seed` (no support); we drop it silently.

    Ok(Value::Object(body))
}

fn encode_tool_def(t: &ToolDef) -> Result<Value, ProviderError> {
    if t.name == RESPOND_TOOL_NAME {
        return Err(ProviderError::BadRequest {
            message: format!(
                "operator tool named '{RESPOND_TOOL_NAME}' conflicts with the Anthropic structured-output synthetic"
            ),
        });
    }
    Ok(json!({
        "name": t.name,
        "description": t.description,
        "input_schema": t.parameters,
    }))
}

/// Returns (system, messages). System messages are pulled out of the
/// flat `messages` list because Anthropic carries them separately.
/// We only honor the FIRST system message; subsequent ones are
/// concatenated with `\n\n`.
fn encode_messages(messages: &[Message]) -> Result<(Option<String>, Vec<Value>), ProviderError> {
    let mut system_parts: Vec<String> = Vec::new();
    let mut out: Vec<Value> = Vec::new();

    for m in messages {
        match m.role {
            Role::System => {
                system_parts.push(m.content.as_text());
            }
            Role::User => match &m.content {
                MessageContent::Text(s) => {
                    out.push(json!({"role": "user", "content": s}));
                }
                MessageContent::Parts(parts) => {
                    out.push(json!({"role": "user", "content": encode_user_parts(parts)}));
                }
            },
            Role::Assistant => {
                let mut content_blocks: Vec<Value> = Vec::new();
                let text = m.content.as_text();
                if !text.is_empty() {
                    content_blocks.push(json!({"type": "text", "text": text}));
                }
                for tc in &m.tool_calls {
                    content_blocks.push(json!({
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": tc.arguments,
                    }));
                }
                if content_blocks.is_empty() {
                    // Anthropic rejects an assistant message with an
                    // empty content array. The engine can produce
                    // this in degenerate paths; emit a placeholder
                    // text block so the request is well-formed.
                    content_blocks.push(json!({"type": "text", "text": ""}));
                }
                out.push(json!({"role": "assistant", "content": content_blocks}));
            }
            Role::Tool => {
                // Tool results in Anthropic ride on a `user` message
                // as `tool_result` blocks.
                let tool_use_id = m.tool_call_id.clone().unwrap_or_default();
                out.push(json!({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": m.content.as_text(),
                    }]
                }));
            }
        }
    }

    let system = if system_parts.is_empty() {
        None
    } else {
        Some(system_parts.join("\n\n"))
    };
    Ok((system, out))
}

/// Encode a [`MessageContent::Parts`] user message as Anthropic's
/// content-block array. Anthropic accepts:
///
/// - `{type: "text", text: "..."}` for prose.
/// - `{type: "image", source: {type: "base64", media_type, data}}`
///   — base64 only; URLs must be fetched and converted upstream.
///   Anthropic's `source.type` also supports `"url"` since
///   2025-04, but we keep the base64 path for compatibility with
///   older models.
/// - `{type: "document", source: {...}, title?, citations?}` for
///   PDFs and other files. Only `application/pdf` is officially
///   supported on the messages API; other MIME types are passed
///   through and may be rejected upstream.
///
/// `mcpg-resource://` sources should have been replaced upstream
/// in the engine via `BackendHost::fetch_content`. If one slips
/// through, we substitute a placeholder text block so the model
/// gets a notice rather than a malformed request.
fn encode_user_parts(parts: &[ContentPart]) -> Value {
    let mut out: Vec<Value> = Vec::with_capacity(parts.len());
    for p in parts {
        match p {
            ContentPart::Text(s) => {
                out.push(json!({"type": "text", "text": s}));
            }
            ContentPart::Image(img) => match &img.source {
                ImageSource::Url(u) => {
                    // Anthropic 2025-04+ accepts `source.type: "url"`;
                    // the engine still prefers to pre-resolve URLs to
                    // base64 in `multimodal` so older models work.
                    // When a URL slips through here we forward it
                    // verbatim using the newer source shape.
                    out.push(json!({
                        "type": "image",
                        "source": { "type": "url", "url": u }
                    }));
                }
                ImageSource::Base64 { mime_type, data } => {
                    out.push(json!({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": mime_type,
                            "data": data,
                        }
                    }));
                }
                ImageSource::McpResource(uri) => {
                    out.push(json!({
                        "type": "text",
                        "text": format!("[unresolved image resource: {uri}]"),
                    }));
                }
            },
            ContentPart::Audio(au) => {
                // Anthropic's chat-completions surface does not
                // accept audio inputs (use Whisper sibling models /
                // separate `audio.transcribe` bindings). Emit a text
                // placeholder so operators see why their audio
                // doesn't reach the model.
                let _ = au;
                out.push(json!({
                    "type": "text",
                    "text": "[audio input not supported by anthropic.chat]",
                }));
            }
            ContentPart::File(f) => match &f.source {
                FileSource::Url(_) | FileSource::Base64 { .. } => {
                    let source = match &f.source {
                        FileSource::Url(u) => json!({"type": "url", "url": u}),
                        FileSource::Base64 { data } => json!({
                            "type": "base64",
                            "media_type": f.mime_type,
                            "data": data,
                        }),
                        FileSource::McpResource(_) => unreachable!(),
                    };
                    let mut block = serde_json::Map::new();
                    block.insert("type".into(), Value::String("document".into()));
                    block.insert("source".into(), source);
                    if let Some(name) = f.filename.as_ref() {
                        block.insert("title".into(), Value::String(name.clone()));
                    }
                    out.push(Value::Object(block));
                }
                FileSource::McpResource(uri) => {
                    out.push(json!({
                        "type": "text",
                        "text": format!("[unresolved file resource: {uri}]"),
                    }));
                }
            },
        }
    }
    Value::Array(out)
}

// ---------------------------------------------------------------------------
// Response decoding
// ---------------------------------------------------------------------------

fn decode_response(
    value: &Value,
    expects_structured_via_respond: bool,
) -> Result<NormalizedChatResponse, ProviderError> {
    let content = value
        .get("content")
        .and_then(|c| c.as_array())
        .ok_or_else(|| ProviderError::Malformed {
            message: "response has no `content` array".into(),
        })?;

    let mut text_parts: Vec<String> = Vec::new();
    let mut tool_calls: Vec<ToolCall> = Vec::new();
    let mut respond_input: Option<Value> = None;

    for block in content {
        let block_type = block
            .get("type")
            .and_then(|v| v.as_str())
            .unwrap_or_default();
        match block_type {
            "text" => {
                if let Some(t) = block.get("text").and_then(|v| v.as_str()) {
                    text_parts.push(t.to_owned());
                }
            }
            "tool_use" => {
                let id = block
                    .get("id")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_owned();
                let name = block
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_owned();
                let input = block.get("input").cloned().unwrap_or(Value::Null);

                if expects_structured_via_respond && name == RESPOND_TOOL_NAME {
                    // The synthetic tool's `input` IS the structured
                    // response. Hold onto it; we'll surface it as
                    // content text below.
                    respond_input = Some(input);
                } else {
                    tool_calls.push(ToolCall {
                        id,
                        name,
                        arguments: input,
                    });
                }
            }
            // Future block types (image, redacted_thinking) are
            // silently dropped; adapter-level support for them is a
            // later addition.
            _ => {}
        }
    }

    let stop_reason = value.get("stop_reason").and_then(|v| v.as_str());
    let finish_reason = match (stop_reason, respond_input.is_some(), tool_calls.is_empty()) {
        // Forced-respond terminal — finish as Stop regardless of what
        // Anthropic called it (it's `tool_use` from their POV).
        (_, true, _) => FinishReason::Stop,
        (Some("tool_use"), _, false) => FinishReason::ToolCalls,
        (Some("max_tokens"), _, _) => FinishReason::Length,
        (Some("end_turn"), _, _) => FinishReason::Stop,
        (Some("stop_sequence"), _, _) => FinishReason::Stop,
        _ => FinishReason::Other,
    };

    let content_text = if let Some(input) = respond_input {
        // Serialize the synthetic respond's input as a JSON string so
        // the engine reads it as ordinary structured-text content.
        serde_json::to_string(&input).map_err(|e| ProviderError::Malformed {
            message: format!("serialize respond.input: {e}"),
        })?
    } else {
        text_parts.join("")
    };

    let usage = decode_usage(value.get("usage"));

    Ok(NormalizedChatResponse {
        content: content_text,
        tool_calls,
        finish_reason,
        usage,
    })
}

fn decode_usage(value: Option<&Value>) -> TokenUsage {
    let Some(u) = value else {
        return TokenUsage::default();
    };
    let input = u.get("input_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
    let output = u.get("output_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
    let cached = u
        .get("cache_read_input_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as u32;
    TokenUsage {
        input_tokens: input,
        output_tokens: output,
        cached_input_tokens: cached,
    }
}

// ---------------------------------------------------------------------------
// Error mapping
// ---------------------------------------------------------------------------

fn map_status_error(status: reqwest::StatusCode, body: &[u8]) -> ProviderError {
    let message = body_excerpt(body);
    let code = status.as_u16();
    if code == 429 {
        return ProviderError::RateLimited { message };
    }
    if code == 401 || code == 403 {
        return ProviderError::AuthFailed { message };
    }
    if code == 400 {
        // Anthropic returns 400 with `type: "invalid_request_error"`
        // and a code like `prompt_too_long` for context-window issues.
        if message.contains("prompt_too_long")
            || message.contains("max_tokens")
            || message.contains("context")
        {
            return ProviderError::ContextLimit { message };
        }
        return ProviderError::BadRequest { message };
    }
    if code == 413 {
        return ProviderError::ContextLimit { message };
    }
    if (500..600).contains(&code) {
        return ProviderError::Server { message };
    }
    // Anthropic's `overloaded_error` (529) is documented as retryable.
    if code == 529 {
        return ProviderError::Server { message };
    }
    ProviderError::Server { message }
}

fn map_reqwest_error(err: reqwest::Error) -> ProviderError {
    if err.is_timeout() {
        return ProviderError::Network {
            message: format!("timeout: {err}"),
        };
    }
    if err.is_connect() {
        return ProviderError::Network {
            message: format!("connect failed: {err}"),
        };
    }
    if err.is_request() || err.is_body() || err.is_decode() {
        return ProviderError::Network {
            message: format!("transport: {err}"),
        };
    }
    ProviderError::Network {
        message: err.to_string(),
    }
}

fn body_excerpt(body: &[u8]) -> String {
    const MAX: usize = 512;
    let s = String::from_utf8_lossy(body);
    if s.len() <= MAX {
        s.into_owned()
    } else {
        format!("{}…[truncated]", &s[..MAX])
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use mcpg_backend_llm_shared::normalized::{Message, ToolDef};

    fn user_only(content: &str) -> Vec<Message> {
        vec![Message::user(content)]
    }

    fn sys_user(sys: &str, user: &str) -> Vec<Message> {
        vec![Message::system(sys), Message::user(user)]
    }

    fn baseline_request(messages: Vec<Message>) -> NormalizedChatRequest {
        NormalizedChatRequest {
            model: "claude-3-5-sonnet".into(),
            messages,
            response_schema: None,
            strict_response: false,
            tools: vec![],
            tool_choice: ToolChoiceWire::Auto,
            temperature: None,
            top_p: None,
            max_completion_tokens: None,
            seed: None,
        }
    }

    #[test]
    fn encode_pulls_system_prompt_to_top_level() {
        let r = baseline_request(sys_user("you are helpful", "hi"));
        let body = encode_request(&r).unwrap();
        assert_eq!(body["system"], json!("you are helpful"));
        assert_eq!(body["messages"][0]["role"], json!("user"));
        assert_eq!(body["messages"][0]["content"], json!("hi"));
        assert!(body.get("tools").is_none());
        assert!(body.get("tool_choice").is_none());
    }

    #[test]
    fn encode_concatenates_multiple_system_messages() {
        let mut msgs = sys_user("first", "hi");
        msgs.insert(1, Message::system("second"));
        let r = baseline_request(msgs);
        let body = encode_request(&r).unwrap();
        assert_eq!(body["system"], json!("first\n\nsecond"));
    }

    #[test]
    fn encode_emits_default_max_tokens_when_unset() {
        let r = baseline_request(user_only("hi"));
        let body = encode_request(&r).unwrap();
        assert_eq!(body["max_tokens"], json!(4096));
    }

    #[test]
    fn encode_with_response_schema_no_tools_forces_respond() {
        let mut r = baseline_request(user_only("hi"));
        r.response_schema = Some(json!({"type": "object"}));
        let body = encode_request(&r).unwrap();
        let tools = body["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["name"], json!(RESPOND_TOOL_NAME));
        assert_eq!(body["tool_choice"]["type"], json!("tool"));
        assert_eq!(body["tool_choice"]["name"], json!(RESPOND_TOOL_NAME));
    }

    #[test]
    fn encode_with_response_schema_and_tools_uses_auto() {
        let mut r = baseline_request(user_only("hi"));
        r.response_schema = Some(json!({"type": "object"}));
        r.tools = vec![ToolDef {
            name: "child".into(),
            description: "x".into(),
            parameters: json!({"type": "object"}),
        }];
        let body = encode_request(&r).unwrap();
        let tools = body["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 2, "operator tool + synthetic respond");
        assert_eq!(body["tool_choice"]["type"], json!("auto"));
    }

    #[test]
    fn encode_required_choice_maps_to_any() {
        let mut r = baseline_request(user_only("hi"));
        r.tools = vec![ToolDef {
            name: "x".into(),
            description: "x".into(),
            parameters: json!({"type": "object"}),
        }];
        r.tool_choice = ToolChoiceWire::Required;
        let body = encode_request(&r).unwrap();
        assert_eq!(body["tool_choice"]["type"], json!("any"));
    }

    #[test]
    fn encode_rejects_operator_tool_named_respond() {
        let mut r = baseline_request(user_only("hi"));
        r.response_schema = Some(json!({"type": "object"}));
        r.tools = vec![ToolDef {
            name: RESPOND_TOOL_NAME.into(),
            description: "x".into(),
            parameters: json!({"type": "object"}),
        }];
        let err = encode_request(&r).unwrap_err();
        assert!(matches!(err, ProviderError::BadRequest { .. }));
    }

    #[test]
    fn encode_assistant_with_tool_calls_emits_content_blocks() {
        let mut msgs = sys_user("sys", "user");
        msgs.push(Message::assistant_tool_calls(vec![ToolCall {
            id: "t_1".into(),
            name: "fetch".into(),
            arguments: json!({"q": "x"}),
        }]));
        msgs.push(Message::tool_result("t_1", "{\"data\":42}"));
        let r = baseline_request(msgs);
        let body = encode_request(&r).unwrap();
        let messages = body["messages"].as_array().unwrap();
        // [user, assistant_with_tool_use, user_with_tool_result]
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[1]["role"], json!("assistant"));
        assert_eq!(messages[1]["content"][0]["type"], json!("tool_use"));
        assert_eq!(messages[2]["role"], json!("user"));
        assert_eq!(messages[2]["content"][0]["type"], json!("tool_result"));
        assert_eq!(messages[2]["content"][0]["tool_use_id"], json!("t_1"));
    }

    #[test]
    fn decode_text_only() {
        let raw = json!({
            "content": [{"type": "text", "text": "hello"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 5, "output_tokens": 1}
        });
        let r = decode_response(&raw, false).unwrap();
        assert_eq!(r.content, "hello");
        assert!(r.tool_calls.is_empty());
        assert_eq!(r.finish_reason, FinishReason::Stop);
        assert_eq!(r.usage.input_tokens, 5);
        assert_eq!(r.usage.output_tokens, 1);
    }

    #[test]
    fn decode_tool_use_only() {
        let raw = json!({
            "content": [{
                "type": "tool_use",
                "id": "toolu_1",
                "name": "fetch",
                "input": {"q": "hello"}
            }],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 10, "output_tokens": 5}
        });
        let r = decode_response(&raw, false).unwrap();
        assert!(r.content.is_empty());
        assert_eq!(r.tool_calls.len(), 1);
        assert_eq!(r.tool_calls[0].id, "toolu_1");
        assert_eq!(r.tool_calls[0].name, "fetch");
        assert_eq!(r.tool_calls[0].arguments, json!({"q": "hello"}));
        assert_eq!(r.finish_reason, FinishReason::ToolCalls);
    }

    #[test]
    fn decode_forced_respond_extracts_input_as_json_string() {
        let raw = json!({
            "content": [{
                "type": "tool_use",
                "id": "toolu_r",
                "name": RESPOND_TOOL_NAME,
                "input": {"answer": "ok", "score": 0.9}
            }],
            "stop_reason": "tool_use"
        });
        let r = decode_response(&raw, true).unwrap();
        assert_eq!(r.finish_reason, FinishReason::Stop, "respond is terminal");
        assert!(
            r.tool_calls.is_empty(),
            "respond is not exposed as a tool_call"
        );
        // Content is the JSON-encoded input — exactly what the engine
        // expects to validate against the operator's output_schema.
        let parsed: Value = serde_json::from_str(&r.content).unwrap();
        assert_eq!(parsed, json!({"answer": "ok", "score": 0.9}));
    }

    #[test]
    fn decode_respond_with_other_tool_use_blocks_extracts_only_respond() {
        // Defensive: if Anthropic emits both a real tool call AND
        // respond in one turn, we treat respond as terminal and let
        // the model's other tool call go un-dispatched. This is rare
        // (tool_choice usually pins one path) but the binding must
        // not crash.
        let raw = json!({
            "content": [
                {"type": "tool_use", "id": "tu1", "name": "fetch", "input": {}},
                {"type": "tool_use", "id": "tu2", "name": RESPOND_TOOL_NAME, "input": {"a": 1}},
            ],
            "stop_reason": "tool_use"
        });
        let r = decode_response(&raw, true).unwrap();
        assert_eq!(r.finish_reason, FinishReason::Stop);
        assert_eq!(r.tool_calls.len(), 1, "non-respond tool_use survives");
        assert_eq!(r.tool_calls[0].name, "fetch");
        let parsed: Value = serde_json::from_str(&r.content).unwrap();
        assert_eq!(parsed, json!({"a": 1}));
    }

    #[test]
    fn decode_max_tokens_stop_reason_maps_to_length() {
        let raw = json!({
            "content": [{"type": "text", "text": "partial"}],
            "stop_reason": "max_tokens"
        });
        let r = decode_response(&raw, false).unwrap();
        assert_eq!(r.finish_reason, FinishReason::Length);
    }

    #[test]
    fn decode_rejects_response_without_content_array() {
        let raw = json!({"stop_reason": "end_turn"});
        let err = decode_response(&raw, false).unwrap_err();
        assert!(matches!(err, ProviderError::Malformed { .. }));
    }

    #[test]
    fn decode_unknown_block_types_are_skipped() {
        let raw = json!({
            "content": [
                {"type": "text", "text": "hi"},
                {"type": "redacted_thinking", "data": "..."}
            ],
            "stop_reason": "end_turn"
        });
        let r = decode_response(&raw, false).unwrap();
        assert_eq!(r.content, "hi");
    }

    #[test]
    fn map_status_overloaded_529_is_server() {
        let e = map_status_error(reqwest::StatusCode::from_u16(529).unwrap(), b"overloaded");
        assert!(matches!(e, ProviderError::Server { .. }));
        assert!(e.is_retryable());
    }

    #[test]
    fn map_status_400_with_prompt_too_long_is_context_limit() {
        // Anthropic returns 400 with `code: "prompt_too_long"` for
        // context-window overflow.
        let e = map_status_error(
            reqwest::StatusCode::from_u16(400).unwrap(),
            b"{\"error\":{\"type\":\"invalid_request_error\",\"code\":\"prompt_too_long\"}}",
        );
        assert!(matches!(e, ProviderError::ContextLimit { .. }));
    }

    #[test]
    fn encode_drops_seed() {
        let mut r = baseline_request(user_only("hi"));
        r.seed = Some(42);
        let body = encode_request(&r).unwrap();
        assert!(
            body.get("seed").is_none(),
            "Anthropic does not support seed"
        );
    }

    #[test]
    fn encode_passes_through_temperature_and_top_p() {
        let mut r = baseline_request(user_only("hi"));
        r.temperature = Some(0.5);
        r.top_p = Some(0.5);
        let body = encode_request(&r).unwrap();
        // Use f64 ≈ for sampling values — f32→f64 round-trips through
        // serde produce small precision wobble (0.3_f32 → 0.30000…).
        // 0.5 round-trips exactly which keeps this assertion clean
        // without giving up the test's intent.
        assert_eq!(body["temperature"].as_f64().unwrap(), 0.5);
        assert_eq!(body["top_p"].as_f64().unwrap(), 0.5);
    }

    // ----- Multimodal user-parts encoding -----

    #[test]
    fn encode_user_image_base64_emits_anthropic_image_block() {
        use mcpg_backend_llm_shared::normalized::{ContentPart, ImageContent, ImageSource};
        let parts = vec![
            ContentPart::Text("what's here".into()),
            ContentPart::Image(ImageContent {
                source: ImageSource::Base64 {
                    mime_type: "image/png".into(),
                    data: "aGVsbG8=".into(),
                },
                detail: None,
            }),
        ];
        let (system, msgs) = encode_messages(&[Message::user_parts(parts)]).unwrap();
        assert!(system.is_none());
        let arr = msgs[0]["content"].as_array().unwrap();
        assert_eq!(arr[0]["type"], "text");
        assert_eq!(arr[1]["type"], "image");
        assert_eq!(arr[1]["source"]["type"], "base64");
        assert_eq!(arr[1]["source"]["media_type"], "image/png");
        assert_eq!(arr[1]["source"]["data"], "aGVsbG8=");
    }

    #[test]
    fn encode_user_image_url_emits_url_source() {
        use mcpg_backend_llm_shared::normalized::{ContentPart, ImageContent, ImageSource};
        let parts = vec![ContentPart::Image(ImageContent {
            source: ImageSource::Url("https://ex.com/a.png".into()),
            detail: None,
        })];
        let (_, msgs) = encode_messages(&[Message::user_parts(parts)]).unwrap();
        assert_eq!(msgs[0]["content"][0]["source"]["type"], "url");
        assert_eq!(
            msgs[0]["content"][0]["source"]["url"],
            "https://ex.com/a.png"
        );
    }

    #[test]
    fn encode_user_pdf_emits_document_block() {
        use mcpg_backend_llm_shared::normalized::{ContentPart, FileContent, FileSource};
        let parts = vec![ContentPart::File(FileContent {
            source: FileSource::Base64 {
                data: "JVBERi0=".into(),
            },
            mime_type: "application/pdf".into(),
            filename: Some("report.pdf".into()),
        })];
        let (_, msgs) = encode_messages(&[Message::user_parts(parts)]).unwrap();
        let block = &msgs[0]["content"][0];
        assert_eq!(block["type"], "document");
        assert_eq!(block["source"]["type"], "base64");
        assert_eq!(block["source"]["media_type"], "application/pdf");
        assert_eq!(block["title"], "report.pdf");
    }
}

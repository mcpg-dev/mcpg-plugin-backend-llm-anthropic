//! `BackendPlugin` impl for Anthropic chat completions.
//!
//! Delegates execution to the shared
//! [`mcpg_backend_llm_shared::ChatEngine`]; only the wire-format
//! adapter ([`crate::adapter::AnthropicAdapter`]) is provider-
//! specific.
//!
//! The plugin stores an installed
//! [`HostHandle`] in a `OnceLock` and routes per-call observability
//! through the unified host surface: per-execute span at
//! `llm_anthropic.execute`, latency histogram + call counter with
//! bounded `outcome` + `model` labels, and an audit event per
//! upstream call (`dev.mcpg.llm.anthropic.{completion,failure}`)
//! with model + token + cost details when known. Streaming
//! completions emit the triad at stream-end with aggregated token
//! counts. The pre-existing internal `tracing` + `metrics::*` calls
//! remain wired in both modes.

use std::collections::BTreeMap;
use std::sync::{Arc, OnceLock, RwLock};

use mcpg_backend_llm_shared::chat_config::ResponseFormatMode;
use mcpg_backend_llm_shared::template::Templates;
use mcpg_backend_llm_shared::{
    ChatEngine, ChatProviderAdapter, ProviderError, build_child_tool_defs, compile_validator,
    resolve_api_key,
};
use mcpg_plugin_protocol::{
    BackendChunk, BackendChunkStream, BackendError, BackendHost, BackendPlugin, BackendRequest,
    BackendResponse, PluginManifest, async_trait, firstparty_manifest, types::PluginIdentity,
};
use mcpg_plugin_sdk::HostHandle;
use serde_json::Value;
use tracing::{Instrument, debug, info_span, warn};

use crate::adapter::AnthropicAdapter;
use crate::config::AnthropicChatSpec;
use crate::host_handle_obs::{
    UsageSnapshot, emit_chat_observability, open_span, open_streaming_span,
};

pub struct AnthropicChatPlugin {
    manifest: PluginManifest,
    engines: Arc<RwLock<BTreeMap<String, Arc<ChatEngine>>>>,
    /// Unified host-observability handle.
    /// `OnceLock` because the boot path installs it exactly once
    /// after construction. Test paths that build the plugin without
    /// wiring a host leave the slot empty; the triad short-circuits.
    host_handle: OnceLock<HostHandle>,
}

impl std::fmt::Debug for AnthropicChatPlugin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AnthropicChatPlugin").finish()
    }
}

impl Default for AnthropicChatPlugin {
    fn default() -> Self {
        Self::new()
    }
}

impl AnthropicChatPlugin {
    pub fn new() -> Self {
        Self {
            manifest: firstparty_manifest! {
                id: "dev.mcpg.backend.anthropic.chat",
                name: "Anthropic Messages API",
                class: Backend,
            },
            engines: Arc::new(RwLock::new(BTreeMap::new())),
            host_handle: OnceLock::new(),
        }
    }

    #[doc(hidden)]
    pub fn registered_profile_count(&self) -> usize {
        self.engines.read().unwrap().len()
    }

    /// Install the unified [`HostHandle`]
    /// surface for per-call observability. Idempotent; a second call
    /// returns `false`.
    pub fn set_host_handle(&self, host: HostHandle) -> bool {
        self.host_handle.set(host).is_ok()
    }

    fn host_handle(&self) -> Option<&HostHandle> {
        self.host_handle.get()
    }
}

#[async_trait]
impl BackendPlugin for AnthropicChatPlugin {
    fn manifest(&self) -> &PluginManifest {
        &self.manifest
    }

    fn kind(&self) -> &str {
        "anthropic.chat"
    }

    async fn register_profile(
        &self,
        backend_name: &str,
        spec: &Value,
        host: Arc<dyn BackendHost>,
    ) -> Result<(), BackendError> {
        let parsed: AnthropicChatSpec =
            serde_json::from_value(spec.clone()).map_err(|e| BackendError::InvalidSpec {
                message: format!("anthropic_chat spec: {e}"),
            })?;
        parsed.validate().map_err(|e| BackendError::InvalidSpec {
            message: e.to_string(),
        })?;

        let api_key = resolve_api_key(&parsed.api_key)?;
        let base_url = parsed.resolved_base_url().to_owned();
        let connect_timeout = parsed.chat.connect_timeout();

        let adapter = AnthropicAdapter::new(base_url, api_key, connect_timeout).map_err(
            |e: ProviderError| BackendError::InvalidSpec {
                message: format!("build anthropic adapter: {e}"),
            },
        )?;
        let adapter: Arc<dyn ChatProviderAdapter> = Arc::new(adapter);

        let templates = Templates::compile(&parsed.chat.prompt.system, &parsed.chat.prompt.user)
            .map_err(|e| BackendError::InvalidSpec {
                message: format!("template: {e}"),
            })?;

        let (validator, raw_output_schema) = if matches!(
            parsed.chat.response_format.mode,
            ResponseFormatMode::JsonSchema
        ) {
            let schema_value = spec.get("output_schema").cloned();
            if let Some(schema) = schema_value {
                let v = compile_validator(&schema).map_err(|e| BackendError::InvalidSpec {
                    message: e.to_string(),
                })?;
                (Some(v), Some(schema))
            } else {
                (None, None)
            }
        } else {
            (None, None)
        };

        let child_tool_defs = build_child_tool_defs(&parsed.chat.tools, |_name| None);

        let engine = ChatEngine {
            backend_name: backend_name.to_owned(),
            adapter,
            templates,
            validator,
            raw_output_schema,
            spec: parsed.chat,
            host,
            child_tool_defs,
            child_tool_validators: Vec::new(),
        };

        self.engines
            .write()
            .map_err(|_| BackendError::InvalidSpec {
                message: "engine map poisoned".into(),
            })?
            .insert(backend_name.to_owned(), Arc::new(engine));

        Ok(())
    }

    async fn execute(
        &self,
        backend_name: &str,
        request: BackendRequest,
    ) -> Result<BackendResponse, BackendError> {
        let engine = self
            .engines
            .read()
            .map_err(|_| BackendError::InvalidSpec {
                message: "engine map poisoned".into(),
            })?
            .get(backend_name)
            .cloned()
            .ok_or_else(|| BackendError::ProfileNotFound {
                backend_name: backend_name.to_owned(),
            })?;

        let args: Value = if request.payload.is_empty() {
            serde_json::json!({})
        } else {
            serde_json::from_slice(&request.payload).map_err(|e| BackendError::InvalidSpec {
                message: format!("execute payload was not JSON: {e}"),
            })?
        };

        let model = engine.spec.model.clone();
        let identity = request.identity.clone();
        let request_id = request.request_id.clone();

        // Wrap the engine call in a per-plugin span
        // so traces from this plugin attribute to
        // `dev.mcpg.backend.llm.anthropic` and operators can route
        // via `plugins[].observability.traces`. Engine
        // adapter+model labels also flow into the existing
        // metrics.
        let internal_span = info_span!(
            "anthropic_chat_execute",
            plugin_id = "dev.mcpg.backend.llm.anthropic",
            binding = %backend_name,
            model = %model,
        );

        // Open the host span BEFORE engine
        // dispatch so the span window covers the full upstream
        // call. Dropped explicitly AFTER the triad emission so
        // span_end lands last.
        let host_span = open_span(self.host_handle(), backend_name, &model);

        let started = std::time::Instant::now();
        let result = engine
            .execute(&args, &request.request_id, request.session_id.as_deref())
            .instrument(internal_span)
            .await;
        let elapsed = started.elapsed();

        metrics::counter!(
            "mcpg_llm_calls_total",
            "binding" => backend_name.to_owned(),
            "provider" => engine.adapter.label().to_string(),
            "model" => model.clone(),
            "status" => if result.is_ok() { "ok" } else { "error" },
        )
        .increment(1);
        metrics::histogram!(
            "mcpg_llm_call_overall_seconds",
            "binding" => backend_name.to_owned(),
            "provider" => engine.adapter.label().to_string(),
            "model" => model.clone(),
        )
        .record(elapsed.as_secs_f64());

        match &result {
            Ok(_) => debug!(
                binding = %backend_name,
                model = %model,
                elapsed_ms = %elapsed.as_millis(),
                "anthropic chat call succeeded"
            ),
            Err(e) => warn!(
                binding = %backend_name,
                model = %model,
                error = %e,
                "anthropic chat call failed"
            ),
        }

        emit_chat_observability(
            self.host_handle(),
            backend_name,
            &model,
            &request_id,
            identity.as_ref(),
            elapsed,
            result.as_ref().map(|_| ()),
            None,
        )
        .await;
        drop(host_span);

        let value = result?;
        let payload = serde_json::to_vec(&value).map_err(|e| BackendError::Transport {
            message: format!("serialize response: {e}"),
        })?;
        Ok(BackendResponse {
            payload,
            truncated: false,
        })
    }

    async fn execute_streaming(
        &self,
        backend_name: &str,
        request: BackendRequest,
    ) -> Result<BackendChunkStream, BackendError> {
        let engine = self
            .engines
            .read()
            .map_err(|_| BackendError::InvalidSpec {
                message: "engine map poisoned".into(),
            })?
            .get(backend_name)
            .cloned()
            .ok_or_else(|| BackendError::ProfileNotFound {
                backend_name: backend_name.to_owned(),
            })?;

        let args: Value = if request.payload.is_empty() {
            serde_json::json!({})
        } else {
            serde_json::from_slice(&request.payload).map_err(|e| BackendError::InvalidSpec {
                message: format!("execute payload was not JSON: {e}"),
            })?
        };

        Ok(wrap_streaming(
            self.host_handle().cloned(),
            backend_name.to_owned(),
            engine.spec.model.clone(),
            request.identity.clone(),
            request.request_id.clone(),
            engine.execute_streaming(args, request.request_id, request.session_id),
        ))
    }
}

/// Wrap the engine's streaming chunk stream so
/// we can observe end-of-stream + accumulate `BackendChunk::Usage`
/// tokens, then emit the host triad once when the stream terminates
/// (either via `Done` or an `Err` item).
fn wrap_streaming(
    host: Option<HostHandle>,
    backend_name: String,
    model: String,
    identity: Option<PluginIdentity>,
    request_id: String,
    inner: BackendChunkStream,
) -> BackendChunkStream {
    use futures::StreamExt;

    let span = open_streaming_span(host.as_ref(), &backend_name, &model);
    let t0 = std::time::Instant::now();

    struct State {
        inner: BackendChunkStream,
        host: Option<HostHandle>,
        backend_name: String,
        model: String,
        identity: Option<PluginIdentity>,
        request_id: String,
        t0: std::time::Instant,
        usage: UsageSnapshot,
        terminated: bool,
        last_err: Option<BackendError>,
        _span: Option<mcpg_plugin_sdk::SpanGuard>,
    }

    let init = State {
        inner,
        host,
        backend_name,
        model,
        identity,
        request_id,
        t0,
        usage: UsageSnapshot::default(),
        terminated: false,
        last_err: None,
        _span: span,
    };

    let stream = futures::stream::unfold(init, |mut state| async move {
        if state.terminated {
            return None;
        }
        match state.inner.next().await {
            Some(Ok(chunk)) => {
                if let BackendChunk::Usage {
                    input_tokens,
                    output_tokens,
                    cached_input_tokens,
                } = &chunk
                {
                    state.usage.input_tokens = state
                        .usage
                        .input_tokens
                        .saturating_add(*input_tokens as u64);
                    state.usage.output_tokens = state
                        .usage
                        .output_tokens
                        .saturating_add(*output_tokens as u64);
                    state.usage.cached_input_tokens = state
                        .usage
                        .cached_input_tokens
                        .saturating_add(*cached_input_tokens as u64);
                }
                let is_done = matches!(chunk, BackendChunk::Done(_));
                if is_done {
                    state.terminated = true;
                    let elapsed = state.t0.elapsed();
                    emit_chat_observability(
                        state.host.as_ref(),
                        &state.backend_name,
                        &state.model,
                        &state.request_id,
                        state.identity.as_ref(),
                        elapsed,
                        Ok(()),
                        Some(state.usage),
                    )
                    .await;
                }
                Some((Ok(chunk), state))
            }
            Some(Err(err)) => {
                state.terminated = true;
                state.last_err = Some(clone_backend_error(&err));
                let elapsed = state.t0.elapsed();
                emit_chat_observability(
                    state.host.as_ref(),
                    &state.backend_name,
                    &state.model,
                    &state.request_id,
                    state.identity.as_ref(),
                    elapsed,
                    Err(state.last_err.as_ref().unwrap()),
                    Some(state.usage),
                )
                .await;
                Some((Err(err), state))
            }
            None => None,
        }
    });
    Box::pin(stream)
}

fn clone_backend_error(err: &BackendError) -> BackendError {
    match err {
        BackendError::ProfileNotFound { backend_name } => BackendError::ProfileNotFound {
            backend_name: backend_name.clone(),
        },
        BackendError::InvalidSpec { message } => BackendError::InvalidSpec {
            message: message.clone(),
        },
        BackendError::Timeout { timeout_ms } => BackendError::Timeout {
            timeout_ms: *timeout_ms,
        },
        BackendError::Transport { message } => BackendError::Transport {
            message: message.clone(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mcpg_plugin_protocol::noop_backend_host;

    #[test]
    fn plugin_kind_and_manifest() {
        let p = AnthropicChatPlugin::new();
        assert_eq!(p.kind(), "anthropic.chat");
        assert_eq!(p.manifest().id, "dev.mcpg.backend.anthropic.chat");
    }

    #[tokio::test]
    async fn register_minimal_spec() {
        let plugin = AnthropicChatPlugin::new();
        plugin
            .register_profile(
                "claude",
                &serde_json::json!({
                    "model": "claude-3-5-sonnet",
                    "api_key": "k",
                    "prompt": { "system": "x", "user": "{{ input.text }}" },
                    "output_schema": { "type": "object", "properties": {"a":{"type":"string"}}, "required": ["a"] }
                }),
                noop_backend_host(),
            )
            .await
            .unwrap();
        assert_eq!(plugin.registered_profile_count(), 1);
    }
}

//! cdylib sync bridge — adapts the async `BackendPlugin` impl of the
//! single Anthropic plugin type onto the sync FFI trait the cdylib
//! vtable expects ([`SyncBackendPlugin`]). The wrapper owns a private
//! multi-thread runtime and `block_on`s the async logic; the make-time
//! [`HostHandle`] is turned into an `Arc<dyn BackendHost>` (via
//! [`HostHandleBackendHost`]) that is passed to `register_profile`,
//! and — since the chat type exposes `set_host_handle` — also installed
//! on the inner plugin for unified observability.
//!
//! Mirrors the openai conversion
//! (`libs/plugins/backend/llms/openai`) and the proven nats / kafka
//! pilots (`libs/plugins/backend/{nats,kafka}`). Deviations from the
//! nats template, all intentional:
//!
//! - The factory ignores `config_json` — the LLM plugins carry no
//!   plugin-level config (api_key, base_url, model, etc. all arrive
//!   per-binding via `register_profile`). nats/kafka parse a
//!   plugin-level connection config; here there is nothing to parse.
//! - No watch-strategy entity (this is a pure backend).
//! - Streaming IS bridged (v34): `execute_streaming` opens the inner
//!   async chunk stream and drains it on the private runtime, pushing
//!   each `BackendChunk` across the FFI `EventSinkRef`; `cancel_stream`
//!   fires a per-token `Notify` to stop the drain. So cdylib LLMs keep
//!   incremental token streaming, not just the buffered `execute`.
//! - `AnthropicChatCdylib` calls `inner.set_host_handle(...)` — the
//!   chat type exposes it.

use std::sync::Arc;

use mcpg_plugin_protocol::{
    BackendError, BackendPlugin, BackendRequest, BackendResponse, PluginManifest,
};
use mcpg_plugin_sdk::ffi::SyncBackendPlugin;
use mcpg_plugin_sdk::{HostHandle, HostHandleBackendHost};

use crate::plugin::AnthropicChatPlugin;

/// Build the private multi-thread runtime each cdylib wrapper uses to
/// `block_on` its async inner plugin. Two worker threads + `enable_all`
/// — copied from the nats template.
fn build_bridge_runtime(thread_name: &str) -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .thread_name(thread_name.to_owned())
        .enable_all()
        .build()
        .unwrap_or_else(|e| panic!("anthropic cdylib: tokio runtime init failed: {e}"))
}

/// Emit a `SyncBackendPlugin` bridge `$wrapper` over inner async plugin
/// `$inner`. `$thread` names the bridge runtime's worker threads.
///
/// The `set_host_handle` arm is optional: pass it (any token) for plugin
/// variants that expose the method; omit it for those that do not. The
/// host-handle is always wrapped into an `Arc<dyn BackendHost>` and
/// passed to `register_profile` regardless.
macro_rules! backend_cdylib_wrapper {
    (
        $(#[$meta:meta])*
        $wrapper:ident, $inner:ty, $thread:literal $(, set_host_handle = $shh:literal)?
    ) => {
        $(#[$meta])*
        pub struct $wrapper {
            inner: $inner,
            host: Arc<dyn mcpg_plugin_protocol::BackendHost>,
            rt: tokio::runtime::Runtime,
            /// Live `execute_streaming` drains, keyed by the cancel token
            /// returned to the host. Each entry holds a sticky
            /// `CancellationToken` (the drain task's `select!` arm) plus a
            /// completion channel `cancel_stream` blocks on so it returns
            /// only once the drain task has stopped emitting — the host frees
            /// the stream bridge the moment `cancel_stream` returns.
            streams: Arc<
                std::sync::Mutex<
                    std::collections::HashMap<
                        usize,
                        (
                            tokio_util::sync::CancellationToken,
                            std::sync::mpsc::Receiver<()>,
                        ),
                    >,
                >,
            >,
            next_stream_id: Arc<std::sync::atomic::AtomicUsize>,
        }

        impl $wrapper {
            /// Infallible cdylib factory. `config_json` is ignored —
            /// the LLM plugins carry no plugin-level config (per-binding
            /// api_key / base_url / model arrive via `register_profile`).
            pub fn from_host_config(_config_json: &str, host: HostHandle) -> Self {
                let inner = <$inner>::new();
                $(
                    // Only emitted for variants that expose the method
                    // ($shh just selects this arm). Install the unified
                    // observability handle on the inner plugin
                    // (idempotent; ignore the returned bool).
                    let _: bool = $shh;
                    let _installed = inner.set_host_handle(host.clone());
                )?
                Self {
                    inner,
                    host: Arc::new(HostHandleBackendHost::new(host)),
                    rt: build_bridge_runtime($thread),
                    streams: Arc::new(std::sync::Mutex::new(std::collections::HashMap::new())),
                    next_stream_id: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
                }
            }
        }

        impl SyncBackendPlugin for $wrapper {
            fn manifest(&self) -> &PluginManifest {
                BackendPlugin::manifest(&self.inner)
            }
            fn kind(&self) -> &str {
                BackendPlugin::kind(&self.inner)
            }
            fn register_profile(
                &self,
                profile_name: &str,
                spec: &serde_json::Value,
            ) -> Result<(), BackendError> {
                self.rt.block_on(BackendPlugin::register_profile(
                    &self.inner,
                    profile_name,
                    spec,
                    Arc::clone(&self.host),
                ))
            }
            fn execute(
                &self,
                profile_name: &str,
                request: BackendRequest,
            ) -> Result<BackendResponse, BackendError> {
                self.rt
                    .block_on(BackendPlugin::execute(&self.inner, profile_name, request))
            }

            fn execute_streaming(
                &self,
                profile_name: &str,
                request: BackendRequest,
                emit: mcpg_plugin_sdk::ffi::BackendChunkEmitter,
            ) -> Result<usize, BackendError> {
                use futures::StreamExt;
                // Open the inner async stream (borrows `inner` only for
                // this await), then drain it on the private runtime,
                // pushing each chunk across the FFI via `emit`. Returns a
                // non-zero cancel token; `cancel_stream` cancels the matching
                // sticky token to stop the drain promptly even mid `next()`,
                // then waits for the drain task to finish.
                let stream = self.rt.block_on(BackendPlugin::execute_streaming(
                    &self.inner,
                    profile_name,
                    request,
                ))?;
                let cancel = tokio_util::sync::CancellationToken::new();
                let (done_tx, done_rx) = std::sync::mpsc::sync_channel::<()>(1);
                // Non-zero token (0 is reserved for "nothing to cancel").
                let token = self
                    .next_stream_id
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                    .wrapping_add(1);
                self.streams
                    .lock()
                    .expect("stream registry poisoned")
                    .insert(token, (cancel.clone(), done_rx));
                let streams = Arc::clone(&self.streams);
                self.rt.spawn(async move {
                    let mut stream = stream;
                    loop {
                        tokio::select! {
                            // Sticky: stays ready once cancelled, so a task
                            // re-parked after a chunk still observes it (no
                            // lost wakeup, unlike `Notify::notify_waiters`).
                            _ = cancel.cancelled() => break,
                            item = stream.next() => match item {
                                Some(chunk) => emit(chunk),
                                None => break,
                            },
                        }
                    }
                    streams
                        .lock()
                        .expect("stream registry poisoned")
                        .remove(&token);
                    // Unblock a waiting `cancel_stream`: no further `emit`
                    // calls will occur, so the host may free the bridge.
                    let _ = done_tx.send(());
                });
                Ok(token)
            }

            fn cancel_stream(&self, token: usize) {
                // Take the entry out under the lock, then release it before
                // blocking so concurrent stream ops aren't held up.
                let entry = self
                    .streams
                    .lock()
                    .expect("stream registry poisoned")
                    .remove(&token);
                if let Some((cancel, done_rx)) = entry {
                    cancel.cancel();
                    // Block until the drain task has left its loop — the host
                    // frees the stream bridge the instant we return, so no
                    // `emit` may run afterwards. A plain channel recv (NOT a
                    // nested `block_on`, which would panic if this runs inside
                    // a runtime); the drain task makes progress on the
                    // wrapper's own runtime. `Err` => task already finished.
                    let _ = done_rx.recv();
                }
            }
        }
    };
}

// --- chat (exposes `set_host_handle`) --------------------------------
backend_cdylib_wrapper! {
    /// `SyncBackendPlugin` bridge over [`AnthropicChatPlugin`].
    AnthropicChatCdylib, AnthropicChatPlugin, "mcpg-llm-anthropic-chat", set_host_handle = true
}

// cdylib export — the single entity under
// `dev.mcpg.backend.llm.anthropic`. The entity reports its real kind
// (anthropic.chat) through its `manifest()` slot at runtime, so the
// host dispatches each binding to the wrapper. The chat type gets the
// bare `""` alias (`{plugin_id}:{inner_name}`).
mcpg_plugin_sdk::declare_plugin! {
    plugin_id: "dev.mcpg.backend.llm.anthropic",
    plugin_version: env!("CARGO_PKG_VERSION"),
    descriptor_yaml: include_str!("../plugin.yaml"),
    capabilities: &[::mcpg_plugin_protocol::capability::Capability::NetworkOutbound],
    // LLM bindings self-configure a dynamic resource list (the gateway
    // merges per-binding `list_resources` output on `resources/list`), so
    // every LLM kind declares `dynamic_list`. Health is request-time (Skip),
    // the label comes from the `LlmRequest` behavioural route (not this
    // profile), and LLM kinds are not pipeline steps — all behaviour-neutral
    // defaults. One profile covers every backend entity this crate exports.
    backend_profile: ::mcpg_plugin_protocol::manifest::BackendProfile {
        dynamic_list: true,
        ..::core::default::Default::default()
    },
    entities: [
        backend as chat {
            inner_name: "",
            plugin_type: AnthropicChatCdylib,
            factory: |cfg, host: ::mcpg_plugin_sdk::HostHandle|
                AnthropicChatCdylib::from_host_config(cfg, host),
        },
    ],
}

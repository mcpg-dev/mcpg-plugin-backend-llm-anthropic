//! # mcpg-plugin-backend-llm-anthropic
//!
//! Anthropic Messages API chat-completion binding plugin for MCPG.
//!
//! Ships [`AnthropicChatPlugin`] (`kind: "anthropic.chat"`).
//! Delegates execution to [`mcpg_backend_llm_shared::ChatEngine`];
//! the [`AnthropicAdapter`] handles the provider-specific wire
//! format (system prompt at top-level, content-blocks array,
//! forced-tool structured output via synthetic `respond`).
//!
//! Anthropic itself has no native embedding API. The companion
//! crate `mcpg-plugin-backend-llm-voyage` (separate dedicated
//! plugin) provides Anthropic's recommended embedding partner.

mod adapter;
/// cdylib sync bridge + `declare_plugin!` export (backend-plugin-migration).
/// Additive: the gateway keeps using the static `new()` + `set_host_handle`
/// path. The `mcpg_plugin_register` FFI symbol is gated behind the
/// `cdylib-export` feature inside the macro expansion. Public so the
/// wrapper types + macro-generated entity modules are part of the
/// crate's public surface (mirrors the openai conversion + the nats /
/// kafka pilots, which keep their bridges at crate root) — this also
/// keeps the wrappers from tripping `dead_code` on the default rlib
/// build where neither `cdylib-export` nor `static-firstparty`
/// references them.
pub mod cdylib;
mod config;
mod host_handle_obs;
mod plugin;

pub use adapter::AnthropicAdapter;
pub use config::AnthropicChatSpec;
pub use plugin::AnthropicChatPlugin;

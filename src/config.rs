//! Operator-facing config for the Anthropic chat binding.
//!
//! Single spec — no Azure-style variant. Default base URL is
//! `https://api.anthropic.com/v1`; operators override only for
//! testing or for forwarding proxies.

use mcpg_backend_llm_shared::{ApiKeyRef, ChatExecutionSpec, ConfigError};
use serde::{Deserialize, Serialize};

/// Spec for `binding_type: anthropic_chat`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicChatSpec {
    /// Override for the default `https://api.anthropic.com/v1`.
    /// Most operators leave this unset.
    #[serde(default)]
    pub base_url: Option<String>,

    pub api_key: ApiKeyRef,

    /// Provider-agnostic execution config flattened into this spec.
    #[serde(flatten)]
    pub chat: ChatExecutionSpec,
}

impl AnthropicChatSpec {
    pub const DEFAULT_BASE_URL: &'static str = "https://api.anthropic.com/v1";

    pub fn validate(&self) -> Result<(), ConfigError> {
        // Anthropic has no provider-specific invariants beyond the
        // shared ChatExecutionSpec rules.
        self.chat.validate()
    }

    pub fn resolved_base_url(&self) -> &str {
        self.base_url.as_deref().unwrap_or(Self::DEFAULT_BASE_URL)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mcpg_backend_llm_shared::PromptSpec;
    use serde_json::json;

    fn minimal() -> ChatExecutionSpec {
        ChatExecutionSpec {
            model: "claude-3-5-sonnet".into(),
            timeout_ms: 30_000,
            connect_timeout_ms: 5_000,
            prompt: PromptSpec {
                system: "you are helpful".into(),
                user: "{{ input.text }}".into(),
                ..Default::default()
            },
            sampling: Default::default(),
            response_format: Default::default(),
            tools: Default::default(),
            streaming: Default::default(),
            retry: Default::default(),
            guardrails: Default::default(),
            cache: Default::default(),
            budget: Default::default(),
        }
    }

    #[test]
    fn default_base_url() {
        let s = AnthropicChatSpec {
            base_url: None,
            api_key: ApiKeyRef::new("k"),
            chat: minimal(),
        };
        assert_eq!(s.resolved_base_url(), "https://api.anthropic.com/v1");
        s.validate().unwrap();
    }

    #[test]
    fn json_round_trip() {
        let json = json!({
            "model": "claude-3-5-sonnet",
            "api_key": "k",
            "prompt": { "system": "x", "user": "y" }
        });
        let s: AnthropicChatSpec = serde_json::from_value(json).unwrap();
        assert!(s.base_url.is_none());
        s.validate().unwrap();
    }
}

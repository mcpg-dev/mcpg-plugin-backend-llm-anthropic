//! HostHandle observability triad for the
//! Anthropic chat backend plugin. Mirrors the OpenAI variant; only
//! the action names + metric prefix differ.

use std::time::Duration;

use mcpg_backend_llm_shared::cost::{bundled_rate_card, compute_chat_cost_usd};
use mcpg_backend_llm_shared::normalized::TokenUsage;
use mcpg_plugin_protocol::BackendError;
use mcpg_plugin_protocol::audit::{AuditEvent, AuditOutcome};
use mcpg_plugin_protocol::types::PluginIdentity;
use mcpg_plugin_sdk::{HostHandle, SpanGuard};

pub(crate) const SPAN_EXECUTE: &str = "llm_anthropic.execute";
pub(crate) const SPAN_STREAMING: &str = "llm_anthropic.execute_streaming";
pub(crate) const LATENCY_METRIC: &str = "mcpg_llm_anthropic_latency_seconds";
pub(crate) const CALLS_METRIC: &str = "mcpg_llm_anthropic_calls_total";
pub(crate) const INPUT_TOKENS_METRIC: &str = "mcpg_llm_anthropic_input_tokens_total";
pub(crate) const OUTPUT_TOKENS_METRIC: &str = "mcpg_llm_anthropic_output_tokens_total";
pub(crate) const COST_METRIC: &str = "mcpg_llm_anthropic_cost_usd_micros_total";
pub(crate) const COMPLETION_ACTION: &str = "dev.mcpg.llm.anthropic.completion";
pub(crate) const FAILURE_ACTION: &str = "dev.mcpg.llm.anthropic.failure";

/// Classify a chat-execute outcome into the bounded label set. The
/// label space is intentionally closed so operator dashboards can
/// pivot on `outcome` without cardinality blow-up.
pub(crate) fn outcome_label(result: Result<(), &BackendError>) -> &'static str {
    let Err(err) = result else {
        return "ok";
    };
    match err {
        BackendError::ProfileNotFound { .. } => "model_not_found",
        BackendError::Timeout { .. } => "timeout",
        BackendError::InvalidSpec { .. } => "client_error",
        BackendError::Transport { message } => transport_message_label(message),
    }
}

fn transport_message_label(message: &str) -> &'static str {
    let lower = message.to_ascii_lowercase();
    if lower.contains("429") || lower.contains("rate limit") || lower.contains("rate_limit") {
        "rate_limited"
    } else if lower.contains("401")
        || lower.contains("403")
        || lower.contains("auth")
        || lower.contains("unauthorized")
        || lower.contains("forbidden")
    {
        "auth_failed"
    } else if lower.contains("404")
        || lower.contains("model not found")
        || lower.contains("does not exist")
    {
        "model_not_found"
    } else if lower.contains("500")
        || lower.contains("502")
        || lower.contains("503")
        || lower.contains("504")
        || lower.contains("server error")
        || lower.contains("overloaded")
    {
        "server_error"
    } else if lower.contains("400") || lower.contains("invalid") {
        "client_error"
    } else if lower.contains("timed out") || lower.contains("timeout") {
        "timeout"
    } else {
        "transport"
    }
}

fn audit_action_for_outcome(label: &str) -> &'static str {
    if label == "ok" {
        COMPLETION_ACTION
    } else {
        FAILURE_ACTION
    }
}

fn synthetic_system_identity() -> PluginIdentity {
    PluginIdentity {
        kind: "system".into(),
        trust_level: "verified".into(),
        subject_id: Some("dev.mcpg.backend.anthropic.chat".into()),
        auth_provider: None,
        issuer: None,
        roles: vec![],
        groups: vec![],
        scopes: vec![],
        attributes: Default::default(),
    }
}

fn rfc3339_now() -> String {
    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true)
}

#[derive(Default, Debug, Clone, Copy)]
pub(crate) struct UsageSnapshot {
    pub(crate) input_tokens: u64,
    pub(crate) output_tokens: u64,
    pub(crate) cached_input_tokens: u64,
}

pub(crate) fn open_span(
    host: Option<&HostHandle>,
    backend_name: &str,
    model: &str,
) -> Option<SpanGuard> {
    host.map(|h| {
        h.span(
            SPAN_EXECUTE,
            serde_json::json!({
                "binding": backend_name,
                "model": model,
            }),
        )
    })
}

pub(crate) fn open_streaming_span(
    host: Option<&HostHandle>,
    backend_name: &str,
    model: &str,
) -> Option<SpanGuard> {
    host.map(|h| {
        h.span(
            SPAN_STREAMING,
            serde_json::json!({
                "binding": backend_name,
                "model": model,
            }),
        )
    })
}

#[allow(clippy::too_many_arguments)] // Bounded per-call observability surface.
pub(crate) async fn emit_chat_observability(
    host: Option<&HostHandle>,
    backend_name: &str,
    model: &str,
    request_id: &str,
    identity: Option<&PluginIdentity>,
    elapsed: Duration,
    result: Result<(), &BackendError>,
    usage: Option<UsageSnapshot>,
) {
    let Some(host) = host else {
        return;
    };
    let label = outcome_label(result);
    let elapsed_secs = elapsed.as_secs_f64();

    host.histogram(
        LATENCY_METRIC,
        elapsed_secs,
        &[("outcome", label), ("model", model)],
    );
    host.counter(CALLS_METRIC, 1, &[("outcome", label), ("model", model)]);

    let cost_usd_micros: Option<u64> = if let Some(snap) = usage {
        host.counter(INPUT_TOKENS_METRIC, snap.input_tokens, &[("model", model)]);
        host.counter(
            OUTPUT_TOKENS_METRIC,
            snap.output_tokens,
            &[("model", model)],
        );
        let usage_struct = TokenUsage {
            input_tokens: snap.input_tokens.min(u32::MAX as u64) as u32,
            output_tokens: snap.output_tokens.min(u32::MAX as u64) as u32,
            cached_input_tokens: snap.cached_input_tokens.min(u32::MAX as u64) as u32,
        };
        let cost = compute_chat_cost_usd(bundled_rate_card(), "anthropic", model, &usage_struct);
        cost.map(|usd| {
            let micros = (usd * 1_000_000.0).round().max(0.0) as u64;
            host.counter(COST_METRIC, micros, &[("model", model)]);
            micros
        })
    } else {
        None
    };

    let action = audit_action_for_outcome(label);
    let actor = identity.cloned().unwrap_or_else(synthetic_system_identity);
    let mut details = serde_json::json!({
        "binding": backend_name,
        "model": model,
        "outcome": label,
        "provider": "anthropic",
        "duration_ms": elapsed.as_millis() as u64,
        "alias": host.alias(),
    });
    if let Some(snap) = usage {
        let object = details.as_object_mut().expect("json object");
        object.insert(
            "input_tokens".into(),
            serde_json::Value::from(snap.input_tokens),
        );
        object.insert(
            "output_tokens".into(),
            serde_json::Value::from(snap.output_tokens),
        );
        object.insert(
            "total_tokens".into(),
            serde_json::Value::from(snap.input_tokens + snap.output_tokens),
        );
        if snap.cached_input_tokens > 0 {
            object.insert(
                "cached_input_tokens".into(),
                serde_json::Value::from(snap.cached_input_tokens),
            );
        }
        if let Some(micros) = cost_usd_micros {
            object.insert("cost_usd_micros".into(), serde_json::Value::from(micros));
        }
    }
    if let Err(err) = result {
        let object = details.as_object_mut().expect("json object");
        object.insert(
            "error_class".into(),
            serde_json::Value::String(label.to_owned()),
        );
        object.insert(
            "error_message".into(),
            serde_json::Value::String(err.to_string()),
        );
    }
    let outcome_class = if result.is_ok() {
        AuditOutcome::Success
    } else {
        AuditOutcome::Failure
    };
    let event = AuditEvent {
        event_id: format!("llm-anthropic-{}-{}", request_id, elapsed.as_nanos()),
        occurred_at: rfc3339_now(),
        actor,
        action: action.to_owned(),
        resource: Some(format!("llm-binding://anthropic/{}", backend_name)),
        outcome: outcome_class,
        request_id: Some(request_id.to_owned()),
        node_id: None,
        details,
        prev_event_hash: None,
    };

    let host_for_audit = host.clone();
    if let Err(join_err) = tokio::task::spawn_blocking(move || {
        let _ = host_for_audit.audit_event(event);
    })
    .await
    {
        tracing::debug!(
            target: "mcpg::llm_anthropic::host_handle",
            error = %join_err,
            "host_handle.audit_event spawn_blocking failed"
        );
    }
}

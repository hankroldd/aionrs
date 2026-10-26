use super::*;
use serde_json::json;

fn tool_chunk(arguments: Value) -> String {
    json!({"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call-fill",
        "function": {"name": "office_fill_template", "arguments": arguments}}]},
        "finish_reason": "tool_calls"}]})
    .to_string()
}

#[test]
fn invalid_arguments_fail_without_emitting_tool_use() {
    for arguments in [
        json!(""),
        json!("{"),
        json!("not-json"),
        json!("{\"secret\":\"private-text"),
        json!({}),
    ] {
        let mut state = StreamState::new();
        let events = parse_sse_chunk(&tool_chunk(arguments), &mut state, false);
        assert!(!events.iter().any(|event| matches!(event, LlmEvent::ToolUse { .. })));
        assert!(state.flush_done().is_none());
        let error = state
            .take_stream_error()
            .expect("invalid arguments must fail the stream");
        assert!(matches!(error, ProviderError::Parse(_)));
        assert!(error.to_string().contains("call-fill"));
        assert!(!error.to_string().contains("private-text"));
    }
}

#[test]
fn valid_arguments_preserve_objects_including_empty_objects() {
    for input in [json!({}), json!({"decision_id": "decision", "content": {"items": []}})] {
        let mut state = StreamState::new();
        let events = parse_sse_chunk(&tool_chunk(json!(input.to_string())), &mut state, false);
        assert!(
            matches!(&events[..], [LlmEvent::ToolUse { id, input: actual, .. }] if id == "call-fill" && actual == &input)
        );
        assert!(matches!(
            state.flush_done(),
            Some(LlmEvent::Done {
                stop_reason: StopReason::ToolUse,
                ..
            })
        ));
    }
}

#[test]
fn invalid_arguments_prevent_partial_tool_round_emission() {
    let chunk = json!({"choices": [{"delta": {"tool_calls": [
        {"index": 0, "id": "good", "function": {"name": "Read", "arguments": "{}"}},
        {"index": 1, "id": "bad", "function": {"name": "Fill", "arguments": "{"}}
    ]}, "finish_reason": "stop"}]})
    .to_string();
    let mut state = StreamState::new();
    let events = parse_sse_chunk(&chunk, &mut state, false);
    assert!(matches!(state.take_stream_error(), Some(ProviderError::Parse(_))));
    assert!(!events.iter().any(|event| matches!(event, LlmEvent::ToolUse { .. })));
}

#[test]
fn length_termination_does_not_emit_incomplete_arguments() {
    let chunk = tool_chunk(json!("{")).replace("\"finish_reason\":\"tool_calls\"", "\"finish_reason\":\"length\"");
    let mut state = StreamState::new();
    let events = parse_sse_chunk(&chunk, &mut state, false);
    assert!(events.is_empty());
    assert!(matches!(
        state.flush_done(),
        Some(LlmEvent::Done {
            stop_reason: StopReason::MaxTokens,
            ..
        })
    ));
}

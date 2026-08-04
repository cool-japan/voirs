//! Tests for `StepExecutor` and `ExecutionContext` (split out of `mod.rs`
//! to keep that file under the workspace's ~2000-line guideline).

use super::*;
use crate::workflow::definition::{ConditionOperator, StepType};

/// Build a minimal `Step` for tests, filling in the fields that are
/// irrelevant to the specific handler under test.
fn make_step(
    name: &str,
    step_type: StepType,
    parameters: HashMap<String, serde_json::Value>,
    condition: Option<Condition>,
) -> Step {
    Step {
        name: name.to_string(),
        step_type,
        description: None,
        parameters,
        condition,
        depends_on: Vec::new(),
        retry: None,
        for_each: None,
        parallel: false,
    }
}

fn unique_temp_path(label: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "voirs_executor_test_{label}_{}_{}",
        std::process::id(),
        fastrand::u64(..)
    ))
}

#[test]
fn test_execution_context_creation() {
    let workflow = Workflow::new("test", "1.0", "Test workflow");
    let context = ExecutionContext::new(workflow);

    assert_eq!(context.completed_steps().len(), 0);
    assert_eq!(context.skipped_steps().len(), 0);
    assert_eq!(context.total_retries(), 0);
}

#[test]
fn test_execution_context_variables() {
    let mut workflow = Workflow::new("test", "1.0", "Test workflow");
    workflow.add_variable(
        "test_var".to_string(),
        super::super::definition::Variable::String("test_value".to_string()),
    );

    let context = ExecutionContext::new(workflow);
    let variables = context.get_variables();

    assert_eq!(variables.len(), 1);
    assert_eq!(
        variables
            .get("test_var")
            .unwrap()
            .as_str()
            .unwrap_or_default(),
        "test_value"
    );
}

#[test]
fn test_step_result_creation() {
    let result = StepResult::success("step1".to_string(), "Success".to_string(), 100);

    assert!(result.success);
    assert_eq!(result.step_name, "step1");
    assert_eq!(result.duration_ms, 100);
}

#[test]
fn test_step_result_with_output() {
    let result = StepResult::success("step1".to_string(), "Success".to_string(), 100)
        .with_output("key1".to_string(), serde_json::json!("value1"));

    assert_eq!(result.output.len(), 1);
    assert_eq!(
        result
            .output
            .get("key1")
            .unwrap()
            .as_str()
            .unwrap_or_default(),
        "value1"
    );
}

#[tokio::test]
async fn test_step_executor_creation() {
    let _executor = StepExecutor::new();
    // Verify creation works without panic
}

#[test]
fn test_execution_result_success() {
    let stats = WorkflowStats::new();
    let result = ExecutionResult::success("test".to_string(), "Done".to_string(), stats);

    assert!(result.success);
    assert_eq!(result.workflow_name, "test");
}

#[test]
fn test_execution_result_failure() {
    let stats = WorkflowStats::new();
    let result = ExecutionResult::failure("test".to_string(), "Failed".to_string(), stats);

    assert!(!result.success);
    assert_eq!(result.message, "Failed");
}

#[tokio::test]
async fn test_execute_file_op_write_then_read() {
    let executor = StepExecutor::new();
    let workflow = Workflow::new("file-op-test", "1.0", "Test file operations");
    let mut context = ExecutionContext::new(workflow);

    let temp_path = unique_temp_path("file_op");

    let mut write_params = HashMap::new();
    write_params.insert("op".to_string(), serde_json::json!("write"));
    write_params.insert(
        "path".to_string(),
        serde_json::json!(temp_path.display().to_string()),
    );
    write_params.insert("content".to_string(), serde_json::json!("hello workflow"));
    let write_step = make_step("write-step", StepType::FileOp, write_params, None);

    let write_result = executor
        .execute_step(&write_step, &mut context)
        .await
        .unwrap();
    assert!(
        write_result.success,
        "write step failed: {}",
        write_result.message
    );
    assert!(temp_path.exists());

    let mut read_params = HashMap::new();
    read_params.insert("op".to_string(), serde_json::json!("read"));
    read_params.insert(
        "path".to_string(),
        serde_json::json!(temp_path.display().to_string()),
    );
    let read_step = make_step("read-step", StepType::FileOp, read_params, None);

    let read_result = executor
        .execute_step(&read_step, &mut context)
        .await
        .unwrap();
    assert!(
        read_result.success,
        "read step failed: {}",
        read_result.message
    );
    assert_eq!(
        read_result.output.get("content").and_then(|v| v.as_str()),
        Some("hello workflow")
    );

    let _ = std::fs::remove_file(&temp_path);
}

#[tokio::test]
async fn test_execute_file_op_unknown_op_errors() {
    let executor = StepExecutor::new();
    let workflow = Workflow::new("file-op-error-test", "1.0", "Test file op error path");
    let mut context = ExecutionContext::new(workflow);

    let mut params = HashMap::new();
    params.insert("op".to_string(), serde_json::json!("frobnicate"));
    params.insert("path".to_string(), serde_json::json!("irrelevant.txt"));
    let step = make_step("bad-op-step", StepType::FileOp, params, None);

    let result = executor.execute_step(&step, &mut context).await.unwrap();
    assert!(!result.success);
    assert!(result.message.contains("frobnicate"));
}

#[tokio::test]
async fn test_execute_command_captures_stdout() {
    let executor = StepExecutor::new();
    let workflow = Workflow::new("command-test", "1.0", "Test command execution");
    let mut context = ExecutionContext::new(workflow);

    let mut params = HashMap::new();
    params.insert("command".to_string(), serde_json::json!("echo hello"));
    let step = make_step("echo-step", StepType::Command, params, None);

    let result = executor.execute_step(&step, &mut context).await.unwrap();
    assert!(result.success, "command step failed: {}", result.message);
    assert_eq!(
        result.output.get("stdout").and_then(|v| v.as_str()),
        Some("hello")
    );
    assert_eq!(
        result.output.get("exit_code").and_then(|v| v.as_i64()),
        Some(0)
    );
}

#[tokio::test]
async fn test_execute_command_nonzero_exit_fails() {
    let executor = StepExecutor::new();
    let workflow = Workflow::new("command-fail-test", "1.0", "Test command failure path");
    let mut context = ExecutionContext::new(workflow);

    let mut params = HashMap::new();
    params.insert("command".to_string(), serde_json::json!("exit 7"));
    let step = make_step("fail-step", StepType::Command, params, None);

    let result = executor.execute_step(&step, &mut context).await.unwrap();
    assert!(!result.success);
    assert!(result.message.contains('7'));
}

#[tokio::test]
async fn test_execute_script_runs_body() {
    let executor = StepExecutor::new();
    let workflow = Workflow::new("script-test", "1.0", "Test script execution");
    let mut context = ExecutionContext::new(workflow);

    let mut params = HashMap::new();
    params.insert("script".to_string(), serde_json::json!("echo scripted"));
    let step = make_step("script-step", StepType::Script, params, None);

    let result = executor.execute_step(&step, &mut context).await.unwrap();
    assert!(result.success, "script step failed: {}", result.message);
    assert_eq!(
        result.output.get("stdout").and_then(|v| v.as_str()),
        Some("scripted")
    );
}

#[tokio::test]
async fn test_execute_branch_picks_path_from_condition() {
    let executor = StepExecutor::new();
    let mut workflow = Workflow::new("branch-test", "1.0", "Test branch evaluation");
    workflow.add_variable(
        "score".to_string(),
        super::super::definition::Variable::Number(4.5),
    );
    let mut context = ExecutionContext::new(workflow);

    // True branch: score (4.5) > 4.0
    let true_step = make_step(
        "branch-true",
        StepType::Branch,
        HashMap::new(),
        Some(Condition::new(
            "${score}".to_string(),
            ConditionOperator::GreaterThan,
            "4.0".to_string(),
        )),
    );
    let true_result = executor
        .execute_step(&true_step, &mut context)
        .await
        .unwrap();
    assert!(true_result.success);
    assert_eq!(
        true_result.output.get("branch_taken"),
        Some(&serde_json::json!(true))
    );
    assert_eq!(
        context.get_variables().get("branch_taken"),
        Some(&serde_json::json!(true))
    );

    // False branch: score (4.5) > 10.0 is false
    let false_step = make_step(
        "branch-false",
        StepType::Branch,
        HashMap::new(),
        Some(Condition::new(
            "${score}".to_string(),
            ConditionOperator::GreaterThan,
            "10.0".to_string(),
        )),
    );
    let false_result = executor
        .execute_step(&false_step, &mut context)
        .await
        .unwrap();
    assert!(false_result.success);
    assert_eq!(
        false_result.output.get("branch_taken"),
        Some(&serde_json::json!(false))
    );
}

#[tokio::test]
async fn test_execute_branch_without_condition_errors() {
    let executor = StepExecutor::new();
    let workflow = Workflow::new("branch-missing-condition", "1.0", "Test missing condition");
    let mut context = ExecutionContext::new(workflow);

    let step = make_step("branch-no-cond", StepType::Branch, HashMap::new(), None);
    let result = executor.execute_step(&step, &mut context).await.unwrap();
    assert!(!result.success);
}

#[tokio::test]
async fn test_execute_branch_condition_from_parameters() {
    // Distinct code path from `test_execute_branch_picks_path_from_condition`:
    // here the condition comes from a `condition` parameter (raw, unresolved
    // at substitution time) instead of the step-level `condition` field.
    let executor = StepExecutor::new();
    let mut workflow = Workflow::new("branch-param-condition", "1.0", "Test param condition");
    workflow.add_variable(
        "score".to_string(),
        super::super::definition::Variable::Number(4.5),
    );
    let mut context = ExecutionContext::new(workflow);

    let mut params = HashMap::new();
    params.insert(
        "condition".to_string(),
        serde_json::json!({"left": "${score}", "operator": ">", "right": "4.0"}),
    );
    let step = make_step("branch-from-params", StepType::Branch, params, None);

    let result = executor.execute_step(&step, &mut context).await.unwrap();
    assert!(result.success, "branch step failed: {}", result.message);
    assert_eq!(
        result.output.get("branch_taken"),
        Some(&serde_json::json!(true))
    );
}

#[tokio::test]
async fn test_execute_loop_count_based() {
    let executor = StepExecutor::new();
    let workflow = Workflow::new("loop-count-test", "1.0", "Test count-based loop");
    let mut context = ExecutionContext::new(workflow);

    let mut params = HashMap::new();
    params.insert("count".to_string(), serde_json::json!(5));
    let step = make_step("loop-step", StepType::Loop, params, None);

    let result = executor.execute_step(&step, &mut context).await.unwrap();
    assert!(result.success, "loop step failed: {}", result.message);
    assert_eq!(result.output.get("iterations"), Some(&serde_json::json!(5)));

    let variables = context.get_variables();
    assert_eq!(
        variables.get("loop-step_iterations"),
        Some(&serde_json::json!(5))
    );
}

#[tokio::test]
async fn test_execute_loop_count_exceeding_max_errors() {
    let executor = StepExecutor::new();
    let workflow = Workflow::new("loop-too-big-test", "1.0", "Test loop cap enforcement");
    let mut context = ExecutionContext::new(workflow);

    let mut params = HashMap::new();
    params.insert(
        "count".to_string(),
        serde_json::json!(MAX_LOOP_ITERATIONS + 1),
    );
    let step = make_step("loop-too-big", StepType::Loop, params, None);

    let result = executor.execute_step(&step, &mut context).await.unwrap();
    assert!(!result.success);
}

#[tokio::test]
async fn test_execute_loop_while_condition() {
    let executor = StepExecutor::new();
    let workflow = Workflow::new("loop-while-test", "1.0", "Test while-loop execution");
    let mut context = ExecutionContext::new(workflow);

    let step = make_step(
        "while-step",
        StepType::Loop,
        HashMap::new(),
        Some(Condition::new(
            "${loop_counter}".to_string(),
            ConditionOperator::LessThan,
            "3".to_string(),
        )),
    );

    let result = executor.execute_step(&step, &mut context).await.unwrap();
    assert!(result.success, "while-loop step failed: {}", result.message);
    assert_eq!(result.output.get("iterations"), Some(&serde_json::json!(3)));
    assert_eq!(
        context.get_variables().get("loop_counter"),
        Some(&serde_json::json!(3))
    );
}

// -- execute_synthesize -------------------------------------------------

#[tokio::test]
async fn test_execute_synthesize_writes_real_audio_that_scales_with_text_length() {
    let executor = StepExecutor::new();
    let workflow = Workflow::new("synth-test", "1.0", "Test synthesize step");
    let mut context = ExecutionContext::new(workflow);

    let short_path = unique_temp_path("synth_short").with_extension("wav");
    let mut short_params = HashMap::new();
    short_params.insert("text".to_string(), serde_json::json!("Hi."));
    short_params.insert(
        "output".to_string(),
        serde_json::json!(short_path.display().to_string()),
    );
    // Avoid attempting a real model download in tests: `test_mode`
    // falls back to the SDK's built-in dummy models, which are still
    // real code that genuinely varies its output with phoneme count
    // (see `DummyAcousticModel::generate_mel_data` in voirs-acoustic).
    short_params.insert("test_mode".to_string(), serde_json::json!(true));
    let short_step = make_step("synth-short", StepType::Synthesize, short_params, None);

    let short_result = executor
        .execute_step(&short_step, &mut context)
        .await
        .unwrap();
    assert!(
        short_result.success,
        "short synth step failed: {}",
        short_result.message
    );
    assert!(short_path.exists());

    let long_path = unique_temp_path("synth_long").with_extension("wav");
    let mut long_params = HashMap::new();
    long_params.insert(
        "text".to_string(),
        serde_json::json!(
            "This is a considerably longer sentence used to prove that synthesis \
                 duration genuinely scales with the amount of input text provided."
        ),
    );
    long_params.insert(
        "output".to_string(),
        serde_json::json!(long_path.display().to_string()),
    );
    long_params.insert("test_mode".to_string(), serde_json::json!(true));
    let long_step = make_step("synth-long", StepType::Synthesize, long_params, None);

    let long_result = executor
        .execute_step(&long_step, &mut context)
        .await
        .unwrap();
    assert!(
        long_result.success,
        "long synth step failed: {}",
        long_result.message
    );
    assert!(long_path.exists());

    // Both outputs must be genuinely decodable WAV audio (real I/O, not
    // a placeholder file), and the longer input must produce longer
    // audio -- proving the step actually threads `text` into real
    // synthesis instead of writing a fixed canned result.
    let short_reader =
        hound::WavReader::open(&short_path).expect("short output should be valid WAV");
    let short_duration =
        short_reader.duration() as f64 / f64::from(short_reader.spec().sample_rate);

    let long_reader = hound::WavReader::open(&long_path).expect("long output should be valid WAV");
    let long_duration = long_reader.duration() as f64 / f64::from(long_reader.spec().sample_rate);

    assert!(
            long_duration > short_duration,
            "expected longer text to produce longer audio: short={short_duration}s long={long_duration}s"
        );

    let reported_short = short_result
        .output
        .get("duration_seconds")
        .and_then(|v| v.as_f64())
        .expect("duration_seconds should be reported");
    assert!((reported_short - short_duration).abs() < 0.01);

    let _ = std::fs::remove_file(&short_path);
    let _ = std::fs::remove_file(&long_path);
}

#[tokio::test]
async fn test_execute_synthesize_missing_text_errors() {
    let executor = StepExecutor::new();
    let workflow = Workflow::new("synth-missing-text", "1.0", "Test missing text param");
    let mut context = ExecutionContext::new(workflow);

    let mut params = HashMap::new();
    params.insert(
        "output".to_string(),
        serde_json::json!(unique_temp_path("synth_notext").display().to_string()),
    );
    let step = make_step("synth-no-text", StepType::Synthesize, params, None);

    let result = executor.execute_step(&step, &mut context).await.unwrap();
    assert!(!result.success);
    assert!(result.message.contains("text"));
}

#[tokio::test]
async fn test_execute_synthesize_missing_output_errors() {
    let executor = StepExecutor::new();
    let workflow = Workflow::new("synth-missing-output", "1.0", "Test missing output param");
    let mut context = ExecutionContext::new(workflow);

    let mut params = HashMap::new();
    params.insert("text".to_string(), serde_json::json!("hello"));
    let step = make_step("synth-no-output", StepType::Synthesize, params, None);

    let result = executor.execute_step(&step, &mut context).await.unwrap();
    assert!(!result.success);
    assert!(result.message.contains("output"));
}

// -- execute_validate -----------------------------------------------------

#[tokio::test]
async fn test_execute_validate_no_checks_errors() {
    let executor = StepExecutor::new();
    let workflow = Workflow::new(
        "validate-empty",
        "1.0",
        "Test validate with nothing to check",
    );
    let mut context = ExecutionContext::new(workflow);

    let step = make_step(
        "validate-empty-step",
        StepType::Validate,
        HashMap::new(),
        None,
    );
    let result = executor.execute_step(&step, &mut context).await.unwrap();
    assert!(!result.success);
    assert!(result.message.contains("at least one check"));
}

#[tokio::test]
async fn test_execute_validate_missing_path_errors() {
    let executor = StepExecutor::new();
    let workflow = Workflow::new("validate-missing-path", "1.0", "Test missing path");
    let mut context = ExecutionContext::new(workflow);

    let mut params = HashMap::new();
    params.insert(
        "path".to_string(),
        serde_json::json!(unique_temp_path("validate_ghost").display().to_string()),
    );
    let step = make_step("validate-ghost", StepType::Validate, params, None);

    let result = executor.execute_step(&step, &mut context).await.unwrap();
    assert!(!result.success);
    assert!(result.message.contains("does not exist"));
}

#[tokio::test]
async fn test_execute_validate_size_checks() {
    let executor = StepExecutor::new();
    let workflow = Workflow::new("validate-size", "1.0", "Test size bound checks");
    let mut context = ExecutionContext::new(workflow);

    let path = unique_temp_path("validate_size");
    std::fs::write(&path, b"0123456789").unwrap(); // 10 bytes

    let mut ok_params = HashMap::new();
    ok_params.insert(
        "path".to_string(),
        serde_json::json!(path.display().to_string()),
    );
    ok_params.insert("min_size_bytes".to_string(), serde_json::json!(5));
    ok_params.insert("max_size_bytes".to_string(), serde_json::json!(20));
    let ok_step = make_step("validate-size-ok", StepType::Validate, ok_params, None);
    let ok_result = executor.execute_step(&ok_step, &mut context).await.unwrap();
    assert!(ok_result.success, "expected pass: {}", ok_result.message);
    assert_eq!(
        ok_result.output.get("size_bytes"),
        Some(&serde_json::json!(10))
    );

    let mut too_small_params = HashMap::new();
    too_small_params.insert(
        "path".to_string(),
        serde_json::json!(path.display().to_string()),
    );
    too_small_params.insert("min_size_bytes".to_string(), serde_json::json!(100));
    let too_small_step = make_step(
        "validate-too-small",
        StepType::Validate,
        too_small_params,
        None,
    );
    let too_small_result = executor
        .execute_step(&too_small_step, &mut context)
        .await
        .unwrap();
    assert!(!too_small_result.success);
    assert!(too_small_result.message.contains("below"));

    let _ = std::fs::remove_file(&path);
}

#[tokio::test]
async fn test_execute_validate_audio_duration_checks() {
    let executor = StepExecutor::new();
    let workflow = Workflow::new("validate-audio", "1.0", "Test audio decodability/duration");
    let mut context = ExecutionContext::new(workflow);

    let path = unique_temp_path("validate_audio").with_extension("wav");
    {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 16_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(&path, spec).unwrap();
        for _ in 0..16_000 {
            // 1.0 second of silence at 16kHz
            writer.write_sample(0_i16).unwrap();
        }
        writer.finalize().unwrap();
    }

    let mut params = HashMap::new();
    params.insert(
        "path".to_string(),
        serde_json::json!(path.display().to_string()),
    );
    params.insert("min_duration_secs".to_string(), serde_json::json!(0.5));
    params.insert("max_duration_secs".to_string(), serde_json::json!(2.0));
    let step = make_step("validate-audio-ok", StepType::Validate, params, None);
    let result = executor.execute_step(&step, &mut context).await.unwrap();
    assert!(result.success, "expected pass: {}", result.message);
    let reported_duration = result
        .output
        .get("audio_duration_secs")
        .and_then(|v| v.as_f64())
        .unwrap();
    assert!((reported_duration - 1.0).abs() < 0.01);

    let mut too_long_params = HashMap::new();
    too_long_params.insert(
        "path".to_string(),
        serde_json::json!(path.display().to_string()),
    );
    too_long_params.insert("max_duration_secs".to_string(), serde_json::json!(0.1));
    let too_long_step = make_step(
        "validate-audio-too-long",
        StepType::Validate,
        too_long_params,
        None,
    );
    let too_long_result = executor
        .execute_step(&too_long_step, &mut context)
        .await
        .unwrap();
    assert!(!too_long_result.success);
    assert!(too_long_result.message.contains("above"));

    let _ = std::fs::remove_file(&path);
}

#[tokio::test]
async fn test_execute_validate_required_fields() {
    let executor = StepExecutor::new();
    let workflow = Workflow::new("validate-json", "1.0", "Test JSON required_fields checks");
    let mut context = ExecutionContext::new(workflow);

    let path = unique_temp_path("validate_json").with_extension("json");
    std::fs::write(&path, r#"{"name": "voirs", "version": "1.0"}"#).unwrap();

    let mut ok_params = HashMap::new();
    ok_params.insert(
        "path".to_string(),
        serde_json::json!(path.display().to_string()),
    );
    ok_params.insert(
        "required_fields".to_string(),
        serde_json::json!(["name", "version"]),
    );
    let ok_step = make_step("validate-json-ok", StepType::Validate, ok_params, None);
    let ok_result = executor.execute_step(&ok_step, &mut context).await.unwrap();
    assert!(ok_result.success, "expected pass: {}", ok_result.message);

    let mut missing_params = HashMap::new();
    missing_params.insert(
        "path".to_string(),
        serde_json::json!(path.display().to_string()),
    );
    missing_params.insert(
        "required_fields".to_string(),
        serde_json::json!(["name", "nonexistent_field"]),
    );
    let missing_step = make_step(
        "validate-json-missing",
        StepType::Validate,
        missing_params,
        None,
    );
    let missing_result = executor
        .execute_step(&missing_step, &mut context)
        .await
        .unwrap();
    assert!(!missing_result.success);
    assert!(missing_result.message.contains("nonexistent_field"));

    let _ = std::fs::remove_file(&path);
}

#[tokio::test]
async fn test_execute_validate_condition_check() {
    let executor = StepExecutor::new();
    let mut workflow = Workflow::new("validate-condition", "1.0", "Test condition check");
    workflow.add_variable(
        "score".to_string(),
        super::super::definition::Variable::Number(4.5),
    );
    let mut context = ExecutionContext::new(workflow);

    let pass_step = make_step(
        "validate-condition-pass",
        StepType::Validate,
        HashMap::new(),
        Some(Condition::new(
            "${score}".to_string(),
            ConditionOperator::GreaterThan,
            "4.0".to_string(),
        )),
    );
    let pass_result = executor
        .execute_step(&pass_step, &mut context)
        .await
        .unwrap();
    assert!(
        pass_result.success,
        "expected pass: {}",
        pass_result.message
    );

    let fail_step = make_step(
        "validate-condition-fail",
        StepType::Validate,
        HashMap::new(),
        Some(Condition::new(
            "${score}".to_string(),
            ConditionOperator::GreaterThan,
            "10.0".to_string(),
        )),
    );
    let fail_result = executor
        .execute_step(&fail_step, &mut context)
        .await
        .unwrap();
    assert!(!fail_result.success);
}

// -- execute_subworkflow ----------------------------------------------

fn unique_temp_dir(label: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "voirs_executor_test_dir_{label}_{}_{}",
        std::process::id(),
        fastrand::u64(..)
    ))
}

#[tokio::test]
async fn test_execute_subworkflow_runs_child_steps() {
    let dir = unique_temp_dir("subworkflow_basic");
    std::fs::create_dir_all(&dir).unwrap();
    let marker_path = dir.join("marker.txt");
    let child_path = dir.join("child.yaml");

    let child_yaml = format!(
        r#"
metadata:
  name: child-workflow
  version: "1.0"
  description: Child workflow
steps:
  - name: write-marker
    type: fileop
    parameters:
      op: write
      path: "{}"
      content: "child ran for real"
"#,
        marker_path.display().to_string().replace('\\', "\\\\")
    );
    std::fs::write(&child_path, child_yaml).unwrap();

    let executor = StepExecutor::new();
    let workflow = Workflow::new("parent-workflow", "1.0", "Test parent workflow");
    let mut context = ExecutionContext::new(workflow);

    let mut params = HashMap::new();
    params.insert(
        "workflow_file".to_string(),
        serde_json::json!(child_path.display().to_string()),
    );
    let step = make_step("call-child", StepType::Workflow, params, None);

    let result = executor.execute_step(&step, &mut context).await.unwrap();
    assert!(
        result.success,
        "subworkflow step failed: {}",
        result.message
    );
    assert_eq!(
        result.output.get("steps_succeeded"),
        Some(&serde_json::json!(1))
    );
    assert_eq!(
        result.output.get("steps_failed"),
        Some(&serde_json::json!(0))
    );

    // The child workflow's own step must have genuinely executed.
    assert!(marker_path.exists());
    let content = std::fs::read_to_string(&marker_path).unwrap();
    assert_eq!(content, "child ran for real");

    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn test_execute_subworkflow_missing_path_param_errors() {
    let executor = StepExecutor::new();
    let workflow = Workflow::new("parent-missing-path", "1.0", "Test missing workflow_file");
    let mut context = ExecutionContext::new(workflow);

    let step = make_step("call-nothing", StepType::Workflow, HashMap::new(), None);
    let result = executor.execute_step(&step, &mut context).await.unwrap();
    assert!(!result.success);
    assert!(result.message.contains("workflow_file"));
}

#[tokio::test]
async fn test_execute_subworkflow_cycle_detection() {
    let dir = unique_temp_dir("subworkflow_cycle");
    std::fs::create_dir_all(&dir).unwrap();
    let a_path = dir.join("a.yaml");
    let b_path = dir.join("b.yaml");

    // a.yaml calls b.yaml, b.yaml calls a.yaml back -- a genuine cycle
    // that only manifests on the second traversal of a given file.
    let a_yaml = format!(
        r#"
metadata:
  name: workflow-a
  version: "1.0"
  description: A
steps:
  - name: call-b
    type: workflow
    parameters:
      workflow_file: "{}"
"#,
        b_path.display().to_string().replace('\\', "\\\\")
    );
    let b_yaml = format!(
        r#"
metadata:
  name: workflow-b
  version: "1.0"
  description: B
steps:
  - name: call-a
    type: workflow
    parameters:
      workflow_file: "{}"
"#,
        a_path.display().to_string().replace('\\', "\\\\")
    );
    std::fs::write(&a_path, a_yaml).unwrap();
    std::fs::write(&b_path, b_yaml).unwrap();

    let executor = StepExecutor::new();
    let workflow = Workflow::new("cycle-root", "1.0", "Test cycle detection");
    let mut context = ExecutionContext::new(workflow);

    let mut params = HashMap::new();
    params.insert(
        "workflow_file".to_string(),
        serde_json::json!(a_path.display().to_string()),
    );
    let step = make_step("call-a-from-root", StepType::Workflow, params, None);

    let result = executor.execute_step(&step, &mut context).await.unwrap();
    assert!(!result.success);
    assert!(
        result.message.to_lowercase().contains("cycle"),
        "expected a cycle-detection error, got: {}",
        result.message
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn test_execute_subworkflow_inputs_pass_variables() {
    let dir = unique_temp_dir("subworkflow_inputs");
    std::fs::create_dir_all(&dir).unwrap();
    let output_path = dir.join("output.txt");
    let child_path = dir.join("child.yaml");

    let child_yaml = format!(
        r#"
metadata:
  name: child-with-inputs
  version: "1.0"
  description: Child workflow reading a passed-in variable
steps:
  - name: write-greeting
    type: fileop
    parameters:
      op: write
      path: "{}"
      content: "${{greeting}}"
"#,
        output_path.display().to_string().replace('\\', "\\\\")
    );
    std::fs::write(&child_path, child_yaml).unwrap();

    let executor = StepExecutor::new();
    let workflow = Workflow::new("parent-with-inputs", "1.0", "Test inputs pass-through");
    let mut context = ExecutionContext::new(workflow);

    let mut params = HashMap::new();
    params.insert(
        "workflow_file".to_string(),
        serde_json::json!(child_path.display().to_string()),
    );
    params.insert(
        "inputs".to_string(),
        serde_json::json!({"greeting": "hello from parent workflow"}),
    );
    let step = make_step("call-child-with-inputs", StepType::Workflow, params, None);

    let result = executor.execute_step(&step, &mut context).await.unwrap();
    assert!(
        result.success,
        "subworkflow step failed: {}",
        result.message
    );

    let content = std::fs::read_to_string(&output_path).unwrap();
    assert_eq!(content, "hello from parent workflow");

    let _ = std::fs::remove_dir_all(&dir);
}

// -- execute_notify -----------------------------------------------------

#[tokio::test]
async fn test_execute_notify_missing_message_errors() {
    let executor = StepExecutor::new();
    let workflow = Workflow::new("notify-missing-message", "1.0", "Test missing message");
    let mut context = ExecutionContext::new(workflow);

    let step = make_step("notify-no-message", StepType::Notify, HashMap::new(), None);
    let result = executor.execute_step(&step, &mut context).await.unwrap();
    assert!(!result.success);
    assert!(result.message.contains("message"));
}

#[tokio::test]
async fn test_execute_notify_stdout_only_when_no_webhook() {
    let executor = StepExecutor::new();
    let workflow = Workflow::new("notify-stdout-only", "1.0", "Test stdout-only notify");
    let mut context = ExecutionContext::new(workflow);

    let mut params = HashMap::new();
    params.insert(
        "message".to_string(),
        serde_json::json!("stdout-only notification"),
    );
    let step = make_step("notify-stdout", StepType::Notify, params, None);

    let result = executor.execute_step(&step, &mut context).await.unwrap();
    assert!(result.success, "notify step failed: {}", result.message);
    assert_eq!(
        result.output.get("delivered_stdout"),
        Some(&serde_json::json!(true))
    );
    assert!(!result.output.contains_key("delivered_webhook"));
}

#[tokio::test]
async fn test_execute_notify_webhook_delivers_real_payload() {
    voirs_sdk::ensure_crypto_provider();

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    let received: std::sync::Arc<tokio::sync::Mutex<Option<String>>> =
        std::sync::Arc::new(tokio::sync::Mutex::new(None));
    let received_clone = received.clone();

    let server = tokio::spawn(async move {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        if let Ok((mut socket, _)) = listener.accept().await {
            let mut buf = vec![0u8; 8192];
            let n = socket.read(&mut buf).await.unwrap_or(0);
            let request = String::from_utf8_lossy(&buf[..n]).to_string();
            *received_clone.lock().await = Some(request);
            let _ = socket
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
                .await;
        }
    });

    let executor = StepExecutor::new();
    let workflow = Workflow::new("notify-webhook-test", "1.0", "Test notify webhook delivery");
    let mut context = ExecutionContext::new(workflow);

    let mut params = HashMap::new();
    params.insert(
        "message".to_string(),
        serde_json::json!("distinctive-test-message-xyz"),
    );
    params.insert(
        "webhook_url".to_string(),
        serde_json::json!(format!("http://{}/hook", addr)),
    );
    let step = make_step("notify-webhook", StepType::Notify, params, None);

    let result = executor.execute_step(&step, &mut context).await.unwrap();
    server.await.unwrap();

    assert!(result.success, "notify step failed: {}", result.message);
    assert_eq!(
        result.output.get("delivered_webhook"),
        Some(&serde_json::json!(true))
    );

    let request = received
        .lock()
        .await
        .clone()
        .expect("webhook server should have received a request");
    assert!(
        request.starts_with("POST"),
        "expected a POST request, got: {request}"
    );
    assert!(
        request.contains("distinctive-test-message-xyz"),
        "webhook payload should contain the notification message, got: {request}"
    );
}

#[tokio::test]
async fn test_execute_notify_webhook_unreachable_errors() {
    let executor = StepExecutor::new();
    let workflow = Workflow::new(
        "notify-webhook-unreachable",
        "1.0",
        "Test unreachable webhook",
    );
    let mut context = ExecutionContext::new(workflow);

    let mut params = HashMap::new();
    params.insert("message".to_string(), serde_json::json!("nobody home"));
    // Port 1 is not listened on; the connection should be refused
    // immediately rather than hang until the client timeout.
    params.insert(
        "webhook_url".to_string(),
        serde_json::json!("http://127.0.0.1:1/hook"),
    );
    let step = make_step("notify-unreachable", StepType::Notify, params, None);

    let result = executor.execute_step(&step, &mut context).await.unwrap();
    assert!(!result.success);
    assert!(result.message.contains("webhook") || result.message.contains("reach"));
}

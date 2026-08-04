//! Shared helpers for `voirs-cli` subprocess integration tests.
//!
//! Real synthesis requires downloading acoustic/vocoder models from a Hugging
//! Face repository. In CI, offline, or credential-less environments that
//! repository returns `HTTP 401 Unauthorized` to anonymous fetches, so any
//! subprocess test that exercises real synthesis fails there for a purely
//! environmental reason -- not a CLI bug.
//!
//! These helpers recognize that specific, known failure mode and let a test
//! skip (with an explanatory [`eprintln!`]) instead of failing. Any *other*
//! failure reason still fails the test loudly and unchanged -- this must
//! never be used to paper over a genuine regression.

use assert_cmd::Command;
use std::process::Output;

/// Substrings that appear in CLI stdout/stderr when the synthesis pipeline
/// could not download its models because the anonymous Hugging Face fetch
/// was rejected (401) or no models are cached/reachable at all.
///
/// Deliberately narrow: only markers tied specifically to the download path
/// belong here. A generic phrase like "Voice not found" is *not* included --
/// that error class can also indicate a genuine voice-registry inconsistency
/// unrelated to model availability, so tests that hit it must verify the
/// root cause themselves rather than relying on a shared marker that every
/// caller would silently trust (see `test_interactive_mode_responsiveness`
/// in `performance/mod.rs`, which does exactly that verification locally).
const MODEL_UNAVAILABLE_MARKERS: &[&str] = &[
    "No synthesis models available",
    "synthesis models",
    "Download failed",
    "HTTP 401",
];

/// Returns `true` if the combined stdout+stderr of `output` matches a known
/// "synthesis models unavailable" failure mode.
pub fn is_model_unavailable(output: &Output) -> bool {
    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    MODEL_UNAVAILABLE_MARKERS
        .iter()
        .any(|marker| combined.contains(marker))
}

/// Run `cmd` to completion and classify the result.
///
/// - `Some(output)` is returned when the process exited successfully.
/// - `None` is returned when the process failed in a recognized "synthesis
///   models unavailable" way. The caller should treat this as a known skip
///   and return early from the test, leaving any assertions that depend on
///   real synthesis output unexecuted.
/// - Any other failure (one that does not match a known model-unavailability
///   marker) is a genuine test failure: this function panics with the full
///   captured output, mirroring `assert_cmd`'s own `.assert().success()`
///   failure message, so callers don't need to re-check `output.status`
///   themselves.
///
/// This must never be used to weaken a test: any failure that is not a
/// recognized model-unavailability marker still fails the test, with the
/// full captured output attached to the panic message for diagnosis.
pub fn run_or_skip_if_models_unavailable(cmd: &mut Command) -> Option<Output> {
    let output = cmd.output().expect("execute voirs binary");

    if output.status.success() {
        return Some(output);
    }

    if is_model_unavailable(&output) {
        eprintln!(
            "skipping: synthesis models unavailable in this environment (offline or \
             credential-less HuggingFace fetch); stdout/stderr matched a known \
             model-unavailability marker.\nstdout={}\nstderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        return None;
    }

    panic!(
        "command failed for a reason other than known model-unavailability markers\n\
         status={:?}\nstdout={}\nstderr={}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

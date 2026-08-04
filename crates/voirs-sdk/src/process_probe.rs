//! Shared helper for invoking short-lived diagnostic subprocesses (e.g.
//! `nvidia-smi`, `system_profiler`, `ps`) with a hard wall-clock timeout.
//!
//! `std::process::Command::output()` blocks the calling thread until the
//! child process exits, with no way to bound how long that takes. On macOS in
//! particular, `system_profiler` is known to occasionally take tens of
//! seconds (or hang outright) depending on system/driver state, and the same
//! risk applies to any of the other tools this crate shells out to for
//! device/resource probing. Calling one of these directly from a path that is
//! (transitively) reachable from an async `.await` chain - as
//! `VoirsPipelineBuilder::build()` does for device-availability checks - can
//! therefore stall the whole pipeline indefinitely.
//!
//! [`run_with_timeout`] spawns the child normally, then polls
//! [`Child::try_wait`] against a deadline instead of blocking on
//! [`Child::wait`]/[`Command::output`]. If the deadline passes before the
//! child exits, the child is killed and reaped, and the probe is reported as
//! "did not respond in time" (`Ok(None)`) rather than left hanging.

use std::io;
use std::process::{Child, Command, Output, Stdio};
use std::time::{Duration, Instant};

/// Default hard deadline for diagnostic subprocess probes started through
/// [`run_with_timeout`]. Every command this module is used for
/// (`nvidia-smi`, `system_profiler`, `ps`, `sysctl`, ...) normally completes
/// in well under a second; a few seconds leaves generous headroom for a
/// loaded machine while still keeping any hang tightly bounded.
pub(crate) const DEFAULT_PROBE_TIMEOUT: Duration = Duration::from_secs(3);

/// How often to poll the child for completion while waiting.
const POLL_INTERVAL: Duration = Duration::from_millis(20);

/// Run `command`, waiting at most `timeout` for it to complete.
///
/// Returns:
/// - `Ok(Some(output))` if the child exited within the deadline.
/// - `Ok(None)` if the deadline elapsed first. The child is killed and
///   reaped before returning, so no process or thread is ever left blocked
///   waiting on it; callers should treat this the same as "probe failed"
///   (e.g. device not available), never fabricate a success.
/// - `Err(_)` if the child could not even be spawned (e.g. binary missing).
pub(crate) fn run_with_timeout(
    command: &mut Command,
    timeout: Duration,
) -> io::Result<Option<Output>> {
    let mut child: Child = command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;

    let deadline = Instant::now() + timeout;
    loop {
        match child.try_wait()? {
            Some(_status) => return child.wait_with_output().map(Some),
            None => {
                if Instant::now() >= deadline {
                    // Best-effort kill + reap; ignore errors since we are
                    // giving up on this probe regardless of the outcome.
                    let _ = child.kill();
                    let _ = child.wait();
                    return Ok(None);
                }
                std::thread::sleep(POLL_INTERVAL);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn completes_fast_command_within_timeout() {
        let mut cmd = Command::new("echo");
        cmd.arg("hi");
        let result = run_with_timeout(&mut cmd, Duration::from_secs(3))
            .expect("spawn should succeed")
            .expect("echo should complete well within the timeout");
        assert!(result.status.success());
        assert_eq!(String::from_utf8_lossy(&result.stdout).trim(), "hi");
    }

    #[test]
    fn kills_and_reports_none_on_timeout() {
        let mut cmd = Command::new("sleep");
        cmd.arg("30");
        let start = Instant::now();
        let result =
            run_with_timeout(&mut cmd, Duration::from_millis(200)).expect("spawn should succeed");
        assert!(result.is_none(), "sleep 30 must be treated as timed out");
        assert!(
            start.elapsed() < Duration::from_secs(2),
            "run_with_timeout must return promptly after killing the child, took {:?}",
            start.elapsed()
        );
    }

    #[test]
    fn spawn_error_is_propagated() {
        let mut cmd = Command::new("voirs-sdk-nonexistent-probe-binary-xyz");
        let result = run_with_timeout(&mut cmd, Duration::from_secs(1));
        assert!(result.is_err());
    }
}

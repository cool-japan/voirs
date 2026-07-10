//! Pure, platform-independent parsing helpers for VoiRS FFI platform queries.
//!
//! Every function in this module is a pure function over already-captured text
//! (typically the stdout of a shell command, or the contents of a `/proc`
//! pseudo-file) with **no** file or process I/O of its own. The OS-specific
//! modules ([`super::linux`], [`super::macos`], [`super::windows`]) are only
//! compiled on their respective target OS (see the `#[cfg(target_os = "...")]`
//! guards on the `mod` declarations in [`super`]), so keeping the parsing
//! logic here -- in a module that is *always* compiled -- lets it be
//! unit-tested against fixed sample strings on any host, regardless of which
//! OS is running the test suite.

/// A single sound card entry parsed from `/proc/asound/cards`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AsoundCardEntry {
    /// ALSA card index (e.g. `0`).
    pub index: u32,
    /// Short card identifier shown in brackets (e.g. `PCH`).
    pub id: String,
    /// Kernel driver name (e.g. `HDA-Intel`).
    pub driver: String,
    /// Human-readable long name (e.g. `HDA Intel PCH`).
    pub long_name: String,
}

/// Parse the contents of `/proc/asound/cards` into a list of card entries.
///
/// Sample input:
/// ```text
///  0 [PCH            ]: HDA-Intel - HDA Intel PCH
///                       HDA Intel PCH at 0xef240000 irq 32
///  1 [NVidia          ]: HDA-Intel - HDA NVidia
///                       HDA NVidia at 0xf0080000 irq 17
/// ```
/// Lines are alternating "header" (card index + `[id]: driver - long name`)
/// and indented continuation/description lines; only header lines are parsed.
pub fn parse_asound_cards(contents: &str) -> Vec<AsoundCardEntry> {
    let mut cards = Vec::new();

    for line in contents.lines() {
        let trimmed = line.trim_start();
        if trimmed.is_empty() {
            continue;
        }

        // Header lines start (after leading whitespace) with the card index
        // digit; indented continuation lines describing the hardware do not.
        if !trimmed.chars().next().is_some_and(|c| c.is_ascii_digit()) {
            continue;
        }

        let Some(bracket_start) = trimmed.find('[') else {
            continue;
        };
        let Some(bracket_end) = trimmed.find(']') else {
            continue;
        };
        if bracket_end <= bracket_start {
            continue;
        }

        let Ok(index) = trimmed[..bracket_start].trim().parse::<u32>() else {
            continue;
        };
        let id = trimmed[bracket_start + 1..bracket_end].trim().to_string();
        let rest = trimmed[bracket_end + 1..]
            .trim()
            .trim_start_matches(':')
            .trim();

        let (driver, long_name) = match rest.split_once(" - ") {
            Some((d, n)) => (d.trim().to_string(), n.trim().to_string()),
            None => (rest.to_string(), String::new()),
        };

        cards.push(AsoundCardEntry {
            index,
            id,
            driver,
            long_name,
        });
    }

    cards
}

/// Parse the `name:` field out of a `/proc/asound/card{N}/pcm{M}{p,c}/info` file.
///
/// Sample input:
/// ```text
/// card: 0
/// device: 0
/// subdevice: 0
/// stream: PLAYBACK
/// id: ALC295 Analog
/// name: ALC295 Analog
/// subname: subdevice #0
/// ```
pub fn parse_pcm_info_name(contents: &str) -> Option<String> {
    for line in contents.lines() {
        if let Some((key, value)) = line.split_once(':') {
            if key.trim() == "name" {
                let value = value.trim();
                if !value.is_empty() {
                    return Some(value.to_string());
                }
            }
        }
    }
    None
}

/// Real-time-negotiated ALSA hardware parameters, as much as can be honestly
/// determined without linking `libasound` (see [`parse_alsa_hw_params`]).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AlsaHwParamsSnapshot {
    /// Sample rate(s) currently negotiated (usually a single value, since
    /// this reflects a live stream rather than the full supported range).
    pub sample_rates: Vec<u32>,
    /// Sample format(s) currently negotiated (e.g. `S16_LE`).
    pub formats: Vec<String>,
    /// Channel count(s) currently negotiated.
    pub channels: Vec<u32>,
    /// Buffer size(s) currently negotiated, in frames.
    pub buffer_sizes: Vec<u32>,
}

/// Parse `/proc/asound/card{N}/pcm{M}{p,c}/sub0/hw_params`.
///
/// This file only reports parameters for a PCM substream that is *currently
/// open*; when nothing has the device open it contains the literal text
/// `closed`, in which case an honestly-empty snapshot is returned (never
/// fabricated placeholder values).
///
/// Sample "open" input:
/// ```text
/// access: RW_INTERLEAVED
/// format: S16_LE
/// subformat: STD
/// channels: 2
/// rate: 44100 (44100/1)
/// period_size: 4410
/// period_time: 100000 (99999/1)
/// buffer_size: 17640
/// buffer_time: 400000 (399999/1)
/// ```
pub fn parse_alsa_hw_params(contents: &str) -> AlsaHwParamsSnapshot {
    let mut snapshot = AlsaHwParamsSnapshot::default();

    if contents.trim() == "closed" {
        return snapshot;
    }

    for line in contents.lines() {
        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        let value = value.trim();

        match key.trim() {
            "format" => snapshot.formats.push(value.to_string()),
            "channels" => {
                if let Ok(channels) = value.parse::<u32>() {
                    snapshot.channels.push(channels);
                }
            }
            "rate" => {
                // "44100 (44100/1)" -> take the first whitespace-delimited token.
                if let Some(rate) = value
                    .split_whitespace()
                    .next()
                    .and_then(|tok| tok.parse::<u32>().ok())
                {
                    snapshot.sample_rates.push(rate);
                }
            }
            "buffer_size" => {
                if let Ok(buffer_size) = value.parse::<u32>() {
                    snapshot.buffer_sizes.push(buffer_size);
                }
            }
            _ => {}
        }
    }

    snapshot
}

/// Fields of interest parsed out of `/proc/meminfo`, in kilobytes (as reported
/// by the kernel).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MemInfo {
    /// `MemTotal` in kB.
    pub mem_total_kb: u64,
    /// `MemAvailable` in kB.
    pub mem_available_kb: u64,
    /// `SwapTotal` in kB.
    pub swap_total_kb: u64,
    /// `SwapFree` in kB.
    pub swap_free_kb: u64,
}

/// Parse `/proc/meminfo` contents into [`MemInfo`]. Missing/unparseable
/// fields are left at `0` rather than a fabricated value.
pub fn parse_meminfo(contents: &str) -> MemInfo {
    let mut info = MemInfo::default();

    for line in contents.lines() {
        let Some((key, rest)) = line.split_once(':') else {
            continue;
        };
        let Some(value) = rest
            .split_whitespace()
            .next()
            .and_then(|tok| tok.parse::<u64>().ok())
        else {
            continue;
        };

        match key.trim() {
            "MemTotal" => info.mem_total_kb = value,
            "MemAvailable" => info.mem_available_kb = value,
            "SwapTotal" => info.swap_total_kb = value,
            "SwapFree" => info.swap_free_kb = value,
            _ => {}
        }
    }

    info
}

/// Parse the three space-separated load-average fields from `/proc/loadavg`
/// (e.g. `"0.52 0.58 0.59 1/532 12345"`) into `(1m, 5m, 15m)`.
pub fn parse_loadavg(contents: &str) -> Option<(f32, f32, f32)> {
    let mut parts = contents.split_whitespace();
    let one = parts.next()?.parse::<f32>().ok()?;
    let five = parts.next()?.parse::<f32>().ok()?;
    let fifteen = parts.next()?.parse::<f32>().ok()?;
    Some((one, five, fifteen))
}

/// Raw CPU tick counters parsed from the aggregate `cpu` line of `/proc/stat`
/// (all fields are in USER_HZ ticks since boot).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CpuTicks {
    pub user: u64,
    pub nice: u64,
    pub system: u64,
    pub idle: u64,
    pub iowait: u64,
    pub irq: u64,
    pub softirq: u64,
    pub steal: u64,
}

impl CpuTicks {
    /// Sum of all tick counters (the denominator for utilization math).
    pub fn total(&self) -> u64 {
        self.user
            + self.nice
            + self.system
            + self.idle
            + self.iowait
            + self.irq
            + self.softirq
            + self.steal
    }
}

/// Parse the aggregate `cpu ` summary line out of `/proc/stat` contents.
///
/// Sample input:
/// ```text
/// cpu  132453 322 34567 4324545 2345 0 234 0 0 0
/// cpu0 66000 100 17000 2162000 1000 0 100 0 0 0
/// ```
/// Requires at least `user nice system idle` to be present; trailing fields
/// (`iowait irq softirq steal`, present on modern kernels) default to `0`
/// when absent rather than failing the whole parse.
pub fn parse_proc_stat_cpu_line(contents: &str) -> Option<CpuTicks> {
    let line = contents.lines().find(|l| l.starts_with("cpu "))?;
    let fields: Vec<u64> = line
        .split_whitespace()
        .skip(1)
        .map(|tok| tok.parse::<u64>().unwrap_or(0))
        .collect();

    Some(CpuTicks {
        user: *fields.first()?,
        nice: *fields.get(1)?,
        system: *fields.get(2)?,
        idle: *fields.get(3)?,
        iowait: fields.get(4).copied().unwrap_or(0),
        irq: fields.get(5).copied().unwrap_or(0),
        softirq: fields.get(6).copied().unwrap_or(0),
        steal: fields.get(7).copied().unwrap_or(0),
    })
}

/// Compute CPU utilization as a percentage (0.0-100.0) between two
/// `/proc/stat` samples taken a short interval apart.
pub fn cpu_usage_percent(prev: CpuTicks, curr: CpuTicks) -> f32 {
    let total_delta = curr.total().saturating_sub(prev.total());
    if total_delta == 0 {
        return 0.0;
    }
    let idle_delta = curr.idle.saturating_sub(prev.idle);
    let busy_delta = total_delta.saturating_sub(idle_delta);
    (busy_delta as f64 / total_delta as f64 * 100.0) as f32
}

/// A single logical audio device parsed from `system_profiler SPAudioDataType
/// -json` output.
#[derive(Debug, Clone, PartialEq)]
pub struct SystemProfilerAudioDevice {
    /// Index of the underlying `_items` entry this was derived from (a
    /// single hardware item can yield both an input and an output device).
    pub index: u32,
    pub name: String,
    pub is_default: bool,
    pub sample_rate: f64,
    pub channels: u32,
    pub is_input: bool,
}

/// Parse `system_profiler SPAudioDataType -json` output into a flat device
/// list. Devices that are both input- and output-capable produce two
/// entries (mirroring the shape of the existing cpal-based enumeration).
///
/// Returns an empty `Vec` (never fabricated devices) if the JSON cannot be
/// parsed or does not contain the expected structure.
pub fn parse_system_profiler_audio(json_str: &str) -> Vec<SystemProfilerAudioDevice> {
    let mut devices = Vec::new();

    let Ok(root) = serde_json::from_str::<serde_json::Value>(json_str) else {
        return devices;
    };

    let items = root
        .get("SPAudioDataType")
        .and_then(|v| v.as_array())
        .and_then(|arr| arr.first())
        .and_then(|entry| entry.get("_items"))
        .and_then(|v| v.as_array());

    let Some(items) = items else {
        return devices;
    };

    for (index, item) in items.iter().enumerate() {
        let index = index as u32;
        let name = item
            .get("_name")
            .and_then(|v| v.as_str())
            .unwrap_or("Unknown Device")
            .to_string();
        let sample_rate = item
            .get("coreaudio_device_srate")
            .and_then(|v| v.as_f64())
            .unwrap_or(44100.0);

        if let Some(channels) = item.get("coreaudio_device_output").and_then(|v| v.as_u64()) {
            let is_default = item
                .get("coreaudio_default_audio_output_device")
                .and_then(|v| v.as_str())
                == Some("spaudio_yes");
            devices.push(SystemProfilerAudioDevice {
                index,
                name: name.clone(),
                is_default,
                sample_rate,
                channels: channels as u32,
                is_input: false,
            });
        }

        if let Some(channels) = item.get("coreaudio_device_input").and_then(|v| v.as_u64()) {
            let is_default = item
                .get("coreaudio_default_audio_input_device")
                .and_then(|v| v.as_str())
                == Some("spaudio_yes");
            devices.push(SystemProfilerAudioDevice {
                index,
                name,
                is_default,
                sample_rate,
                channels: channels as u32,
                is_input: true,
            });
        }
    }

    devices
}

/// Parse the stdout of `osascript -e 'output volume of (get volume
/// settings)'` (an integer 0-100 as text, e.g. `"69"`) into a 0.0-1.0
/// fraction. Unparseable input honestly falls back to `0.0` rather than a
/// fabricated "reasonable-looking" value.
pub fn parse_volume_output(raw: &str) -> f32 {
    raw.trim()
        .parse::<f32>()
        .map(|v| (v / 100.0).clamp(0.0, 1.0))
        .unwrap_or(0.0)
}

/// Normalize the stdout of `defaults read -g AppleLocale` (e.g. `"ja_JP"`)
/// into a hyphenated locale tag (e.g. `"ja-JP"`). Returns `None` for empty
/// input so the caller can honestly report an error instead of fabricating a
/// default locale.
pub fn parse_locale(raw: &str) -> Option<String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }
    Some(trimmed.replace('_', "-"))
}

/// Interpret the stdout of `defaults read -g AppleInterfaceStyle`.
///
/// `Some("Dark")` (command succeeded, key set) means dark mode; a failed
/// command (`None`, the caller passes this when the process exits non-zero)
/// means the key is absent, which is exactly how macOS represents "Light
/// mode" -- so `None` legitimately maps to `"light"`, not an error.
pub fn parse_appearance(raw: Option<&str>) -> &'static str {
    match raw {
        Some(s) if s.trim().eq_ignore_ascii_case("dark") => "dark",
        _ => "light",
    }
}

/// Normalize a raw `kern.memorystatus_vm_pressure_level` sysctl value
/// (XNU: `1`=normal, `2`=warn, `4`=critical) to a `0.0`-`1.0` pressure
/// fraction, preserving the original fake-value's `f32` 0.0-1.0 contract
/// while now reflecting a real kernel reading.
pub fn normalize_memory_pressure_level(raw: u32) -> f32 {
    match raw {
        0 | 1 => 0.0,
        2 => 0.5,
        4 => 1.0,
        other => (other as f32 / 4.0).clamp(0.0, 1.0),
    }
}

/// Parse `pmset -g therm` output into a coarse thermal state string.
///
/// When the system reports a `CPU_Speed_Limit` / `CPU_Scheduler_Limit`
/// percentage below 100, the system is thermally throttling; when nothing of
/// the sort has ever been recorded (the common case, e.g.
/// `"Note: No thermal warning level has been recorded"`), that honestly
/// means "nominal" -- no warning has ever fired.
pub fn parse_thermal_state(output: &str) -> String {
    for line in output.lines() {
        let rest = line
            .split_once("CPU_Speed_Limit")
            .or_else(|| line.split_once("CPU_Scheduler_Limit"))
            .map(|(_, r)| r);

        if let Some(rest) = rest {
            let value = rest
                .trim_start_matches([' ', '='])
                .split_whitespace()
                .next();
            if let Some(percent) = value.and_then(|v| v.parse::<u32>().ok()) {
                return if percent >= 100 {
                    "nominal".to_string()
                } else if percent >= 50 {
                    "throttled".to_string()
                } else {
                    "critical".to_string()
                };
            }
        }
    }

    "nominal".to_string()
}

/// Parse `pmset -g batt` output's first line (`"Now drawing from 'AC
/// Power'"` / `"Now drawing from 'Battery Power'"`) into a power-state tag.
pub fn parse_power_state(output: &str) -> String {
    let first_line = output.lines().next().unwrap_or("");
    if first_line.contains("AC Power") {
        "ac_power".to_string()
    } else if first_line.contains("Battery Power") {
        "battery".to_string()
    } else {
        "unknown".to_string()
    }
}

/// Sum the per-process percentages from `ps -A -o %cpu=` output (one float
/// per line) into an aggregate CPU-percent figure. Lines that fail to parse
/// are skipped (not counted as `0` fabrication, simply ignored).
pub fn parse_ps_cpu_output(output: &str) -> f32 {
    output
        .lines()
        .filter_map(|line| line.trim().parse::<f32>().ok())
        .sum()
}

/// Compute CPU utilization (0.0-100.0) from a Windows `GetSystemTimes` delta.
///
/// Per the Win32 API contract, `lpKernelTime` **includes** idle time, so
/// total non-idle (busy) ticks are `(kernel_delta + user_delta) -
/// idle_delta`. Pure integer/float math -- no FFI -- so it is unit-testable
/// on any host even though `GetSystemTimes` itself only exists on Windows.
pub fn cpu_usage_from_ticks(idle_delta: u64, kernel_delta: u64, user_delta: u64) -> f32 {
    let total_delta = kernel_delta.saturating_add(user_delta);
    if total_delta == 0 {
        return 0.0;
    }
    let busy_delta = total_delta.saturating_sub(idle_delta);
    (busy_delta as f64 / total_delta as f64 * 100.0) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- parse_asound_cards -------------------------------------------------

    const SAMPLE_ASOUND_CARDS: &str = " 0 [PCH            ]: HDA-Intel - HDA Intel PCH\n                      HDA Intel PCH at 0xef240000 irq 32\n 1 [NVidia          ]: HDA-Intel - HDA NVidia\n                      HDA NVidia at 0xf0080000 irq 17\n";

    #[test]
    fn test_parse_asound_cards_exact_entries() {
        let cards = parse_asound_cards(SAMPLE_ASOUND_CARDS);
        assert_eq!(
            cards,
            vec![
                AsoundCardEntry {
                    index: 0,
                    id: "PCH".to_string(),
                    driver: "HDA-Intel".to_string(),
                    long_name: "HDA Intel PCH".to_string(),
                },
                AsoundCardEntry {
                    index: 1,
                    id: "NVidia".to_string(),
                    driver: "HDA-Intel".to_string(),
                    long_name: "HDA NVidia".to_string(),
                },
            ]
        );
    }

    #[test]
    fn test_parse_asound_cards_empty_input() {
        assert!(parse_asound_cards("").is_empty());
        assert!(parse_asound_cards("   \n   \n").is_empty());
    }

    #[test]
    fn test_parse_asound_cards_ignores_malformed_lines() {
        let malformed = "not a card line\n 0 missing brackets\n";
        assert!(parse_asound_cards(malformed).is_empty());
    }

    // ---- parse_pcm_info_name -------------------------------------------------

    #[test]
    fn test_parse_pcm_info_name() {
        let info = "card: 0\ndevice: 0\nsubdevice: 0\nstream: PLAYBACK\nid: ALC295 Analog\nname: ALC295 Analog\nsubname: subdevice #0\n";
        assert_eq!(parse_pcm_info_name(info), Some("ALC295 Analog".to_string()));
    }

    #[test]
    fn test_parse_pcm_info_name_missing() {
        assert_eq!(parse_pcm_info_name("card: 0\ndevice: 0\n"), None);
    }

    // ---- parse_alsa_hw_params --------------------------------------------------

    #[test]
    fn test_parse_alsa_hw_params_open_stream() {
        let contents = "access: RW_INTERLEAVED\nformat: S16_LE\nsubformat: STD\nchannels: 2\nrate: 44100 (44100/1)\nperiod_size: 4410\nperiod_time: 100000 (99999/1)\nbuffer_size: 17640\nbuffer_time: 400000 (399999/1)\n";
        let snapshot = parse_alsa_hw_params(contents);
        assert_eq!(snapshot.formats, vec!["S16_LE".to_string()]);
        assert_eq!(snapshot.channels, vec![2]);
        assert_eq!(snapshot.sample_rates, vec![44100]);
        assert_eq!(snapshot.buffer_sizes, vec![17640]);
    }

    #[test]
    fn test_parse_alsa_hw_params_closed_stream_is_honestly_empty() {
        let snapshot = parse_alsa_hw_params("closed\n");
        assert_eq!(snapshot, AlsaHwParamsSnapshot::default());
        assert!(snapshot.sample_rates.is_empty());
        assert!(snapshot.formats.is_empty());
        assert!(snapshot.channels.is_empty());
        assert!(snapshot.buffer_sizes.is_empty());
    }

    // ---- parse_meminfo -------------------------------------------------------

    const SAMPLE_MEMINFO: &str = "MemTotal:       16384000 kB\nMemFree:         1234000 kB\nMemAvailable:    8000000 kB\nBuffers:          234000 kB\nCached:          3456000 kB\nSwapCached:            0 kB\nSwapTotal:       2097152 kB\nSwapFree:        2097152 kB\n";

    #[test]
    fn test_parse_meminfo() {
        let info = parse_meminfo(SAMPLE_MEMINFO);
        assert_eq!(info.mem_total_kb, 16_384_000);
        assert_eq!(info.mem_available_kb, 8_000_000);
        assert_eq!(info.swap_total_kb, 2_097_152);
        assert_eq!(info.swap_free_kb, 2_097_152);
    }

    #[test]
    fn test_parse_meminfo_empty_input_is_all_zero() {
        assert_eq!(parse_meminfo(""), MemInfo::default());
    }

    // ---- parse_loadavg -------------------------------------------------------

    #[test]
    fn test_parse_loadavg() {
        assert_eq!(
            parse_loadavg("0.52 0.58 0.59 1/532 12345"),
            Some((0.52, 0.58, 0.59))
        );
    }

    #[test]
    fn test_parse_loadavg_malformed() {
        assert_eq!(parse_loadavg(""), None);
        assert_eq!(parse_loadavg("not numbers here"), None);
    }

    // ---- parse_proc_stat_cpu_line / cpu_usage_percent -------------------------

    const SAMPLE_PROC_STAT: &str = "cpu  132453 322 34567 4324545 2345 0 234 0 0 0\ncpu0 66000 100 17000 2162000 1000 0 100 0 0 0\nintr 1234567 0 0 0\nctxt 9876543\n";

    #[test]
    fn test_parse_proc_stat_cpu_line() {
        let ticks =
            parse_proc_stat_cpu_line(SAMPLE_PROC_STAT).expect("cpu line present in fixture");
        assert_eq!(
            ticks,
            CpuTicks {
                user: 132_453,
                nice: 322,
                system: 34_567,
                idle: 4_324_545,
                iowait: 2_345,
                irq: 0,
                softirq: 234,
                steal: 0,
            }
        );
    }

    #[test]
    fn test_parse_proc_stat_cpu_line_missing() {
        assert_eq!(parse_proc_stat_cpu_line("intr 12345\nctxt 6789\n"), None);
    }

    #[test]
    fn test_cpu_usage_percent_50_percent_busy() {
        let prev = CpuTicks {
            user: 100,
            idle: 100,
            ..Default::default()
        };
        let curr = CpuTicks {
            user: 200,
            idle: 200,
            ..Default::default()
        };
        // delta: user +100, idle +100 => 50% busy.
        let usage = cpu_usage_percent(prev, curr);
        assert!((usage - 50.0).abs() < 1e-4, "expected ~50.0, got {usage}");
    }

    #[test]
    fn test_cpu_usage_percent_no_time_elapsed_is_zero() {
        let sample = CpuTicks {
            user: 100,
            idle: 100,
            ..Default::default()
        };
        assert_eq!(cpu_usage_percent(sample, sample), 0.0);
    }

    #[test]
    fn test_cpu_usage_percent_fully_idle() {
        let prev = CpuTicks {
            idle: 1000,
            ..Default::default()
        };
        let curr = CpuTicks {
            idle: 2000,
            ..Default::default()
        };
        assert_eq!(cpu_usage_percent(prev, curr), 0.0);
    }

    #[test]
    fn test_cpu_usage_percent_fully_busy() {
        let prev = CpuTicks {
            user: 0,
            idle: 1000,
            ..Default::default()
        };
        let curr = CpuTicks {
            user: 1000,
            idle: 1000,
            ..Default::default()
        };
        let usage = cpu_usage_percent(prev, curr);
        assert!((usage - 100.0).abs() < 1e-4, "expected ~100.0, got {usage}");
    }

    // ---- parse_system_profiler_audio ------------------------------------------

    // Real `system_profiler SPAudioDataType -json` output captured on a
    // MacBook Pro (Apple Silicon) development host.
    const SAMPLE_SYSTEM_PROFILER_AUDIO: &str = r#"{
  "SPAudioDataType" : [
    {
      "_items" : [
        {
          "_name" : "Built-in Microphone",
          "coreaudio_default_audio_input_device" : "spaudio_yes",
          "coreaudio_device_input" : 1,
          "coreaudio_device_manufacturer" : "Apple Inc.",
          "coreaudio_device_srate" : 48000,
          "coreaudio_device_transport" : "coreaudio_device_type_builtin",
          "coreaudio_input_source" : "Built-in Microphone"
        },
        {
          "_name" : "Built-in Speakers",
          "_properties" : "coreaudio_default_audio_system_device",
          "coreaudio_default_audio_output_device" : "spaudio_yes",
          "coreaudio_default_audio_system_device" : "spaudio_yes",
          "coreaudio_device_manufacturer" : "Apple Inc.",
          "coreaudio_device_output" : 2,
          "coreaudio_device_srate" : 48000,
          "coreaudio_device_transport" : "coreaudio_device_type_builtin",
          "coreaudio_output_source" : "Built-in Speakers"
        }
      ],
      "_name" : "coreaudio_device"
    }
  ]
}"#;

    #[test]
    fn test_parse_system_profiler_audio() {
        let devices = parse_system_profiler_audio(SAMPLE_SYSTEM_PROFILER_AUDIO);
        assert_eq!(devices.len(), 2, "one input + one output device expected");

        let input = devices
            .iter()
            .find(|d| d.is_input)
            .expect("input device present");
        assert_eq!(input.name, "Built-in Microphone");
        assert_eq!(input.channels, 1);
        assert_eq!(input.sample_rate, 48000.0);
        assert!(input.is_default);

        let output = devices
            .iter()
            .find(|d| !d.is_input)
            .expect("output device present");
        assert_eq!(output.name, "Built-in Speakers");
        assert_eq!(output.channels, 2);
        assert_eq!(output.sample_rate, 48000.0);
        assert!(output.is_default);
    }

    #[test]
    fn test_parse_system_profiler_audio_malformed_json_is_honestly_empty() {
        assert!(parse_system_profiler_audio("not json").is_empty());
        assert!(parse_system_profiler_audio("{}").is_empty());
    }

    // ---- parse_volume_output --------------------------------------------------

    #[test]
    fn test_parse_volume_output() {
        assert_eq!(parse_volume_output("69"), 0.69);
        assert_eq!(parse_volume_output("0"), 0.0);
        assert_eq!(parse_volume_output("100"), 1.0);
        assert_eq!(parse_volume_output("  42\n"), 0.42);
    }

    #[test]
    fn test_parse_volume_output_clamps_out_of_range() {
        assert_eq!(parse_volume_output("150"), 1.0);
        assert_eq!(parse_volume_output("-10"), 0.0);
    }

    #[test]
    fn test_parse_volume_output_malformed_is_honest_zero_not_fabricated() {
        assert_eq!(parse_volume_output("not a number"), 0.0);
    }

    // ---- parse_locale ----------------------------------------------------------

    #[test]
    fn test_parse_locale_normalizes_underscore_to_hyphen() {
        assert_eq!(parse_locale("ja_JP"), Some("ja-JP".to_string()));
        assert_eq!(parse_locale("en_US\n"), Some("en-US".to_string()));
    }

    #[test]
    fn test_parse_locale_empty_is_none() {
        assert_eq!(parse_locale(""), None);
        assert_eq!(parse_locale("   \n"), None);
    }

    // ---- parse_appearance -------------------------------------------------------

    #[test]
    fn test_parse_appearance_dark() {
        assert_eq!(parse_appearance(Some("Dark")), "dark");
        assert_eq!(parse_appearance(Some("dark\n")), "dark");
    }

    #[test]
    fn test_parse_appearance_absent_or_err_means_light() {
        assert_eq!(parse_appearance(None), "light");
        assert_eq!(parse_appearance(Some("")), "light");
    }

    // ---- normalize_memory_pressure_level ---------------------------------------

    #[test]
    fn test_normalize_memory_pressure_level_known_xnu_values() {
        assert_eq!(normalize_memory_pressure_level(1), 0.0); // normal
        assert_eq!(normalize_memory_pressure_level(2), 0.5); // warn
        assert_eq!(normalize_memory_pressure_level(4), 1.0); // critical
    }

    #[test]
    fn test_normalize_memory_pressure_level_clamps_unknown_values() {
        assert_eq!(normalize_memory_pressure_level(0), 0.0);
        assert!(normalize_memory_pressure_level(100) <= 1.0);
    }

    // ---- parse_thermal_state ---------------------------------------------------

    #[test]
    fn test_parse_thermal_state_no_warning_recorded_is_nominal() {
        // Real `pmset -g therm` output captured on a development host that has
        // never hit a thermal limit.
        let output = "Note: No thermal warning level has been recorded\nNote: No performance warning level has been recorded\nNote: No CPU power status has been recorded\n";
        assert_eq!(parse_thermal_state(output), "nominal");
    }

    #[test]
    fn test_parse_thermal_state_throttled() {
        assert_eq!(parse_thermal_state("CPU_Speed_Limit = 50"), "throttled");
    }

    #[test]
    fn test_parse_thermal_state_critical() {
        assert_eq!(parse_thermal_state("CPU_Speed_Limit = 10"), "critical");
    }

    #[test]
    fn test_parse_thermal_state_full_speed_is_nominal() {
        assert_eq!(parse_thermal_state("CPU_Speed_Limit = 100"), "nominal");
    }

    // ---- parse_power_state -------------------------------------------------------

    #[test]
    fn test_parse_power_state_ac() {
        let output = "Now drawing from 'AC Power'\n -InternalBattery-0 (id=27132003)\t100%; charged; 0:00 remaining present: true\n";
        assert_eq!(parse_power_state(output), "ac_power");
    }

    #[test]
    fn test_parse_power_state_battery() {
        let output = "Now drawing from 'Battery Power'\n -InternalBattery-0 (id=27132003)\t80%; discharging; 3:00 remaining present: true\n";
        assert_eq!(parse_power_state(output), "battery");
    }

    #[test]
    fn test_parse_power_state_unknown() {
        assert_eq!(parse_power_state(""), "unknown");
    }

    // ---- parse_ps_cpu_output ----------------------------------------------------

    #[test]
    fn test_parse_ps_cpu_output_sums_lines() {
        let output = "  0.5\n  0.4\n  1.6\n  0.0\n";
        let total = parse_ps_cpu_output(output);
        assert!((total - 2.5).abs() < 1e-4, "expected ~2.5, got {total}");
    }

    #[test]
    fn test_parse_ps_cpu_output_empty() {
        assert_eq!(parse_ps_cpu_output(""), 0.0);
    }

    // ---- cpu_usage_from_ticks (Windows GetSystemTimes math) ---------------------

    #[test]
    fn test_cpu_usage_from_ticks_50_percent() {
        // kernel_delta includes idle_delta per the Win32 API contract.
        let usage = cpu_usage_from_ticks(500, 800, 200);
        assert!((usage - 50.0).abs() < 1e-4, "expected ~50.0, got {usage}");
    }

    #[test]
    fn test_cpu_usage_from_ticks_fully_idle() {
        assert_eq!(cpu_usage_from_ticks(1000, 1000, 0), 0.0);
    }

    #[test]
    fn test_cpu_usage_from_ticks_no_time_elapsed() {
        assert_eq!(cpu_usage_from_ticks(0, 0, 0), 0.0);
    }
}

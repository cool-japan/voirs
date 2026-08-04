//! Environment sensor implementations for [`super::AdaptiveAcousticEnvironment`].
//!
//! Split out of `adaptive_acoustics.rs` to keep that file under the
//! project's file-size guideline; the sensor *struct definitions* stay in
//! the parent module (many other types there reference them), while their
//! real update logic lives here.
//!
//! - [`super::TemperatureSensor`] / [`super::HumiditySensor`]: no in-process
//!   data source exists for ambient temperature/humidity, so these only
//!   record real externally-supplied readings (see
//!   `TemperatureSensor::record_reading` / `HumiditySensor::record_reading`)
//!   rather than fabricating a random walk.
//! - [`super::NoiseSensor`] / [`super::OccupancySensor`]: derive real
//!   readings from an audio buffer via FFT-based spectral analysis and
//!   RMS/zero-crossing-rate activity detection, respectively (see
//!   `update_from_audio` on each).
//! - [`super::MaterialDetector`] / [`super::AcousticProbe`]: real
//!   acoustic-analysis-based re-detection/probing is not implemented yet
//!   (see their `update` doc comments); they intentionally do not fabricate
//!   new readings on every tick.

use super::*;

/// Real external sensor inputs for one [`AdaptiveAcousticEnvironment::update`]
/// tick.
///
/// Every field is optional: whichever readings the caller actually has this
/// tick (from real hardware, a captured audio buffer, ...) are applied.
/// Omitted fields leave that sensor at its last known reading instead of
/// fabricating a new one - this crate has no in-process source of ambient
/// temperature or humidity, so those two *must* be supplied externally to
/// change at all.
#[derive(Debug, Clone, Copy, Default)]
pub struct SensorInputs<'a> {
    /// Real ambient temperature reading in Celsius, if available.
    pub temperature_celsius: Option<f32>,
    /// Real relative-humidity reading (0.0-1.0), if available.
    pub relative_humidity: Option<f32>,
    /// A real captured audio buffer to analyze for noise level/spectrum and
    /// voice-activity-based occupancy (see [`NoiseSensor::update_from_audio`]
    /// and [`OccupancySensor::update_from_audio`]). Required alongside
    /// `sample_rate` to update either sensor.
    pub audio_buffer: Option<&'a [f32]>,
    /// Sample rate of `audio_buffer` in Hz; ignored if `audio_buffer` is `None`.
    pub sample_rate: u32,
}

impl EnvironmentSensors {
    pub(super) fn new(_config: &SensorConfig) -> Result<Self> {
        Ok(Self {
            temperature_sensor: TemperatureSensor::new(),
            humidity_sensor: HumiditySensor::new(),
            noise_sensor: NoiseSensor::new(),
            occupancy_sensor: OccupancySensor::new(),
            material_detector: MaterialDetector::new(),
            acoustic_probe: AcousticProbe::new(),
        })
    }

    /// Update every sensor using whatever real data `inputs` provides this
    /// tick. Sensors with no corresponding data in `inputs` are left
    /// unchanged (see [`SensorInputs`]).
    pub(super) fn update(&mut self, inputs: &SensorInputs<'_>) -> Result<()> {
        if let Some(celsius) = inputs.temperature_celsius {
            self.temperature_sensor.record_reading(celsius);
        }
        if let Some(humidity) = inputs.relative_humidity {
            self.humidity_sensor.record_reading(humidity);
        }
        if let Some(audio) = inputs.audio_buffer {
            self.noise_sensor
                .update_from_audio(audio, inputs.sample_rate)?;
            self.occupancy_sensor.update_from_audio(audio)?;
        }
        self.material_detector.update()?;
        self.acoustic_probe.update()?;
        Ok(())
    }
}

impl TemperatureSensor {
    pub(super) fn new() -> Self {
        Self {
            current_temperature: 22.0, // Room temperature
            temperature_history: VecDeque::new(),
            calibration_offset: 0.0,
            accuracy: 0.5,
        }
    }

    /// Record a real external temperature reading.
    ///
    /// This crate has no in-process source of ambient temperature (no
    /// microphone-derived signal implies a room's air temperature), so -
    /// unlike [`NoiseSensor`]/[`OccupancySensor`], which can derive real
    /// readings from an audio buffer - this sensor can only be driven by a
    /// caller-supplied measurement. It applies the calibration offset the
    /// same way the previous simulated implementation's construction-time
    /// baseline did.
    pub(super) fn record_reading(&mut self, celsius: f32) {
        let calibrated = celsius + self.calibration_offset;
        let reading = SensorReading {
            value: calibrated,
            timestamp: AdaptiveAcousticEnvironment::get_current_timestamp(),
            confidence: (1.0 - self.accuracy / 10.0).clamp(0.0, 1.0),
        };

        self.current_temperature = calibrated;
        self.temperature_history.push_back(reading);
        if self.temperature_history.len() > 100 {
            self.temperature_history.pop_front();
        }
    }
}

impl HumiditySensor {
    pub(super) fn new() -> Self {
        Self {
            current_humidity: 0.45, // 45% RH
            humidity_history: VecDeque::new(),
            calibration_params: HumidityCalibration {
                linear_coeff: 1.0,
                offset: 0.0,
                temp_compensation: 0.01,
            },
        }
    }

    /// Record a real external relative-humidity reading (0.0-1.0).
    ///
    /// See [`TemperatureSensor::record_reading`] for why this cannot be
    /// derived in-process and must be supplied by the caller.
    pub(super) fn record_reading(&mut self, relative_humidity: f32) {
        let calibrated = (relative_humidity * self.calibration_params.linear_coeff
            + self.calibration_params.offset)
            .clamp(0.0, 1.0);
        let reading = SensorReading {
            value: calibrated,
            timestamp: AdaptiveAcousticEnvironment::get_current_timestamp(),
            confidence: 0.9,
        };

        self.current_humidity = calibrated;
        self.humidity_history.push_back(reading);
        if self.humidity_history.len() > 100 {
            self.humidity_history.pop_front();
        }
    }
}

/// Compute a real frequency-domain noise spectrum from an audio buffer via
/// FFT, replacing a frozen construction-time spectrum.
///
/// Power is aggregated into the same six octave-band centers
/// [`AcousticProbe::new`] uses for its RT60 table (125 Hz .. 4 kHz), plus a
/// real spectral centroid and 85%-energy rolloff frequency.
fn compute_noise_spectrum(audio: &[f32], sample_rate: u32) -> Result<NoiseSpectrum> {
    const BAND_CENTERS: [f32; 6] = [125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0];
    const ROLLOFF_ENERGY_FRACTION: f32 = 0.85;

    let fft_len = audio.len().max(64).next_power_of_two();
    let mut windowed = audio.to_vec();
    windowed.resize(fft_len, 0.0);

    let mut planner = RealFftPlanner::<f32>::new();
    let fwd = planner.plan_fft_forward(fft_len);
    let n_bins = fft_len / 2 + 1;
    let mut spectrum = vec![Complex::new(0.0f32, 0.0f32); n_bins];
    fwd.process(&windowed, &mut spectrum)
        .map_err(|_| Error::LegacyRoom("FFT-based noise spectrum analysis failed".to_string()))?;

    let bin_hz = sample_rate as f32 / fft_len as f32;
    let power: Vec<f32> = spectrum.iter().map(|c| c.re * c.re + c.im * c.im).collect();
    let total_power: f32 = power.iter().sum::<f32>().max(1e-12);

    let centroid = power
        .iter()
        .enumerate()
        .map(|(k, &p)| k as f32 * bin_hz * p)
        .sum::<f32>()
        / total_power;

    let rolloff_threshold = total_power * ROLLOFF_ENERGY_FRACTION;
    let mut cumulative = 0.0f32;
    let mut rolloff = bin_hz * (n_bins.saturating_sub(1)) as f32;
    for (k, &p) in power.iter().enumerate() {
        cumulative += p;
        if cumulative >= rolloff_threshold {
            rolloff = k as f32 * bin_hz;
            break;
        }
    }

    let power_levels: Vec<f32> = BAND_CENTERS
        .iter()
        .map(|&center| {
            let lo = center / std::f32::consts::SQRT_2;
            let hi = center * std::f32::consts::SQRT_2;
            let band_power: f32 = power
                .iter()
                .enumerate()
                .filter(|(k, _)| {
                    let f = *k as f32 * bin_hz;
                    f >= lo && f < hi
                })
                .map(|(_, &p)| p)
                .sum();
            10.0 * band_power.max(1e-12).log10()
        })
        .collect();

    Ok(NoiseSpectrum {
        frequency_bands: BAND_CENTERS.to_vec(),
        power_levels,
        centroid,
        rolloff,
    })
}

/// Classify a coarse [`NoiseType`] from real spectral-shape features.
///
/// This is a simple, documented heuristic (not a trained classifier): energy
/// concentrated below 300 Hz reads as HVAC/mechanical hum, a centroid in the
/// speech range with a moderate rolloff reads as conversation, and a
/// broadband spectrum with high rolloff reads as (white) noise; anything
/// else is reported as mixed rather than guessed at.
fn classify_noise_type(spectrum: &NoiseSpectrum) -> NoiseType {
    if spectrum.centroid < 300.0 {
        NoiseType::HVAC
    } else if (300.0..3000.0).contains(&spectrum.centroid) && spectrum.rolloff < 4000.0 {
        NoiseType::Conversation
    } else if spectrum.rolloff > 8000.0 {
        NoiseType::White
    } else {
        NoiseType::Mixed
    }
}

impl NoiseSensor {
    pub(super) fn new() -> Self {
        Self {
            current_level: 35.0, // 35 dB SPL
            spectrum: NoiseSpectrum {
                frequency_bands: vec![125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0],
                power_levels: vec![30.0, 32.0, 34.0, 35.0, 33.0, 31.0, 29.0],
                centroid: 800.0,
                rolloff: 2000.0,
            },
            noise_type: NoiseType::HVAC,
            noise_history: VecDeque::new(),
        }
    }

    /// Update the noise level and spectrum from a *real* audio buffer (e.g.
    /// a microphone capture, or the currently-rendered scene audio),
    /// replacing the previous random walk around a fixed constant.
    ///
    /// The level is computed from the buffer's real RMS amplitude, converted
    /// to dB and offset by `REFERENCE_SPL_OFFSET_DB` (94 dB SPL at 0 dBFS is
    /// a common microphone calibration reference point - adjust for a
    /// specific capture chain if a different one is used). The spectrum is a
    /// real FFT analysis (see [`compute_noise_spectrum`]), and the noise
    /// type is classified from that real spectral shape (see
    /// [`classify_noise_type`]).
    pub(super) fn update_from_audio(&mut self, audio: &[f32], sample_rate: u32) -> Result<()> {
        const REFERENCE_SPL_OFFSET_DB: f32 = 94.0;

        if audio.is_empty() {
            return Err(Error::LegacyRoom(
                "NoiseSensor::update_from_audio called with an empty audio buffer".to_string(),
            ));
        }
        if sample_rate == 0 {
            return Err(Error::LegacyRoom(
                "NoiseSensor::update_from_audio called with sample_rate = 0".to_string(),
            ));
        }

        let rms = (audio.iter().map(|&s| s * s).sum::<f32>() / audio.len() as f32).sqrt();
        let dbfs = 20.0 * rms.max(1e-9).log10();
        self.current_level = dbfs + REFERENCE_SPL_OFFSET_DB;
        self.spectrum = compute_noise_spectrum(audio, sample_rate)?;
        self.noise_type = classify_noise_type(&self.spectrum);

        let reading = NoiseReading {
            level: self.current_level,
            spectrum: self.spectrum.clone(),
            noise_type: self.noise_type,
            timestamp: AdaptiveAcousticEnvironment::get_current_timestamp(),
            confidence: 0.9,
        };

        self.noise_history.push_back(reading);
        if self.noise_history.len() > 1000 {
            self.noise_history.pop_front();
        }

        Ok(())
    }
}

impl OccupancySensor {
    pub(super) fn new() -> Self {
        Self {
            occupant_count: 1,
            occupant_positions: vec![Position3D::new(0.0, 0.0, 1.7)], // Standing height
            activity_level: ActivityLevel::Low,
            confidence: 0.85,
        }
    }

    /// Derive an activity-level estimate from real voice-activity
    /// characteristics (RMS energy and zero-crossing rate) of an audio
    /// buffer, replacing the previous complete no-op.
    ///
    /// A single-channel audio buffer has no way to count distinct people or
    /// locate them in space, so `occupant_count`/`occupant_positions` are
    /// intentionally left untouched here (this crate has no real data
    /// source for either) - only `activity_level`/`confidence` are updated.
    pub(super) fn update_from_audio(&mut self, audio: &[f32]) -> Result<()> {
        if audio.is_empty() {
            return Err(Error::LegacyRoom(
                "OccupancySensor::update_from_audio called with an empty audio buffer".to_string(),
            ));
        }

        let rms = (audio.iter().map(|&s| s * s).sum::<f32>() / audio.len() as f32).sqrt();
        // Zero-crossing rate: a real, simple proxy that tends to be higher
        // for voiced/fricative speech than for silence or steady-state hum.
        let zero_crossings = audio
            .windows(2)
            .filter(|w| (w[0] >= 0.0) != (w[1] >= 0.0))
            .count();
        let zcr = zero_crossings as f32 / audio.len() as f32;

        self.activity_level = if rms < 0.005 {
            ActivityLevel::None
        } else if rms < 0.02 && zcr < 0.15 {
            ActivityLevel::Low
        } else if rms < 0.1 && zcr < 0.3 {
            ActivityLevel::Medium
        } else if rms < 0.3 {
            ActivityLevel::High
        } else {
            ActivityLevel::VeryHigh
        };
        self.confidence = if rms > 0.005 { 0.6 } else { 0.3 };

        Ok(())
    }
}

impl MaterialDetector {
    fn new() -> Self {
        let mut detected_materials = HashMap::new();
        detected_materials.insert(
            "carpet".to_string(),
            MaterialProperties {
                name: "Carpet".to_string(),
                absorption: [
                    ("125Hz", 0.02),
                    ("250Hz", 0.04),
                    ("500Hz", 0.08),
                    ("1000Hz", 0.12),
                    ("2000Hz", 0.22),
                    ("4000Hz", 0.35),
                ]
                .iter()
                .map(|(k, v)| (k.to_string(), *v))
                .collect(),
                scattering: HashMap::new(),
                transmission: HashMap::new(),
                density: 400.0,
                roughness: 0.3,
            },
        );

        Self {
            detected_materials,
            detection_confidence: HashMap::new(),
            last_update: Instant::now(),
            detection_method: MaterialDetectionMethod::AcousticAnalysis,
        }
    }

    /// Refresh the "last checked" timestamp.
    ///
    /// Real acoustic-analysis-based material re-detection (matching an
    /// impulse response's decay/absorption signature against a material
    /// database) is not implemented yet; this intentionally does not
    /// fabricate a new `detected_materials` entry on every tick. Until real
    /// re-detection exists, callers should treat `detected_materials` as the
    /// construction-time default rather than a live reading.
    fn update(&mut self) -> Result<()> {
        self.last_update = Instant::now();
        Ok(())
    }
}

impl AcousticProbe {
    fn new() -> Self {
        Self {
            impulse_responses: HashMap::new(),
            rt60_measurements: [
                ("125Hz", 1.2),
                ("250Hz", 1.1),
                ("500Hz", 0.9),
                ("1000Hz", 0.8),
                ("2000Hz", 0.7),
                ("4000Hz", 0.6),
            ]
            .iter()
            .map(|(k, v)| (k.to_string(), *v))
            .collect(),
            frequency_responses: HashMap::new(),
            last_probe_time: Instant::now(),
        }
    }

    /// Reset the periodic probe timer.
    ///
    /// Real acoustic probing (emitting a probe signal and recording the
    /// resulting impulse response) requires driving an actual output/input
    /// audio path, which this sensor-only type does not own; that
    /// integration is not implemented yet, so no `rt60_measurements`/
    /// `impulse_responses` entries are fabricated here.
    fn update(&mut self) -> Result<()> {
        if self.last_probe_time.elapsed() > Duration::from_secs(60) {
            self.last_probe_time = Instant::now();
        }
        Ok(())
    }
}

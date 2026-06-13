//! Reverberation DSP: image-source early reflections and a feedback-delay-network
//! (FDN) late reverberator.
//!
//! The early-reflection stage models a shoebox room with the image-source method,
//! turning each reflection path into a delayed, attenuated tap. The late-reverb
//! stage is a Jot-style FDN whose delay lines are mixed through a lossless
//! orthogonal (Householder) matrix scaled by per-line decay gains so that the
//! network's T60 matches the requested reverberation time. A short all-pass
//! diffusion chain follows the FDN to build echo density.

use super::{ReflectionPath, RoomConfig};
use crate::types::Position3D;
use scirs2_core::ndarray::{Array1, Array2};
use std::collections::VecDeque;

/// Speed of sound in air at ~20 °C (m/s).
const SPEED_OF_SOUND: f32 = 343.0;
/// Default processing sample rate (Hz).
const DEFAULT_SAMPLE_RATE: f32 = 44_100.0;
/// Maximum inter-aural time difference in samples (~0.65 ms across the head at 44.1 kHz).
const MAX_ITD_SAMPLES: f32 = 29.0;

/// Early reflection processor.
///
/// Computes image-source reflection paths for a shoebox room and renders them as
/// a bank of delayed/attenuated taps applied to the dry signal.
#[derive(Debug, Clone)]
pub struct EarlyReflectionProcessor {
    /// Cached image-source reflection paths (lazily populated on first use).
    reflection_paths: Vec<ReflectionPath>,
    /// Maximum reflection order for the image-source model.
    max_order: usize,
    /// Speed of sound in m/s.
    speed_of_sound: f32,
    /// Sample rate in Hz.
    sample_rate: f32,
    /// Room dimensions (width, height, depth) in meters.
    dimensions: (f32, f32, f32),
    /// Average wall reflection coefficient (1 - absorption).
    reflection_coefficient: f32,
}

/// Late reverberation processor built around a feedback delay network.
#[derive(Debug, Clone)]
pub struct LateReverbProcessor {
    /// Parallel feedback delay networks forming the reverb tank.
    feedback_networks: Vec<FeedbackDelayNetwork>,
    /// All-pass diffusion chain for the left channel.
    diffusion_filters: Vec<AllPassFilter>,
    /// All-pass diffusion chain for the right channel.
    diffusion_filters_right: Vec<AllPassFilter>,
    /// Wet output gain applied to the reverb tail.
    wet_gain: f32,
}

/// Combined reverberation processor.
#[derive(Debug, Clone)]
pub struct ReverbProcessor {
    /// Early reflections.
    #[allow(dead_code)]
    early_processor: EarlyReflectionProcessor,
    /// Late reverb.
    #[allow(dead_code)]
    late_processor: LateReverbProcessor,
    /// Crossover frequency between early and late.
    #[allow(dead_code)]
    crossover_frequency: f32,
    /// Mix levels.
    #[allow(dead_code)]
    dry_level: f32,
    #[allow(dead_code)]
    early_level: f32,
    #[allow(dead_code)]
    late_level: f32,
}

/// Feedback delay network for late reverberation.
#[derive(Debug, Clone)]
pub struct FeedbackDelayNetwork {
    /// Delay lines.
    delay_lines: Vec<DelayLine>,
    /// Lossless mixing matrix scaled by per-line decay gains (`A = diag(g)·H`).
    feedback_matrix: Array2<f32>,
    /// Per-line input gains.
    input_gains: Array1<f32>,
    /// Per-line output gains.
    output_gains: Array1<f32>,
}

/// All-pass filter for diffusion.
#[derive(Debug, Clone)]
pub struct AllPassFilter {
    /// Delay line.
    delay_line: DelayLine,
    /// Feedback coefficient.
    feedback: f32,
    /// Feed-forward coefficient (`-feedback`).
    feedforward: f32,
}

/// Delay line backed by a ring buffer.
#[derive(Debug, Clone)]
pub struct DelayLine {
    /// Buffer.
    buffer: VecDeque<f32>,
    /// Delay in samples.
    delay_samples: f32,
    /// Maximum delay.
    max_delay: usize,
}

impl EarlyReflectionProcessor {
    /// Create new early reflection processor.
    pub fn new(config: &RoomConfig) -> crate::Result<Self> {
        // Reflectivity is the complement of the room's mean absorption coefficient.
        let reflection_coefficient = (1.0 - config.average_absorption()).clamp(0.0, 0.99);
        Ok(Self {
            reflection_paths: Vec::new(),
            max_order: 3,
            speed_of_sound: SPEED_OF_SOUND,
            sample_rate: DEFAULT_SAMPLE_RATE,
            dimensions: config.dimensions,
            reflection_coefficient,
        })
    }

    /// Build image-source reflection paths for a shoebox room.
    ///
    /// Uses the Allen & Berkley image-source enumeration: for lattice indices
    /// `m ∈ [-order, order]³` and parities `p ∈ {0, 1}³`, the image position is
    /// `(1 - 2·p)·s + 2·m·L` and the number of wall reflections along the axis is
    /// `|2·m - p|`. Paths whose total reflection order exceeds `max_order` are
    /// discarded.
    fn build_image_source_paths(
        &self,
        source: Position3D,
        listener: Position3D,
    ) -> Vec<ReflectionPath> {
        let (lx, ly, lz) = self.dimensions;
        if lx <= 0.0 || ly <= 0.0 || lz <= 0.0 {
            return Vec::new();
        }

        let order = self.max_order as i32;
        let mut paths = Vec::new();
        for mx in -order..=order {
            for my in -order..=order {
                for mz in -order..=order {
                    for &px in &[0i32, 1] {
                        for &py in &[0i32, 1] {
                            for &pz in &[0i32, 1] {
                                let reflection_order = ((2 * mx - px).abs()
                                    + (2 * my - py).abs()
                                    + (2 * mz - pz).abs())
                                    as usize;
                                // Skip the direct path; cap to the configured order.
                                if reflection_order == 0 || reflection_order > self.max_order {
                                    continue;
                                }

                                let image = Position3D::new(
                                    (1 - 2 * px) as f32 * source.x + 2.0 * mx as f32 * lx,
                                    (1 - 2 * py) as f32 * source.y + 2.0 * my as f32 * ly,
                                    (1 - 2 * pz) as f32 * source.z + 2.0 * mz as f32 * lz,
                                );
                                let distance = image.distance_to(&listener);
                                if distance < 1e-3 {
                                    continue;
                                }

                                // Reflection-coefficient product along the path times
                                // the 1/distance spreading law (clamped in the near field).
                                let reflection_gain =
                                    self.reflection_coefficient.powi(reflection_order as i32);
                                let attenuation = reflection_gain / distance.max(1.0);
                                if attenuation < 1e-4 {
                                    continue;
                                }

                                let delay_samples =
                                    (distance / self.speed_of_sound * self.sample_rate).round()
                                        as usize;
                                paths.push(ReflectionPath {
                                    path: vec![source, image, listener],
                                    length: distance,
                                    delay_samples,
                                    attenuation,
                                    surfaces: Vec::new(),
                                });
                            }
                        }
                    }
                }
            }
        }
        paths
    }

    /// Process early reflections.
    ///
    /// Each cached reflection path contributes a delayed and attenuated copy of the
    /// dry signal. A small inter-aural time difference derived from the image's
    /// lateral position keeps the two channels distinct.
    pub async fn process(
        &mut self,
        left_channel: &mut Array1<f32>,
        right_channel: &mut Array1<f32>,
        source_position: &Position3D,
    ) -> crate::Result<()> {
        // Populate the image-source paths on first use (assuming a centred listener).
        if self.reflection_paths.is_empty() {
            let (lx, ly, lz) = self.dimensions;
            let listener = Position3D::new(lx * 0.5, ly * 0.5, lz * 0.5);
            self.reflection_paths = self.build_image_source_paths(*source_position, listener);
        }
        if self.reflection_paths.is_empty() {
            return Ok(());
        }

        let len = left_channel.len().min(right_channel.len());
        // Convolve the *dry* signal with the reflection taps (an FIR), so the
        // reflections never feed back on themselves.
        let dry_left = left_channel.clone();
        let dry_right = right_channel.clone();

        for path in &self.reflection_paths {
            let delay = path.delay_samples;
            if delay >= len {
                continue;
            }
            let attenuation = path.attenuation;

            // Inter-aural time difference from the image's lateral offset; only the
            // far ear is delayed further, keeping the near-ear delay exact.
            let (left_delay, right_delay) = match (path.path.get(1), path.path.get(2)) {
                (Some(image), Some(listener)) if path.length > 0.0 => {
                    let lateral = (image.x - listener.x) / path.length;
                    let itd = (lateral.abs() * MAX_ITD_SAMPLES).round() as usize;
                    if lateral < 0.0 {
                        (delay, delay + itd) // image to the left → left ear first
                    } else {
                        (delay + itd, delay) // image to the right → right ear first
                    }
                }
                _ => (delay, delay),
            };

            if left_delay < len {
                for i in left_delay..len {
                    left_channel[i] += dry_left[i - left_delay] * attenuation;
                }
            }
            if right_delay < len {
                for i in right_delay..len {
                    right_channel[i] += dry_right[i - right_delay] * attenuation;
                }
            }
        }
        Ok(())
    }
}

impl LateReverbProcessor {
    /// Create new late reverb processor.
    pub fn new(config: &RoomConfig) -> crate::Result<Self> {
        let reverb_time = config.reverb_time.max(0.05);
        // Mutually incommensurate delays (seconds) for high modal density.
        let delays = [0.0297, 0.0371, 0.0411, 0.0437];
        let feedback_networks = vec![FeedbackDelayNetwork::new(
            &delays,
            DEFAULT_SAMPLE_RATE,
            reverb_time,
        )?];

        // Two slightly detuned all-pass chains decorrelate the stereo tail.
        let diffusion_filters = vec![
            AllPassFilter::new(0.0050, 0.7)?,
            AllPassFilter::new(0.0117, 0.5)?,
        ];
        let diffusion_filters_right = vec![
            AllPassFilter::new(0.0053, 0.7)?,
            AllPassFilter::new(0.0123, 0.5)?,
        ];

        Ok(Self {
            feedback_networks,
            diffusion_filters,
            diffusion_filters_right,
            wet_gain: 0.35,
        })
    }

    /// Process late reverberation through the feedback delay network.
    pub async fn process(
        &mut self,
        left_channel: &mut Array1<f32>,
        right_channel: &mut Array1<f32>,
    ) -> crate::Result<()> {
        let len = left_channel.len().min(right_channel.len());
        for idx in 0..len {
            let input = 0.5 * (left_channel[idx] + right_channel[idx]);

            // Sum the (decorrelated) stereo output of every FDN in the tank.
            let mut wet_left = 0.0f32;
            let mut wet_right = 0.0f32;
            for network in &mut self.feedback_networks {
                let (left, right) = network.process(input);
                wet_left += left;
                wet_right += right;
            }

            // All-pass diffusion increases echo density on the output path.
            for filter in &mut self.diffusion_filters {
                wet_left = filter.process(wet_left);
            }
            for filter in &mut self.diffusion_filters_right {
                wet_right = filter.process(wet_right);
            }

            left_channel[idx] += wet_left * self.wet_gain;
            right_channel[idx] += wet_right * self.wet_gain;
        }
        Ok(())
    }
}

impl ReverbProcessor {
    /// Create new reverb processor.
    pub fn new(config: &RoomConfig) -> crate::Result<Self> {
        Ok(Self {
            early_processor: EarlyReflectionProcessor::new(config)?,
            late_processor: LateReverbProcessor::new(config)?,
            crossover_frequency: 500.0,
            dry_level: 0.7,
            early_level: 0.3,
            late_level: 0.4,
        })
    }
}

impl FeedbackDelayNetwork {
    /// Create new feedback delay network.
    ///
    /// `delays` are per-line delay times in seconds. The feedback matrix is a
    /// Householder reflection (lossless/orthogonal) with each row scaled by the
    /// per-line decay gain `g_i = 10^(-3·τ_i/RT60)`, which makes the network's
    /// energy decay reach -60 dB after `reverb_time` seconds.
    pub fn new(delays: &[f32], sample_rate: f32, reverb_time: f32) -> crate::Result<Self> {
        let size = delays.len();
        let mut delay_lines = Vec::with_capacity(size);
        for &delay in delays {
            delay_lines.push(DelayLine::new(delay, sample_rate)?);
        }

        let reverb_time = reverb_time.max(0.05);
        // Householder reflection H = I - (2/N)·1·1ᵀ is orthogonal, hence lossless.
        let two_over_n = 2.0 / size as f32;
        let mut feedback_matrix = Array2::<f32>::zeros((size, size));
        for i in 0..size {
            let decay_gain = 10.0f32.powf(-3.0 * delays[i] / reverb_time);
            for j in 0..size {
                let householder = if i == j {
                    1.0 - two_over_n
                } else {
                    -two_over_n
                };
                feedback_matrix[[i, j]] = decay_gain * householder;
            }
        }

        // Energy-preserving normalisation for the injection/extraction taps.
        let normalization = 1.0 / (size as f32).sqrt();
        let input_gains = Array1::from_elem(size, normalization);
        let output_gains = Array1::from_elem(size, normalization);

        Ok(Self {
            delay_lines,
            feedback_matrix,
            input_gains,
            output_gains,
        })
    }

    /// Advance the network by one sample, returning the decorrelated
    /// `(left, right)` wet outputs.
    fn process(&mut self, input: f32) -> (f32, f32) {
        let size = self.delay_lines.len();

        // Read the current delayed outputs of every line.
        let mut outputs = Array1::<f32>::zeros(size);
        for (i, line) in self.delay_lines.iter().enumerate() {
            outputs[i] = line.read();
        }

        // Lossless feedback mixing (decay gains are folded into the matrix).
        let mut feedback = Array1::<f32>::zeros(size);
        for i in 0..size {
            let mut accumulator = 0.0f32;
            for j in 0..size {
                accumulator += self.feedback_matrix[[i, j]] * outputs[j];
            }
            feedback[i] = accumulator;
        }

        // Inject the input and recirculate the mixed signal.
        for i in 0..size {
            let excitation = input * self.input_gains[i] + feedback[i];
            self.delay_lines[i].write(excitation);
        }

        // Decorrelated stereo down-mix: flip the sign for odd lines on the right.
        let mut left = 0.0f32;
        let mut right = 0.0f32;
        for i in 0..size {
            let weighted = self.output_gains[i] * outputs[i];
            left += weighted;
            right += if i % 2 == 0 { weighted } else { -weighted };
        }
        (left, right)
    }
}

impl AllPassFilter {
    /// Create new all-pass filter.
    pub fn new(delay: f32, feedback: f32) -> crate::Result<Self> {
        Ok(Self {
            delay_line: DelayLine::new(delay, DEFAULT_SAMPLE_RATE)?,
            feedback,
            feedforward: -feedback,
        })
    }

    /// Process one sample through a Schroeder all-pass section.
    pub fn process(&mut self, input: f32) -> f32 {
        let delayed = self.delay_line.read();
        let recirculated = input + self.feedback * delayed;
        let output = self.feedforward * input + delayed;
        self.delay_line.write(recirculated);
        output
    }
}

impl DelayLine {
    /// Create new delay line.
    pub fn new(delay_time: f32, sample_rate: f32) -> crate::Result<Self> {
        let delay_samples = delay_time * sample_rate;
        let max_delay = delay_samples.ceil() as usize + 1;
        let buffer = VecDeque::with_capacity(max_delay);

        Ok(Self {
            buffer,
            delay_samples,
            max_delay,
        })
    }

    /// Read the currently delayed sample without modifying the buffer.
    pub fn read(&self) -> f32 {
        let delay_index = self.delay_samples as usize;
        if self.buffer.len() > delay_index {
            self.buffer[self.buffer.len() - 1 - delay_index]
        } else {
            0.0
        }
    }

    /// Push a new sample into the delay line, advancing its state.
    pub fn write(&mut self, input: f32) {
        self.buffer.push_back(input);
        while self.buffer.len() > self.max_delay {
            self.buffer.pop_front();
        }
    }

    /// Push a sample and return the delayed output in one call.
    pub fn process(&mut self, input: f32) -> f32 {
        self.write(input);
        self.read()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::room::RoomConfig;
    use crate::types::Position3D;
    use scirs2_core::ndarray::Array1;

    /// Sum of squared samples over `[start, end)`.
    fn energy(signal: &Array1<f32>, start: usize, end: usize) -> f32 {
        let end = end.min(signal.len());
        (start..end).map(|i| signal[i] * signal[i]).sum()
    }

    async fn late_window_energy(reverb_time: f32, len: usize, window: (usize, usize)) -> f32 {
        let config = RoomConfig::new((8.0, 5.0, 6.0), reverb_time);
        let mut late = LateReverbProcessor::new(&config).expect("late reverb");
        let mut left = Array1::zeros(len);
        let mut right = Array1::zeros(len);
        left[0] = 1.0;
        right[0] = 1.0;
        late.process(&mut left, &mut right).await.expect("process");
        energy(&left, window.0, window.1)
    }

    #[tokio::test]
    async fn test_late_reverb_impulse_produces_decaying_tail() {
        let config = RoomConfig::new((8.0, 5.0, 6.0), 1.0);
        let mut late = LateReverbProcessor::new(&config).expect("late reverb");
        let len = 30_000;
        let mut left = Array1::zeros(len);
        let mut right = Array1::zeros(len);
        left[0] = 1.0;
        right[0] = 1.0;
        late.process(&mut left, &mut right).await.expect("process");

        // Energy must appear after the direct (impulse) sample.
        let tail = energy(&left, 1, len);
        assert!(tail > 0.0, "late reverb should produce a tail, got {tail}");

        // The tail must decay: an earlier window holds more energy than a later one.
        let early_window = energy(&left, 2_000, 7_000);
        let late_window = energy(&left, 17_000, 22_000);
        assert!(early_window > 0.0, "no early tail energy: {early_window}");
        assert!(
            early_window > late_window,
            "tail should decay: early={early_window}, late={late_window}"
        );
    }

    #[tokio::test]
    async fn test_late_reverb_longer_tail_for_larger_reverb_time() {
        let len = 40_000;
        let window = (25_000, 40_000);
        let short_tail = late_window_energy(0.5, len, window).await;
        let long_tail = late_window_energy(2.0, len, window).await;
        assert!(
            long_tail > short_tail,
            "RT60=2.0 tail ({long_tail}) should exceed RT60=0.5 tail ({short_tail})"
        );
    }

    #[tokio::test]
    async fn test_late_reverb_stays_bounded_over_one_second() {
        let config = RoomConfig::new((8.0, 5.0, 6.0), 2.0);
        let mut late = LateReverbProcessor::new(&config).expect("late reverb");
        let len = 44_100; // 1 second at 44.1 kHz
                          // Continuous sinusoidal excitation to stress the feedback loop.
        let mut left = Array1::from_shape_fn(len, |i| (i as f32 * 0.05).sin());
        let mut right = Array1::from_shape_fn(len, |i| (i as f32 * 0.05).cos());
        late.process(&mut left, &mut right).await.expect("process");

        for i in 0..len {
            assert!(
                left[i].is_finite() && right[i].is_finite(),
                "non-finite sample at {i}"
            );
            assert!(
                left[i].abs() < 20.0 && right[i].abs() < 20.0,
                "blow-up at {i}: left={}, right={}",
                left[i],
                right[i]
            );
        }
    }

    #[tokio::test]
    async fn test_early_reflections_add_energy_at_expected_delay() {
        let config = RoomConfig::new((6.0, 4.0, 5.0), 0.8);
        let mut early = EarlyReflectionProcessor::new(&config).expect("early reflections");
        let len = 8_000;
        let mut left = Array1::zeros(len);
        let mut right = Array1::zeros(len);
        left[0] = 1.0;
        right[0] = 1.0;
        let source = Position3D::new(1.0, 1.0, 1.0);
        early
            .process(&mut left, &mut right, &source)
            .await
            .expect("process");

        // The image-source model must have produced reflection paths.
        assert!(
            !early.reflection_paths.is_empty(),
            "image sources should be built"
        );

        // Reflections add delayed energy beyond the direct sample.
        let tail = energy(&left, 1, len) + energy(&right, 1, len);
        assert!(tail > 0.0, "early reflections should add delayed energy");

        // Energy must appear at the earliest reflection delay (the near ear hits it
        // exactly, since only the far ear receives the extra ITD offset).
        let min_delay = early
            .reflection_paths
            .iter()
            .map(|path| path.delay_samples)
            .filter(|&delay| delay > 0 && delay < len)
            .min()
            .expect("at least one reflection inside the buffer");
        let at_delay = left[min_delay].abs() + right[min_delay].abs();
        assert!(at_delay > 0.0, "expected energy at delay {min_delay}");
    }
}

use crate::ai::{StyleEmbedding, StyleTransfer};
use crate::types::VoiceCharacteristics;
use crate::Error;
use scirs2_core::Complex;

use super::{ConversionQuality, SpeakerEmbedding};

pub(super) fn estimate_average_f0(samples: &[f32]) -> Result<f32, Error> {
    const SAMPLE_RATE: f32 = 44100.0;
    const FRAME_SIZE: usize = 2048;
    const HOP_SIZE: usize = 512;
    let tau_min = (SAMPLE_RATE / 800.0).ceil() as usize;
    let tau_max = (SAMPLE_RATE / 80.0).ceil() as usize;

    let normed_ac = |centered: &[f32], tau: usize| -> f32 {
        let n_ov = FRAME_SIZE.saturating_sub(tau);
        if n_ov == 0 {
            return 0.0;
        }
        let mut cross = 0.0f32;
        let mut left_sq = 0.0f32;
        let mut right_sq = 0.0f32;
        for i in 0..n_ov {
            cross += centered[i] * centered[i + tau];
            left_sq += centered[i] * centered[i];
            right_sq += centered[i + tau] * centered[i + tau];
        }
        let denom = (left_sq * right_sq).sqrt();
        if denom > 1e-10 {
            cross / denom
        } else {
            0.0
        }
    };

    if samples.len() < FRAME_SIZE {
        return Ok(0.0);
    }

    let mut voiced_f0s: Vec<f32> = Vec::new();
    let mut frame_start = 0;

    while frame_start + FRAME_SIZE <= samples.len() {
        let frame = &samples[frame_start..frame_start + FRAME_SIZE];

        let mean: f32 = frame.iter().sum::<f32>() / FRAME_SIZE as f32;
        let centered: Vec<f32> = frame.iter().map(|&s| s - mean).collect();

        let full_energy: f32 = centered.iter().map(|&s| s * s).sum();
        if full_energy < 1e-10 {
            frame_start += HOP_SIZE;
            continue;
        }

        let tau_limit = tau_max.min(FRAME_SIZE - 1);

        let mut global_best_corr = -1.0f32;
        let mut global_best_tau = tau_min;
        for tau in tau_min..=tau_limit {
            let r = normed_ac(&centered, tau);
            if r > global_best_corr {
                global_best_corr = r;
                global_best_tau = tau;
            }
        }

        let threshold = global_best_corr * 0.92;
        let mut best_tau = global_best_tau;
        for tau in tau_min..=tau_limit {
            let r = normed_ac(&centered, tau);
            if r >= threshold {
                best_tau = tau;
                break;
            }
        }
        let best_corr = normed_ac(&centered, best_tau);

        if best_corr > 0.35 {
            let f0 = (SAMPLE_RATE / best_tau as f32).clamp(80.0, 500.0);
            voiced_f0s.push(f0);
        }

        frame_start += HOP_SIZE;
    }

    if voiced_f0s.is_empty() {
        return Ok(0.0);
    }

    voiced_f0s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = voiced_f0s.len() / 2;
    let median = if voiced_f0s.len().is_multiple_of(2) {
        (voiced_f0s[mid - 1] + voiced_f0s[mid]) / 2.0
    } else {
        voiced_f0s[mid]
    };
    Ok(median)
}

pub(super) fn extract_speaker_features(samples: &[f32]) -> Result<Vec<f32>, Error> {
    const FRAME_SIZE: usize = 2048;
    const N_MFCC: usize = 13;
    const N_FILT: usize = 26;
    const SAMPLE_RATE: f32 = 44100.0;

    if samples.is_empty() {
        return Ok(vec![0.0f32; 512]);
    }

    let compute_mfcc_frame = |frame: &[f32]| -> [f32; N_MFCC] {
        let len = frame.len();
        if len == 0 {
            return [0.0f32; N_MFCC];
        }
        let windowed: Vec<f64> = frame
            .iter()
            .enumerate()
            .map(|(i, &s)| {
                let w = 0.5
                    * (1.0
                        - (2.0 * std::f64::consts::PI * i as f64 / (len - 1).max(1) as f64).cos());
                s as f64 * w
            })
            .collect();

        let complex_in: Vec<Complex<f64>> =
            windowed.iter().map(|&x| Complex::new(x, 0.0)).collect();
        let fft_out = match scirs2_fft::fft(&complex_in, None) {
            Ok(v) => v,
            Err(_) => return [0.0f32; N_MFCC],
        };
        let n_bins = len / 2 + 1;
        let power: Vec<f64> = fft_out[..n_bins]
            .iter()
            .map(|c| (c.re * c.re + c.im * c.im).max(1e-30))
            .collect();

        let nyquist = SAMPLE_RATE as f64 / 2.0;
        let hz_to_mel = |hz: f64| 2595.0 * (1.0 + hz / 700.0).log10();
        let mel_to_hz = |mel: f64| 700.0 * (10.0_f64.powf(mel / 2595.0) - 1.0);
        let mel_low = hz_to_mel(0.0);
        let mel_high = hz_to_mel(nyquist);
        let mel_pts: Vec<f64> = (0..=N_FILT + 1)
            .map(|i| mel_low + (mel_high - mel_low) * i as f64 / (N_FILT + 1) as f64)
            .collect();
        let hz_pts: Vec<f64> = mel_pts.iter().map(|&m| mel_to_hz(m)).collect();
        let bin_pts: Vec<usize> = hz_pts
            .iter()
            .map(|&hz| ((hz / nyquist) * (n_bins - 1) as f64).round() as usize)
            .collect();

        let mut filt_energies = [0.0f64; N_FILT];
        for m in 0..N_FILT {
            let start = bin_pts[m];
            let center = bin_pts[m + 1];
            let end = bin_pts[m + 2];
            for k in start..center {
                if k < power.len() && center > start {
                    let w = (k - start) as f64 / (center - start) as f64;
                    filt_energies[m] += power[k] * w;
                }
            }
            for k in center..end {
                if k < power.len() && end > center {
                    let w = (end - k) as f64 / (end - center) as f64;
                    filt_energies[m] += power[k] * w;
                }
            }
            filt_energies[m] = filt_energies[m].max(1e-30).ln();
        }

        let dct_out = match scirs2_fft::dct(&filt_energies, None, Some("ortho")) {
            Ok(v) => v,
            Err(_) => return [0.0f32; N_MFCC],
        };
        let mut mfcc = [0.0f32; N_MFCC];
        for (i, m) in mfcc.iter_mut().enumerate() {
            *m = dct_out.get(i).copied().unwrap_or(0.0) as f32;
        }
        mfcc
    };

    let len = samples.len();
    let a_start = 0;
    let a_end = (a_start + FRAME_SIZE).min(len);
    let b_start = if len >= FRAME_SIZE + 512 { 512 } else { 0 };
    let b_end = (b_start + FRAME_SIZE).min(len);
    let c_start = if len >= FRAME_SIZE + 1024 {
        1024
    } else {
        b_start
    };
    let c_end = (c_start + FRAME_SIZE).min(len);

    let mfcc_a = compute_mfcc_frame(&samples[a_start..a_end]);
    let mfcc_b = compute_mfcc_frame(&samples[b_start..b_end]);
    let mfcc_c = compute_mfcc_frame(&samples[c_start..c_end]);

    let delta: [f32; N_MFCC] = std::array::from_fn(|i| mfcc_b[i] - mfcc_a[i]);
    let delta2: [f32; N_MFCC] = std::array::from_fn(|i| mfcc_c[i] - mfcc_b[i]);

    let mfcc_mean: [f32; N_MFCC] =
        std::array::from_fn(|i| (mfcc_a[i] + mfcc_b[i] + mfcc_c[i]) / 3.0);
    let mfcc_var: [f32; N_MFCC] = std::array::from_fn(|i| {
        let mu = mfcc_mean[i];
        ((mfcc_a[i] - mu).powi(2) + (mfcc_b[i] - mu).powi(2) + (mfcc_c[i] - mu).powi(2)) / 3.0
    });
    let mfcc_min: [f32; N_MFCC] = std::array::from_fn(|i| mfcc_a[i].min(mfcc_b[i]).min(mfcc_c[i]));
    let mfcc_max: [f32; N_MFCC] = std::array::from_fn(|i| mfcc_a[i].max(mfcc_b[i]).max(mfcc_c[i]));

    const HOP: usize = 512;
    let rms_vals: Vec<f32> = samples
        .chunks(HOP)
        .filter(|c| !c.is_empty())
        .map(|chunk| {
            let sq: f32 = chunk.iter().map(|&s| s * s).sum();
            (sq / chunk.len() as f32).sqrt()
        })
        .collect();
    let energy_mean = if rms_vals.is_empty() {
        0.0f32
    } else {
        rms_vals.iter().sum::<f32>() / rms_vals.len() as f32
    };
    let energy_std = if rms_vals.is_empty() {
        0.0f32
    } else {
        let mu = energy_mean;
        (rms_vals.iter().map(|&r| (r - mu).powi(2)).sum::<f32>() / rms_vals.len() as f32).sqrt()
    };
    let energy_max = rms_vals.iter().cloned().fold(0.0f32, f32::max);

    let zcr = if samples.len() < 2 {
        0.0f32
    } else {
        samples.windows(2).filter(|w| w[0] * w[1] < 0.0).count() as f32 / (samples.len() - 1) as f32
    };

    let f0_norm = estimate_average_f0(samples).unwrap_or(220.0) / SAMPLE_RATE;

    let mut features = vec![0.0f32; 512];
    for i in 0..N_MFCC {
        features[i] = mfcc_a[i];
        features[N_MFCC + i] = delta[i];
        features[2 * N_MFCC + i] = delta2[i];
    }
    features[39..(N_MFCC + 39)].copy_from_slice(&mfcc_mean[..N_MFCC]);
    features[52..(N_MFCC + 52)].copy_from_slice(&mfcc_var[..N_MFCC]);
    features[65..(N_MFCC + 65)].copy_from_slice(&mfcc_min[..N_MFCC]);
    features[78..(N_MFCC + 78)].copy_from_slice(&mfcc_max[..N_MFCC]);
    features[91] = energy_mean;
    features[92] = energy_std;
    features[93] = energy_max;
    features[94] = zcr;
    features[95] = f0_norm;

    for i in 96..512 {
        let base_idx = (i - 96) % 39;
        let scale = 1.0 + i as f32 * 1e-4;
        features[i] = features[base_idx] * scale;
    }

    Ok(features)
}

pub(super) fn extract_formants(samples: &[f32]) -> Result<Vec<f32>, Error> {
    const FRAME_SIZE: usize = 2048;
    const P: usize = 12;
    const N_SPEC: usize = 512;
    const SAMPLE_RATE: f32 = 44100.0;
    const MIN_FORMANT_HZ: f32 = 50.0;

    let frame_len = samples.len().min(FRAME_SIZE);
    if frame_len < P + 2 {
        return Ok(vec![800.0, 1200.0, 2600.0]);
    }

    let windowed: Vec<f32> = samples[..frame_len]
        .iter()
        .enumerate()
        .map(|(i, &s)| {
            let w = 0.5
                * (1.0
                    - (2.0 * std::f32::consts::PI * i as f32 / (frame_len - 1).max(1) as f32)
                        .cos());
            s * w
        })
        .collect();

    let mut r = [0.0f32; P + 2];
    for k in 0..=P {
        let mut acc = 0.0f32;
        for i in 0..(frame_len - k) {
            acc += windowed[i] * windowed[i + k];
        }
        r[k] = acc;
    }

    if r[0].abs() < 1e-15 {
        return Ok(vec![800.0, 1200.0, 2600.0]);
    }

    let mut a = [0.0f32; P + 1];
    let mut a_prev = [0.0f32; P + 1];
    let mut pred_error = r[0];

    for m in 1..=P {
        let mut lambda = r[m];
        for j in 1..m {
            lambda -= a[j] * r[m - j];
        }
        if pred_error.abs() < 1e-15 {
            break;
        }
        let km = -lambda / pred_error;
        a_prev[..=P].copy_from_slice(&a[..=P]);
        a[m] = km;
        for j in 1..m {
            a[j] = a_prev[j] + km * a_prev[m - j];
        }
        pred_error *= 1.0 - km * km;
    }

    let mut spectrum = [0.0f32; N_SPEC];
    for (bin, s) in spectrum.iter_mut().enumerate() {
        let omega = std::f32::consts::PI * bin as f32 / N_SPEC as f32;
        let mut re = 1.0f32;
        let mut im = 0.0f32;
        for k in 1..=P {
            let angle = -(k as f32) * omega;
            re += a[k] * angle.cos();
            im += a[k] * angle.sin();
        }
        *s = 1.0 / (re * re + im * im).sqrt().max(1e-10);
    }

    let min_bin = (MIN_FORMANT_HZ / (SAMPLE_RATE / 2.0) * N_SPEC as f32).ceil() as usize;
    let mut formants: Vec<f32> = Vec::with_capacity(5);

    for bin in min_bin.max(1)..(N_SPEC - 1) {
        if spectrum[bin] > spectrum[bin - 1] && spectrum[bin] > spectrum[bin + 1] {
            let alpha = spectrum[bin - 1];
            let beta = spectrum[bin];
            let gamma = spectrum[bin + 1];
            let denom = alpha - 2.0 * beta + gamma;
            let delta_bin = if denom.abs() > 1e-10 {
                0.5 * (alpha - gamma) / denom
            } else {
                0.0
            };
            let peak_bin = bin as f32 + delta_bin;
            formants.push(peak_bin * (SAMPLE_RATE / 2.0) / N_SPEC as f32);
            if formants.len() >= 5 {
                break;
            }
        }
    }

    let defaults = [800.0f32, 1200.0, 2600.0, 3200.0, 4000.0];
    while formants.len() < 3 {
        formants.push(defaults[formants.len()]);
    }

    Ok(formants[..3].to_vec())
}

pub(super) fn spectral_conversion(
    source_audio: &[f32],
    sample_rate: u32,
    target_speaker: &SpeakerEmbedding,
    quality: &ConversionQuality,
) -> Result<Vec<f32>, Error> {
    let n = source_audio.len();
    if n == 0 {
        return Ok(Vec::new());
    }

    const N_FFT: usize = 1024;
    const HOP: usize = 256;

    let hann: Vec<f64> = (0..N_FFT)
        .map(|i| 0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / (N_FFT - 1) as f64).cos()))
        .collect();

    let win_norm: f64 = hann.iter().map(|w| w * w).sum::<f64>() / HOP as f64;
    let win_norm = win_norm.max(1e-12);

    const SIGMA: f64 = 200.0;
    const AMP: f64 = 1.5;

    let target_envelope: Vec<f64> = (0..=N_FFT / 2)
        .map(|bin| {
            let freq = bin as f64 * sample_rate as f64 / N_FFT as f64;
            let gaussian_sum: f64 = target_speaker
                .formants
                .iter()
                .map(|&f_hz| {
                    let d = freq - f_hz as f64;
                    AMP * (-d * d / (2.0 * SIGMA * SIGMA)).exp()
                })
                .sum();
            1.0 + gaussian_sum
        })
        .collect();

    let mut output_acc = vec![0.0f64; n + N_FFT];
    let mut weight_acc = vec![0.0f64; n + N_FFT];

    let num_frames = n.div_ceil(HOP);

    for frame_idx in 0..num_frames {
        let start = frame_idx * HOP;

        let frame_complex: Vec<Complex<f64>> = (0..N_FFT)
            .map(|k| {
                let src_idx = start + k;
                let sample = if src_idx < n {
                    source_audio[src_idx] as f64
                } else {
                    0.0
                };
                Complex::new(sample * hann[k], 0.0)
            })
            .collect();

        let spectrum = scirs2_fft::fft(&frame_complex, Some(N_FFT))
            .map_err(|e| Error::Processing(format!("FFT error in spectral_conversion: {e}")))?;

        let half = N_FFT / 2 + 1;
        let mag: Vec<f64> = spectrum[..half].iter().map(|c| c.norm()).collect();

        const SMOOTH_BINS: usize = 20;
        let smooth_half = SMOOTH_BINS / 2;
        let source_envelope: Vec<f64> = (0..half)
            .map(|i| {
                let lo = i.saturating_sub(smooth_half);
                let hi = (i + smooth_half + 1).min(half);
                let sum: f64 = mag[lo..hi].iter().sum();
                sum / (hi - lo) as f64
            })
            .collect();

        let transfer: Vec<f64> = (0..half)
            .map(|i| target_envelope[i] / source_envelope[i].max(0.01))
            .collect();

        let mut modified: Vec<Complex<f64>> = spectrum.clone();
        for i in 0..half {
            modified[i] = Complex::new(spectrum[i].re * transfer[i], spectrum[i].im * transfer[i]);
        }
        for i in 1..(N_FFT / 2) {
            let mirror = N_FFT - i;
            modified[mirror] = Complex::new(modified[i].re, -modified[i].im);
        }

        let time_frame = scirs2_fft::ifft(&modified, Some(N_FFT))
            .map_err(|e| Error::Processing(format!("IFFT error in spectral_conversion: {e}")))?;

        for k in 0..N_FFT {
            let out_idx = start + k;
            if out_idx < output_acc.len() {
                output_acc[out_idx] += time_frame[k].re * hann[k];
                weight_acc[out_idx] += hann[k] * hann[k];
            }
        }
    }

    let blend = quality.conversion_strength.clamp(0.0, 1.0);
    let converted: Vec<f32> = (0..n)
        .map(|i| {
            let norm_denom = weight_acc[i].max(win_norm * 1e-6);
            let converted_sample = (output_acc[i] / norm_denom) as f32;
            source_audio[i] * (1.0 - blend) + converted_sample * blend
        })
        .collect();

    Ok(converted)
}

pub(super) fn formant_conversion(
    source_audio: &[f32],
    _sample_rate: u32,
    target_speaker: &SpeakerEmbedding,
    _quality: &ConversionQuality,
) -> Result<Vec<f32>, Error> {
    let mut converted = source_audio.to_vec();
    let brightness_factor = target_speaker.quality_metrics.brightness;
    for sample in &mut converted {
        *sample *= brightness_factor;
    }
    Ok(converted)
}

pub(super) fn neural_transfer_conversion(
    source_audio: &[f32],
    source_characteristics: &VoiceCharacteristics,
    target_speaker: &SpeakerEmbedding,
    quality: &ConversionQuality,
) -> Result<Vec<f32>, Error> {
    let source_f0 = source_characteristics.f0_mean.max(80.0);
    let target_f0 = target_speaker.avg_f0.max(80.0);
    let pitch_ratio = target_f0 / source_f0;

    let n = source_audio.len();
    if n == 0 {
        return Ok(Vec::new());
    }

    let resampled_len = ((n as f64) / (pitch_ratio as f64)).round().max(1.0) as usize;

    let mut resampled = Vec::with_capacity(resampled_len);
    for i in 0..resampled_len {
        let src_pos = (i as f64) * (pitch_ratio as f64);
        let src_idx = src_pos.floor() as usize;
        let frac = (src_pos - src_pos.floor()) as f32;

        let s0 = source_audio.get(src_idx).copied().unwrap_or(0.0);
        let s1 = source_audio.get(src_idx + 1).copied().unwrap_or(0.0);
        resampled.push(s0 + frac * (s1 - s0));
    }

    let mut output = vec![0.0f32; n];
    let copy_len = resampled_len.min(n);
    output[..copy_len].copy_from_slice(&resampled[..copy_len]);

    let blend = quality.conversion_strength.clamp(0.0, 1.0);
    for (i, out) in output.iter_mut().enumerate() {
        let orig = source_audio[i];
        *out = orig * (1.0 - blend) + *out * blend;
    }

    Ok(output)
}

pub(super) fn hybrid_conversion(
    source_audio: &[f32],
    sample_rate: u32,
    source_characteristics: &VoiceCharacteristics,
    target_speaker: &SpeakerEmbedding,
    quality: &ConversionQuality,
) -> Result<Vec<f32>, Error> {
    let neural_result = neural_transfer_conversion(
        source_audio,
        source_characteristics,
        target_speaker,
        quality,
    )?;
    let spectral_result = spectral_conversion(source_audio, sample_rate, target_speaker, quality)?;

    let blend_factor = quality.conversion_strength;
    let mut hybrid_result = Vec::with_capacity(source_audio.len());

    for (i, &sample) in source_audio.iter().enumerate() {
        let neural_sample = neural_result.get(i).copied().unwrap_or(0.0);
        let spectral_sample = spectral_result.get(i).copied().unwrap_or(0.0);

        let blended = sample * (1.0 - blend_factor)
            + (neural_sample * 0.5 + spectral_sample * 0.5) * blend_factor;
        hybrid_result.push(blended);
    }

    Ok(hybrid_result)
}

pub(super) fn calculate_pitch_shift(
    source: &VoiceCharacteristics,
    target: &VoiceCharacteristics,
) -> f32 {
    let source_f0 = get_average_f0_for_voice_type(source.voice_type);
    let target_f0 = get_average_f0_for_voice_type(target.voice_type);
    target_f0 / source_f0
}

pub(super) fn get_average_f0_for_voice_type(voice_type: crate::types::VoiceType) -> f32 {
    use crate::types::VoiceType;
    match voice_type {
        VoiceType::Soprano => 220.0,
        VoiceType::MezzoSoprano => 196.0,
        VoiceType::Alto => 175.0,
        VoiceType::Tenor => 147.0,
        VoiceType::Baritone => 123.0,
        VoiceType::Bass => 98.0,
    }
}

//! Temporal analysis implementation methods

use super::researcher::AudioQualityResearcher;
use crate::Error;

impl AudioQualityResearcher {
    pub(crate) fn calculate_temporal_coherence(
        &self,
        original: &[f32],
        processed: &[f32],
    ) -> Result<f32, Error> {
        // Calculate frame-by-frame correlation
        let frame_size = 512;
        let mut correlations = Vec::new();

        for i in (0..original.len()).step_by(frame_size) {
            let end_idx = (i + frame_size).min(original.len());
            if end_idx - i >= frame_size / 2 {
                // Ensure minimum frame size
                let orig_frame = &original[i..end_idx];
                let proc_frame = &processed[i..end_idx];
                let correlation = self.calculate_correlation(orig_frame, proc_frame);
                correlations.push(correlation);
            }
        }

        if correlations.is_empty() {
            Ok(1.0)
        } else {
            Ok(correlations.iter().sum::<f32>() / correlations.len() as f32)
        }
    }

    pub(crate) fn calculate_envelope_correlation(
        &self,
        original: &[f32],
        processed: &[f32],
    ) -> Result<f32, Error> {
        let orig_envelope = self.calculate_envelope(original);
        let proc_envelope = self.calculate_envelope(processed);
        Ok(self.calculate_correlation(&orig_envelope, &proc_envelope))
    }

    pub(crate) fn calculate_zcr_deviation(
        &self,
        original: &[f32],
        processed: &[f32],
    ) -> Result<f32, Error> {
        let orig_zcr = self.calculate_zero_crossing_rate(original);
        let proc_zcr = self.calculate_zero_crossing_rate(processed);
        Ok((orig_zcr - proc_zcr).abs())
    }

    pub(crate) fn calculate_temporal_smoothness(&self, audio: &[f32]) -> Result<f32, Error> {
        if audio.len() < 2 {
            return Ok(1.0);
        }

        let mut smoothness = 0.0;
        for i in 1..audio.len() {
            smoothness += (audio[i] - audio[i - 1]).abs();
        }

        // Normalize and invert (higher values = smoother)
        let normalized_roughness = smoothness / (audio.len() - 1) as f32;
        Ok((1.0 / (1.0 + normalized_roughness * 10.0)).clamp(0.0, 1.0))
    }

    pub(crate) fn calculate_phase_coherence(
        &self,
        original: &[f32],
        processed: &[f32],
    ) -> Result<f32, Error> {
        // Simplified phase coherence based on instantaneous phase differences
        let frame_size = 256;
        let mut phase_differences = Vec::new();

        for chunk in original
            .chunks(frame_size)
            .zip(processed.chunks(frame_size))
        {
            let (orig_chunk, proc_chunk) = chunk;
            if orig_chunk.len() == proc_chunk.len() && orig_chunk.len() >= 16 {
                let orig_phase = self.calculate_instantaneous_phase(orig_chunk);
                let proc_phase = self.calculate_instantaneous_phase(proc_chunk);

                for (&op, &pp) in orig_phase.iter().zip(proc_phase.iter()) {
                    let phase_diff = ((op - pp + std::f32::consts::PI)
                        % (2.0 * std::f32::consts::PI)
                        - std::f32::consts::PI)
                        .abs();
                    phase_differences.push(1.0 - phase_diff / std::f32::consts::PI);
                }
            }
        }

        if phase_differences.is_empty() {
            Ok(1.0)
        } else {
            Ok(phase_differences.iter().sum::<f32>() / phase_differences.len() as f32)
        }
    }

    pub(crate) fn calculate_rhythm_preservation(
        &self,
        original: &[f32],
        processed: &[f32],
    ) -> Result<f32, Error> {
        let orig_energy_envelope = self.calculate_energy_envelope(original);
        let proc_energy_envelope = self.calculate_energy_envelope(processed);
        Ok(self.calculate_correlation(&orig_energy_envelope, &proc_energy_envelope))
    }
}

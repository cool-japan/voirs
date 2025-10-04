//! CLI-specific wrapper types for VoiRS types.

use clap::ValueEnum;
use std::str::FromStr;
use voirs_sdk::{AudioFormat, QualityLevel};

/// CLI wrapper for AudioFormat to implement clap traits
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum CliAudioFormat {
    Wav,
    Flac,
    Mp3,
    Opus,
    Ogg,
}

impl From<CliAudioFormat> for AudioFormat {
    fn from(cli_format: CliAudioFormat) -> Self {
        match cli_format {
            CliAudioFormat::Wav => AudioFormat::Wav,
            CliAudioFormat::Flac => AudioFormat::Flac,
            CliAudioFormat::Mp3 => AudioFormat::Mp3,
            CliAudioFormat::Opus => AudioFormat::Opus,
            CliAudioFormat::Ogg => AudioFormat::Ogg,
        }
    }
}

impl From<AudioFormat> for CliAudioFormat {
    fn from(format: AudioFormat) -> Self {
        match format {
            AudioFormat::Wav => CliAudioFormat::Wav,
            AudioFormat::Flac => CliAudioFormat::Flac,
            AudioFormat::Mp3 => CliAudioFormat::Mp3,
            AudioFormat::Opus => CliAudioFormat::Opus,
            AudioFormat::Ogg => CliAudioFormat::Ogg,
        }
    }
}

impl Default for CliAudioFormat {
    fn default() -> Self {
        CliAudioFormat::Wav
    }
}

/// CLI wrapper for QualityLevel to implement clap traits
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum CliQualityLevel {
    Low,
    Medium,
    High,
    Ultra,
}

impl From<CliQualityLevel> for QualityLevel {
    fn from(cli_quality: CliQualityLevel) -> Self {
        match cli_quality {
            CliQualityLevel::Low => QualityLevel::Low,
            CliQualityLevel::Medium => QualityLevel::Medium,
            CliQualityLevel::High => QualityLevel::High,
            CliQualityLevel::Ultra => QualityLevel::Ultra,
        }
    }
}

impl From<QualityLevel> for CliQualityLevel {
    fn from(quality: QualityLevel) -> Self {
        match quality {
            QualityLevel::Low => CliQualityLevel::Low,
            QualityLevel::Medium => CliQualityLevel::Medium,
            QualityLevel::High => CliQualityLevel::High,
            QualityLevel::Ultra => CliQualityLevel::Ultra,
        }
    }
}

impl Default for CliQualityLevel {
    fn default() -> Self {
        CliQualityLevel::High
    }
}

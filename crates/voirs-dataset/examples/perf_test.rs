//! Audio loading performance test
//!
//! Simple performance test to measure audio loading times for wav files.

use hound::WavSpec;
use std::time::Instant;
use tempfile::TempDir;
use voirs_dataset::audio::io::load_wav;

fn main() {
    let temp_dir = TempDir::new().unwrap();
    let path = temp_dir.path().join("test.wav");

    let spec = WavSpec {
        channels: 1,
        sample_rate: 22050,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = hound::WavWriter::create(&path, spec).unwrap();
    for i in 0..22050 {
        let sample = (i as f32 * 0.01).sin() * 16384.0;
        writer.write_sample(sample as i16).unwrap();
    }
    writer.finalize().unwrap();

    let start = Instant::now();
    let audio = load_wav(&path).unwrap();
    let duration = start.elapsed();

    println!("Loaded {} samples in {:?}", audio.samples().len(), duration);
}

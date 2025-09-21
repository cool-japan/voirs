#!/usr/bin/env python3
"""
VoiRS Evaluation Python Integration Example

This example demonstrates how to use the VoiRS evaluation system Python bindings
with NumPy, SciPy, Pandas, and Matplotlib for comprehensive audio quality analysis.

Requirements:
    pip install voirs-evaluation numpy scipy pandas matplotlib librosa soundfile

Usage:
    python examples/python_integration.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy.signal as signal
import librosa
import soundfile as sf
from typing import List, Tuple, Dict, Any
import os
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

try:
    import voirs_evaluation as ve
    VOIRS_AVAILABLE = True
except ImportError:
    print("VoiRS evaluation not available. Install with: pip install voirs-evaluation")
    VOIRS_AVAILABLE = False


def generate_test_audio(duration: float = 1.0, sample_rate: int = 16000) -> np.ndarray:
    """Generate test audio signal with multiple components."""
    t = np.linspace(0, duration, int(duration * sample_rate), False)
    
    # Create a complex test signal
    signal = (
        0.3 * np.sin(2 * np.pi * 440 * t) +  # A4 tone
        0.2 * np.sin(2 * np.pi * 880 * t) +  # A5 tone (octave)
        0.1 * np.sin(2 * np.pi * 1320 * t)   # E6 tone (fifth)
    )
    
    # Add slight amplitude modulation (vibrato)
    vibrato = 1 + 0.05 * np.sin(2 * np.pi * 6 * t)
    signal *= vibrato
    
    # Add gentle fade in/out
    fade_samples = int(0.1 * sample_rate)
    signal[:fade_samples] *= np.linspace(0, 1, fade_samples)
    signal[-fade_samples:] *= np.linspace(1, 0, fade_samples)
    
    return signal.astype(np.float32)


def add_realistic_noise(signal: np.ndarray, snr_db: float = 20.0) -> np.ndarray:
    """Add realistic noise to the signal at specified SNR."""
    # Generate colored noise (more realistic than white noise)
    noise = np.random.randn(len(signal))
    
    # Apply 1/f filter to create pink noise
    freqs = np.fft.fftfreq(len(noise))
    freqs[0] = 1e-8  # Avoid division by zero
    filter_response = 1 / np.sqrt(np.abs(freqs))
    filter_response[0] = 0
    
    noise_fft = np.fft.fft(noise)
    colored_noise_fft = noise_fft * filter_response
    colored_noise = np.real(np.fft.ifft(colored_noise_fft))
    
    # Normalize noise to achieve target SNR
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(colored_noise ** 2)
    noise_scaling = np.sqrt(signal_power / (noise_power * 10**(snr_db / 10)))
    
    return signal + noise_scaling * colored_noise


def simulate_codec_degradation(signal: np.ndarray, sample_rate: int, bitrate: int = 64) -> np.ndarray:
    """Simulate codec degradation by downsampling and requantization."""
    # Simulate lower bitrate by reducing bit depth
    if bitrate <= 32:
        bits = 8
    elif bitrate <= 64:
        bits = 12
    else:
        bits = 16
    
    # Quantize signal
    max_val = 2 ** (bits - 1) - 1
    quantized = np.round(signal * max_val) / max_val
    
    # Add slight DC offset and distortion typical of low-bitrate codecs
    quantized += 0.001 * np.random.randn(len(quantized))
    
    return quantized.astype(np.float32)


def comprehensive_evaluation_demo():
    """Demonstrate comprehensive evaluation with statistical analysis."""
    if not VOIRS_AVAILABLE:
        print("VoiRS evaluation not available. Skipping evaluation demo.")
        return
    
    print("🎵 VoiRS Evaluation Python Integration Demo")
    print("=" * 50)
    
    # Generate test dataset
    print("\n📊 Generating test dataset...")
    sample_rate = 16000
    duration = 2.0
    num_samples = 10
    
    # Create reference audio
    reference_audio = generate_test_audio(duration, sample_rate)
    
    # Create variations with different quality levels
    test_conditions = [
        ("Original", reference_audio),
        ("Light Noise (30dB SNR)", add_realistic_noise(reference_audio, 30.0)),
        ("Moderate Noise (20dB SNR)", add_realistic_noise(reference_audio, 20.0)),
        ("Heavy Noise (10dB SNR)", add_realistic_noise(reference_audio, 10.0)),
        ("Codec 128kbps", simulate_codec_degradation(reference_audio, sample_rate, 128)),
        ("Codec 64kbps", simulate_codec_degradation(reference_audio, sample_rate, 64)),
        ("Codec 32kbps", simulate_codec_degradation(reference_audio, sample_rate, 32)),
    ]
    
    # Initialize evaluators
    quality_evaluator = ve.PyQualityEvaluator()
    statistical_analyzer = ve.PyStatisticalAnalyzer()
    
    print(f"✅ Created {len(test_conditions)} test conditions")
    
    # Evaluate all conditions
    print("\n🔍 Evaluating audio quality...")
    results = []
    
    for condition_name, degraded_audio in test_conditions:
        try:
            result = quality_evaluator.evaluate(
                reference_audio,
                degraded_audio,
                sample_rate
            )
            
            # Convert to dictionary for easier handling
            result_dict = result.to_dict()
            result_dict['condition'] = condition_name
            result_dict['snr_calculated'] = ve.calculate_snr(reference_audio, degraded_audio - reference_audio)
            
            results.append(result_dict)
            print(f"  ✓ {condition_name}: PESQ={result_dict['pesq']:.3f}, STOI={result_dict['stoi']:.3f}")
            
        except Exception as e:
            print(f"  ❌ {condition_name}: Error - {e}")
    
    # Convert to pandas DataFrame for analysis
    df = pd.DataFrame(results)
    print(f"\n📋 Created evaluation results DataFrame with {len(df)} rows")
    print("\nDataFrame Summary:")
    print(df[['condition', 'pesq', 'stoi', 'mcd', 'overall_score']].round(3))
    
    # Statistical analysis
    print("\n📈 Statistical Analysis")
    print("-" * 30)
    
    # Correlation analysis between metrics
    metrics = ['pesq', 'stoi', 'mcd', 'overall_score', 'snr_calculated']
    correlation_matrix = df[metrics].corr()
    
    print("\nCorrelation Matrix:")
    print(correlation_matrix.round(3))
    
    # Perform statistical tests
    print("\n🧮 Statistical Tests:")
    
    # Compare original vs different noise levels
    original_scores = df[df['condition'] == 'Original']['overall_score'].values
    for condition in ['Light Noise (30dB SNR)', 'Heavy Noise (10dB SNR)', 'Codec 32kbps']:
        condition_scores = df[df['condition'] == condition]['overall_score'].values
        if len(condition_scores) > 0 and len(original_scores) > 0:
            # Use scipy for t-test
            t_stat, p_value = stats.ttest_ind(original_scores, condition_scores)
            effect_size = (np.mean(original_scores) - np.mean(condition_scores)) / np.sqrt(
                (np.var(original_scores) + np.var(condition_scores)) / 2
            )
            print(f"  {condition} vs Original: t={t_stat:.3f}, p={p_value:.3f}, effect_size={effect_size:.3f}")
    
    # Visualizations
    print("\n📊 Creating visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('VoiRS Evaluation Results Analysis', fontsize=16, fontweight='bold')
    
    # 1. Quality metrics comparison
    ax1 = axes[0, 0]
    metrics_to_plot = ['pesq', 'stoi', 'overall_score']
    x_pos = np.arange(len(df))
    width = 0.25
    
    for i, metric in enumerate(metrics_to_plot):
        ax1.bar(x_pos + i * width, df[metric], width, label=metric.upper(), alpha=0.7)
    
    ax1.set_xlabel('Test Conditions')
    ax1.set_ylabel('Score')
    ax1.set_title('Quality Metrics Comparison')
    ax1.set_xticks(x_pos + width)
    ax1.set_xticklabels(df['condition'], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. SNR vs Quality correlation
    ax2 = axes[0, 1]
    valid_snr = df['snr_calculated'][np.isfinite(df['snr_calculated'])]
    valid_quality = df['overall_score'][np.isfinite(df['snr_calculated'])]
    
    if len(valid_snr) > 0:
        ax2.scatter(valid_snr, valid_quality, alpha=0.7, s=60)
        
        # Add trend line
        if len(valid_snr) > 1:
            z = np.polyfit(valid_snr, valid_quality, 1)
            p = np.poly1d(z)
            ax2.plot(valid_snr, p(valid_snr), "r--", alpha=0.8)
        
        ax2.set_xlabel('SNR (dB)')
        ax2.set_ylabel('Overall Quality Score')
        ax2.set_title('SNR vs Quality Correlation')
        ax2.grid(True, alpha=0.3)
    
    # 3. Metric distributions
    ax3 = axes[1, 0]
    df['overall_score'].hist(bins=10, alpha=0.7, ax=ax3, edgecolor='black')
    ax3.axvline(df['overall_score'].mean(), color='red', linestyle='--', 
                label=f'Mean: {df["overall_score"].mean():.3f}')
    ax3.set_xlabel('Overall Quality Score')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Quality Score Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Correlation heatmap
    ax4 = axes[1, 1]
    im = ax4.imshow(correlation_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    ax4.set_xticks(range(len(metrics)))
    ax4.set_yticks(range(len(metrics)))
    ax4.set_xticklabels(metrics, rotation=45, ha='right')
    ax4.set_yticklabels(metrics)
    ax4.set_title('Metrics Correlation Heatmap')
    
    # Add correlation values to heatmap
    for i in range(len(metrics)):
        for j in range(len(metrics)):
            text = ax4.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                           ha="center", va="center", color="black", fontweight='bold')
    
    plt.colorbar(im, ax=ax4)
    plt.tight_layout()
    
    # Save the plot
    output_path = "voirs_evaluation_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 Saved analysis plot to: {output_path}")
    
    # Export results to CSV
    csv_path = "voirs_evaluation_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"💾 Saved results to: {csv_path}")
    
    print("\n✅ Analysis complete!")
    return df


def pronunciation_evaluation_demo():
    """Demonstrate pronunciation evaluation capabilities."""
    if not VOIRS_AVAILABLE:
        print("VoiRS evaluation not available. Skipping pronunciation demo.")
        return
    
    print("\n🗣️  Pronunciation Evaluation Demo")
    print("=" * 40)
    
    # Generate test audio for pronunciation
    sample_rate = 16000
    test_audio = generate_test_audio(1.5, sample_rate)
    
    # Create pronunciation evaluator
    pron_evaluator = ve.PyPronunciationEvaluator()
    
    # Test different reference texts
    test_cases = [
        "Hello world",
        "The quick brown fox jumps over the lazy dog",
        "Speech recognition and synthesis technology",
        "Artificial intelligence and machine learning",
    ]
    
    print("Evaluating pronunciation for different texts:")
    pronunciation_results = []
    
    for text in test_cases:
        try:
            result = pron_evaluator.evaluate(test_audio, text, sample_rate)
            result_dict = result.to_dict()
            result_dict['text'] = text
            pronunciation_results.append(result_dict)
            
            print(f"  '{text}': Overall={result_dict['overall_score']:.3f}, "
                  f"Phoneme={result_dict['phoneme_accuracy']:.3f}")
            
        except Exception as e:
            print(f"  Error evaluating '{text}': {e}")
    
    # Create DataFrame for pronunciation results
    if pronunciation_results:
        pron_df = pd.DataFrame(pronunciation_results)
        print(f"\n📋 Pronunciation evaluation completed for {len(pron_df)} texts")
        print("\nPronunciation Summary:")
        print(pron_df[['text', 'overall_score', 'phoneme_accuracy', 'fluency_score']].round(3))
        
        return pron_df
    
    return None


def real_time_simulation_demo():
    """Demonstrate real-time evaluation capabilities."""
    print("\n⚡ Real-time Evaluation Simulation")
    print("=" * 40)
    
    # Simulate streaming audio chunks
    sample_rate = 16000
    chunk_duration = 0.1  # 100ms chunks
    chunk_size = int(chunk_duration * sample_rate)
    total_duration = 2.0
    total_chunks = int(total_duration / chunk_duration)
    
    print(f"Simulating {total_chunks} audio chunks of {chunk_duration}s each")
    
    # Generate continuous test signal
    full_audio = generate_test_audio(total_duration, sample_rate)
    
    # Simulate real-time processing
    chunk_qualities = []
    
    for i in range(total_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(full_audio))
        chunk = full_audio[start_idx:end_idx]
        
        # Add some processing variation
        chunk_with_noise = add_realistic_noise(chunk, snr_db=25 + 10 * np.sin(i * 0.5))
        
        # Calculate simple quality metrics
        snr = ve.calculate_snr(chunk, chunk_with_noise - chunk) if len(chunk) > 0 else 0
        
        chunk_qualities.append({
            'chunk_id': i,
            'timestamp': i * chunk_duration,
            'snr': snr,
            'rms': np.sqrt(np.mean(chunk ** 2)),
            'peak': np.max(np.abs(chunk)),
        })
    
    # Create DataFrame for real-time results
    rt_df = pd.DataFrame(chunk_qualities)
    
    print(f"📊 Processed {len(rt_df)} chunks")
    print(f"Average SNR: {rt_df['snr'].mean():.2f} dB")
    print(f"RMS range: {rt_df['rms'].min():.4f} - {rt_df['rms'].max():.4f}")
    
    # Plot real-time metrics
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(rt_df['timestamp'], rt_df['snr'], 'b-', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('SNR (dB)')
    plt.title('Real-time SNR Monitoring')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    plt.plot(rt_df['timestamp'], rt_df['rms'], 'g-', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('RMS Level')
    plt.title('Real-time RMS Monitoring')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    plt.plot(rt_df['timestamp'], rt_df['peak'], 'r-', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Peak Level')
    plt.title('Real-time Peak Monitoring')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    plt.hist(rt_df['snr'], bins=15, alpha=0.7, edgecolor='black')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Frequency')
    plt.title('SNR Distribution')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('realtime_monitoring.png', dpi=300, bbox_inches='tight')
    print("📊 Saved real-time monitoring plot to: realtime_monitoring.png")
    
    return rt_df


def main():
    """Main demo function."""
    print("🚀 VoiRS Evaluation Python Integration")
    print("=====================================")
    print("This demo showcases the integration of VoiRS evaluation")
    print("with Python scientific computing tools.")
    print()
    
    # Run all demos
    try:
        # 1. Comprehensive evaluation with statistical analysis
        quality_df = comprehensive_evaluation_demo()
        
        # 2. Pronunciation evaluation
        pronunciation_df = pronunciation_evaluation_demo()
        
        # 3. Real-time simulation
        realtime_df = real_time_simulation_demo()
        
        print("\n🎉 All demos completed successfully!")
        print("\nGenerated files:")
        print("  📊 voirs_evaluation_analysis.png - Quality analysis plots")
        print("  📄 voirs_evaluation_results.csv - Evaluation results data")
        print("  📊 realtime_monitoring.png - Real-time monitoring plots")
        
        if VOIRS_AVAILABLE:
            print(f"\n📈 Summary Statistics:")
            if quality_df is not None:
                print(f"  Quality evaluations: {len(quality_df)} conditions tested")
                print(f"  Average PESQ: {quality_df['pesq'].mean():.3f}")
                print(f"  Average STOI: {quality_df['stoi'].mean():.3f}")
            
            if pronunciation_df is not None:
                print(f"  Pronunciation tests: {len(pronunciation_df)} texts evaluated")
                print(f"  Average pronunciation score: {pronunciation_df['overall_score'].mean():.3f}")
            
            if realtime_df is not None:
                print(f"  Real-time chunks: {len(realtime_df)} processed")
                print(f"  Average SNR: {realtime_df['snr'].mean():.2f} dB")
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
# Scientific Computing Integration Examples

This document provides comprehensive examples for integrating VoiRS FFI with popular scientific computing frameworks and environments for research, data analysis, and computational workflows.

## Table of Contents

1. [Jupyter Notebook Integration](#jupyter-notebook-integration)
2. [NumPy/SciPy Integration](#numpyscipy-integration)
3. [Pandas Data Processing](#pandas-data-processing)
4. [MATLAB Integration](#matlab-integration)
5. [R Integration](#r-integration)
6. [Apache Spark Integration](#apache-spark-integration)

## Jupyter Notebook Integration

### VoiRS Jupyter Magic Commands

#### Installation and Setup (setup.py)

```python
from setuptools import setup, find_packages

setup(
    name="voirs-jupyter",
    version="0.1.0",
    description="VoiRS FFI integration for Jupyter notebooks",
    packages=find_packages(),
    install_requires=[
        "jupyter",
        "ipython",
        "ipywidgets",
        "matplotlib",
        "librosa",
        "soundfile",
        "numpy",
        "pandas",
        "voirs-ffi>=0.1.0"
    ],
    entry_points={
        'console_scripts': [
            'voirs-jupyter-install=voirs_jupyter.install:main',
        ],
    }
)
```

#### Jupyter Magic Extension (voirs_jupyter/magic.py)

```python
from IPython.core.magic import Magics, magics_class, line_magic, cell_magic
from IPython.core.magic_arguments import (argument, magic_arguments, parse_argstring)
from IPython.display import Audio, display, HTML
import voirs_ffi
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf
import tempfile
import io
import base64
from typing import Optional, Dict, Any

@magics_class
class VoiRSMagics(Magics):
    def __init__(self, shell):
        super().__init__(shell)
        self.engine = None
        self.last_result = None
        self.synthesis_history = []
        self.initialize_engine()
    
    def initialize_engine(self):
        """Initialize VoiRS engine with optimal settings for research"""
        try:
            self.engine = voirs_ffi.Engine()
            config = voirs_ffi.SynthesisConfig(
                quality=voirs_ffi.Quality.HIGH,
                thread_count=2,  # Conservative for notebook environment
                cache_size=512 * 1024  # 512KB cache
            )
            
            if self.engine.initialize(config):
                print("✓ VoiRS engine initialized successfully")
            else:
                print("✗ Failed to initialize VoiRS engine")
                self.engine = None
        except Exception as e:
            print(f"✗ VoiRS initialization error: {e}")
            self.engine = None
    
    @line_magic
    @magic_arguments()
    @argument('--voice', '-v', type=str, default='default', help='Voice ID to use')
    @argument('--quality', '-q', type=str, default='high', 
              choices=['low', 'medium', 'high', 'ultra'], help='Synthesis quality')
    @argument('--speed', '-s', type=float, default=1.0, help='Speech speed (0.1-3.0)')
    @argument('--volume', type=float, default=1.0, help='Volume (0.0-2.0)')
    @argument('--format', '-f', type=str, default='wav', 
              choices=['wav', 'mp3', 'flac'], help='Output format')
    @argument('--display', '-d', action='store_true', help='Display audio player')
    @argument('--analyze', '-a', action='store_true', help='Show audio analysis')
    @argument('--save', type=str, help='Save audio to file')
    def voirs(self, line):
        """
        Synthesize text using VoiRS.
        
        Usage: %voirs [options] text to synthesize
        """
        if not self.engine:
            print("VoiRS engine not available")
            return
        
        args = parse_argstring(self.voirs, line)
        text_parts = line.split()
        
        # Extract text (everything after the last option)
        text_start = 0
        for i, part in enumerate(text_parts):
            if not part.startswith('-'):
                text_start = i
                break
        
        text = ' '.join(text_parts[text_start:]) if text_start < len(text_parts) else ""
        
        if not text.strip():
            print("Please provide text to synthesize")
            return
        
        try:
            # Create synthesis config
            config = voirs_ffi.SynthesisConfig(
                quality=getattr(voirs_ffi.Quality, args.quality.upper()),
                speed=args.speed,
                volume=args.volume,
                voice_id=args.voice,
                output_format=getattr(voirs_ffi.Format, args.format.upper())
            )
            
            # Perform synthesis
            result = self.engine.synthesize(text, config)
            
            if result.success:
                self.last_result = {
                    'text': text,
                    'audio_data': result.audio_data,
                    'sample_rate': result.sample_rate,
                    'duration': result.duration,
                    'config': config
                }
                
                # Add to history
                self.synthesis_history.append(self.last_result.copy())
                
                print(f"✓ Synthesis completed ({result.duration:.2f}s)")
                
                # Display audio player
                if args.display:
                    self.display_audio(result.audio_data, result.sample_rate)
                
                # Show analysis
                if args.analyze:
                    self.analyze_audio(result.audio_data, result.sample_rate)
                
                # Save to file
                if args.save:
                    self.save_audio(result.audio_data, result.sample_rate, args.save)
                
                return result.audio_data
                
            else:
                print("✗ Synthesis failed")
                return None
                
        except Exception as e:
            print(f"✗ Error: {e}")
            return None
    
    @cell_magic
    @magic_arguments()
    @argument('--voice', '-v', type=str, default='default', help='Voice ID to use')
    @argument('--quality', '-q', type=str, default='high', help='Synthesis quality')
    @argument('--speed', '-s', type=float, default=1.0, help='Speech speed')
    @argument('--batch', '-b', action='store_true', help='Process as batch')
    @argument('--compare', '-c', action='store_true', help='Compare different voices')
    def voirs_cell(self, line, cell):
        """
        Synthesize multi-line text or process batch synthesis.
        
        Usage: %%voirs_cell [options]
               Text to synthesize
               across multiple lines
        """
        if not self.engine:
            print("VoiRS engine not available")
            return
        
        args = parse_argstring(self.voirs_cell, line)
        
        if args.batch:
            return self.process_batch(cell, args)
        elif args.compare:
            return self.compare_voices(cell, args)
        else:
            return self.synthesize_long_text(cell, args)
    
    def process_batch(self, cell_content: str, args) -> Dict[str, Any]:
        """Process multiple texts as batch"""
        texts = [line.strip() for line in cell_content.split('\n') if line.strip()]
        
        if not texts:
            print("No texts found for batch processing")
            return None
        
        print(f"Processing batch of {len(texts)} texts...")
        
        results = []
        config = voirs_ffi.SynthesisConfig(
            quality=getattr(voirs_ffi.Quality, args.quality.upper()),
            speed=args.speed,
            voice_id=args.voice
        )
        
        for i, text in enumerate(texts, 1):
            try:
                result = self.engine.synthesize(text, config)
                if result.success:
                    results.append({
                        'index': i - 1,
                        'text': text,
                        'success': True,
                        'audio_data': result.audio_data,
                        'duration': result.duration,
                        'sample_rate': result.sample_rate
                    })
                    print(f"  ✓ {i}/{len(texts)}: {text[:50]}...")
                else:
                    results.append({
                        'index': i - 1,
                        'text': text,
                        'success': False,
                        'error': 'Synthesis failed'
                    })
                    print(f"  ✗ {i}/{len(texts)}: Failed")
                    
            except Exception as e:
                results.append({
                    'index': i - 1,
                    'text': text,
                    'success': False,
                    'error': str(e)
                })
                print(f"  ✗ {i}/{len(texts)}: Error - {e}")
        
        # Create DataFrame for analysis
        df = pd.DataFrame([
            {
                'text': r['text'],
                'success': r['success'],
                'duration': r.get('duration', 0),
                'length': len(r['text']),
                'words': len(r['text'].split()),
                'error': r.get('error', '')
            }
            for r in results
        ])
        
        print(f"\nBatch Summary:")
        print(f"  Total: {len(results)}")
        print(f"  Successful: {df['success'].sum()}")
        print(f"  Failed: {(~df['success']).sum()}")
        print(f"  Total duration: {df['duration'].sum():.2f}s")
        print(f"  Average duration: {df[df['success']]['duration'].mean():.2f}s")
        
        return {
            'results': results,
            'summary': df,
            'audio_data': [r.get('audio_data') for r in results if r['success']]
        }
    
    def compare_voices(self, cell_content: str, args) -> Dict[str, Any]:
        """Compare different voices with the same text"""
        text = cell_content.strip()
        if not text:
            print("No text provided for voice comparison")
            return None
        
        # Available voices for comparison
        voices = ['default', 'male_young', 'female_young', 'male_deep', 'female_warm']
        
        print(f"Comparing voices for: {text[:50]}...")
        
        results = {}
        for voice in voices:
            try:
                config = voirs_ffi.SynthesisConfig(
                    quality=getattr(voirs_ffi.Quality, args.quality.upper()),
                    speed=args.speed,
                    voice_id=voice
                )
                
                result = self.engine.synthesize(text, config)
                if result.success:
                    results[voice] = {
                        'audio_data': result.audio_data,
                        'duration': result.duration,
                        'sample_rate': result.sample_rate,
                        'success': True
                    }
                    print(f"  ✓ {voice}: {result.duration:.2f}s")
                else:
                    results[voice] = {'success': False, 'error': 'Synthesis failed'}
                    print(f"  ✗ {voice}: Failed")
                    
            except Exception as e:
                results[voice] = {'success': False, 'error': str(e)}
                print(f"  ✗ {voice}: Error - {e}")
        
        # Display audio players for comparison
        successful_voices = [v for v, r in results.items() if r['success']]
        if successful_voices:
            html = "<div style='display: flex; flex-wrap: wrap; gap: 20px;'>"
            for voice in successful_voices:
                audio_base64 = base64.b64encode(results[voice]['audio_data']).decode()
                html += f"""
                <div style='border: 1px solid #ccc; padding: 10px; border-radius: 5px;'>
                    <h4>{voice}</h4>
                    <audio controls>
                        <source src="data:audio/wav;base64,{audio_base64}" type="audio/wav">
                    </audio>
                    <p>Duration: {results[voice]['duration']:.2f}s</p>
                </div>
                """
            html += "</div>"
            display(HTML(html))
        
        return results
    
    def display_audio(self, audio_data: bytes, sample_rate: int):
        """Display audio player in notebook"""
        display(Audio(audio_data, rate=sample_rate))
    
    def analyze_audio(self, audio_data: bytes, sample_rate: int):
        """Perform and display audio analysis"""
        # Convert audio data to numpy array
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            sf.write(tmp_file.name, 
                    np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0,
                    sample_rate)
            
            # Load with librosa for analysis
            y, sr = librosa.load(tmp_file.name, sr=sample_rate)
        
        # Create analysis plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Waveform
        axes[0, 0].plot(np.linspace(0, len(y)/sr, len(y)), y)
        axes[0, 0].set_title('Waveform')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Amplitude')
        
        # Spectrogram
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        librosa.display.specshow(D, y_axis='linear', x_axis='time', 
                               sr=sr, ax=axes[0, 1])
        axes[0, 1].set_title('Spectrogram')
        
        # Mel spectrogram
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
        mel_spec_db = librosa.amplitude_to_db(mel_spec, ref=np.max)
        librosa.display.specshow(mel_spec_db, y_axis='mel', x_axis='time',
                               sr=sr, ax=axes[1, 0])
        axes[1, 0].set_title('Mel Spectrogram')
        
        # Fundamental frequency
        f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'),
                                                   fmax=librosa.note_to_hz('C7'))
        times = librosa.frames_to_time(range(len(f0)), sr=sr)
        axes[1, 1].plot(times, f0, 'o', markersize=1)
        axes[1, 1].set_title('Fundamental Frequency')
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('F0 (Hz)')
        
        plt.tight_layout()
        plt.show()
        
        # Print statistics
        print("\nAudio Analysis:")
        print(f"  Duration: {len(y)/sr:.2f}s")
        print(f"  Sample Rate: {sr} Hz")
        print(f"  RMS Energy: {np.sqrt(np.mean(y**2)):.4f}")
        print(f"  Zero Crossing Rate: {np.mean(librosa.feature.zero_crossing_rate(y)):.4f}")
        print(f"  Spectral Centroid: {np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)):.2f} Hz")
        print(f"  Spectral Rolloff: {np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)):.2f} Hz")
    
    def save_audio(self, audio_data: bytes, sample_rate: int, filename: str):
        """Save audio to file"""
        try:
            # Convert to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            sf.write(filename, audio_array, sample_rate)
            print(f"✓ Audio saved to {filename}")
        except Exception as e:
            print(f"✗ Failed to save audio: {e}")
    
    @line_magic
    def voirs_history(self, line):
        """Show synthesis history"""
        if not self.synthesis_history:
            print("No synthesis history available")
            return
        
        df = pd.DataFrame([
            {
                'index': i,
                'text': h['text'][:50] + ('...' if len(h['text']) > 50 else ''),
                'duration': h['duration'],
                'voice': h['config'].voice_id,
                'quality': h['config'].quality.name,
                'speed': h['config'].speed
            }
            for i, h in enumerate(self.synthesis_history)
        ])
        
        display(df)
        return df
    
    @line_magic
    def voirs_clear(self, line):
        """Clear synthesis history"""
        self.synthesis_history.clear()
        self.last_result = None
        print("✓ Synthesis history cleared")

# Register the magic
def load_ipython_extension(ipython):
    ipython.register_magic_function(VoiRSMagics)
```

#### Example Jupyter Notebook

```python
# Cell 1: Load the VoiRS extension
%load_ext voirs_jupyter.magic

# Cell 2: Basic synthesis
%voirs --display --analyze Hello, this is a VoiRS demonstration in Jupyter!

# Cell 3: Compare different voices
%%voirs_cell --compare
Welcome to the scientific computing integration of VoiRS FFI.

# Cell 4: Batch processing
%%voirs_cell --batch --quality high
The quick brown fox jumps over the lazy dog.
Data science is revolutionizing how we understand the world.
Machine learning models require large datasets for training.
VoiRS provides high-quality text-to-speech synthesis.
Jupyter notebooks are excellent for interactive research.

# Cell 5: Analyze synthesis history
%voirs_history

# Cell 6: Advanced analysis with custom code
import numpy as np
import matplotlib.pyplot as plt
import librosa

# Get the last synthesis result
if hasattr(get_ipython().magic_voirs, 'last_result') and get_ipython().magic_voirs.last_result:
    result = get_ipython().magic_voirs.last_result
    
    # Convert audio data for analysis
    audio_float = np.frombuffer(result['audio_data'], dtype=np.int16).astype(np.float32) / 32768.0
    
    # Extract acoustic features
    mfccs = librosa.feature.mfcc(y=audio_float, sr=result['sample_rate'], n_mfcc=13)
    chroma = librosa.feature.chroma(y=audio_float, sr=result['sample_rate'])
    spectral_contrast = librosa.feature.spectral_contrast(y=audio_float, sr=result['sample_rate'])
    
    # Plot features
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    librosa.display.specshow(mfccs, x_axis='time', ax=axes[0])
    axes[0].set_title('MFCC Features')
    axes[0].set_ylabel('MFCC')
    
    librosa.display.specshow(chroma, x_axis='time', y_axis='chroma', ax=axes[1])
    axes[1].set_title('Chroma Features')
    
    librosa.display.specshow(spectral_contrast, x_axis='time', ax=axes[2])
    axes[2].set_title('Spectral Contrast')
    axes[2].set_ylabel('Frequency bands')
    
    plt.tight_layout()
    plt.show()
    
    print(f"Feature dimensions:")
    print(f"  MFCC: {mfccs.shape}")
    print(f"  Chroma: {chroma.shape}")
    print(f"  Spectral Contrast: {spectral_contrast.shape}")
```

## NumPy/SciPy Integration

### Scientific Audio Processing

#### VoiRS NumPy Interface (voirs_numpy.py)

```python
import numpy as np
import scipy.signal
import scipy.fft
from scipy.io import wavfile
import voirs_ffi
from typing import Tuple, Optional, List, Dict, Any
import warnings

class VoiRSNumPyInterface:
    """NumPy-optimized interface for VoiRS FFI"""
    
    def __init__(self):
        self.engine = voirs_ffi.Engine()
        self.sample_rate = 22050  # Default sample rate
        self.initialized = False
        
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """Initialize VoiRS engine with scientific computing optimizations"""
        default_config = {
            'quality': voirs_ffi.Quality.HIGH,
            'thread_count': 1,  # Single-threaded for reproducibility
            'use_simd': True,   # Enable SIMD for performance
            'cache_size': 2 * 1024 * 1024  # 2MB cache
        }
        
        if config:
            default_config.update(config)
        
        voirs_config = voirs_ffi.SynthesisConfig(**default_config)
        self.initialized = self.engine.initialize(voirs_config)
        return self.initialized
    
    def synthesize_to_array(self, text: str, 
                          config: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, int]:
        """
        Synthesize text and return as NumPy array
        
        Returns:
            Tuple of (audio_array, sample_rate)
        """
        if not self.initialized:
            raise RuntimeError("VoiRS engine not initialized")
        
        # Default synthesis configuration
        synthesis_config = voirs_ffi.SynthesisConfig(
            quality=voirs_ffi.Quality.HIGH,
            output_format=voirs_ffi.Format.WAV
        )
        
        if config:
            for key, value in config.items():
                if hasattr(synthesis_config, key):
                    setattr(synthesis_config, key, value)
        
        # Perform synthesis
        result = self.engine.synthesize(text, synthesis_config)
        
        if not result.success:
            raise RuntimeError("Synthesis failed")
        
        # Convert to NumPy array
        audio_int16 = np.frombuffer(result.audio_data, dtype=np.int16)
        audio_float = audio_int16.astype(np.float32) / 32768.0
        
        return audio_float, result.sample_rate
    
    def batch_synthesize(self, texts: List[str], 
                        config: Optional[Dict[str, Any]] = None) -> List[Tuple[np.ndarray, int]]:
        """Batch synthesis returning NumPy arrays"""
        results = []
        
        for text in texts:
            try:
                audio, sr = self.synthesize_to_array(text, config)
                results.append((audio, sr))
            except Exception as e:
                warnings.warn(f"Failed to synthesize '{text[:50]}...': {e}")
                results.append((np.array([]), 0))
        
        return results
    
    def synthesize_with_features(self, text: str, 
                               config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Synthesize and extract acoustic features"""
        audio, sr = self.synthesize_to_array(text, config)
        
        # Extract features using scipy
        features = self.extract_features(audio, sr)
        
        return {
            'audio': audio,
            'sample_rate': sr,
            'duration': len(audio) / sr,
            'features': features,
            'text': text
        }
    
    def extract_features(self, audio: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Extract comprehensive acoustic features"""
        # Time-domain features
        rms_energy = np.sqrt(np.mean(audio ** 2))
        zero_crossings = np.sum(np.diff(np.sign(audio)) != 0) / len(audio)
        
        # Frequency-domain features
        fft = scipy.fft.fft(audio)
        magnitude_spectrum = np.abs(fft)
        power_spectrum = magnitude_spectrum ** 2
        
        # Spectral features
        freqs = scipy.fft.fftfreq(len(audio), 1/sample_rate)
        positive_freqs = freqs[:len(freqs)//2]
        positive_magnitude = magnitude_spectrum[:len(magnitude_spectrum)//2]
        
        # Spectral centroid
        spectral_centroid = np.sum(positive_freqs * positive_magnitude) / np.sum(positive_magnitude)
        
        # Spectral rolloff (95% of energy)
        cumulative_energy = np.cumsum(positive_magnitude)
        total_energy = cumulative_energy[-1]
        rolloff_idx = np.where(cumulative_energy >= 0.95 * total_energy)[0]
        spectral_rolloff = positive_freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else 0
        
        # Spectral bandwidth
        spectral_spread = np.sqrt(np.sum(((positive_freqs - spectral_centroid) ** 2) * positive_magnitude) 
                                / np.sum(positive_magnitude))
        
        # Fundamental frequency estimation using autocorrelation
        autocorr = np.correlate(audio, audio, mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        
        # Find peaks in autocorrelation
        min_period = int(sample_rate / 800)  # Max F0 = 800 Hz
        max_period = int(sample_rate / 80)   # Min F0 = 80 Hz
        
        if len(autocorr) > max_period:
            autocorr_segment = autocorr[min_period:max_period]
            if len(autocorr_segment) > 0:
                f0_period = np.argmax(autocorr_segment) + min_period
                f0_estimate = sample_rate / f0_period
            else:
                f0_estimate = 0
        else:
            f0_estimate = 0
        
        return {
            'rms_energy': float(rms_energy),
            'zero_crossing_rate': float(zero_crossings),
            'spectral_centroid': float(spectral_centroid),
            'spectral_rolloff': float(spectral_rolloff),
            'spectral_bandwidth': float(spectral_spread),
            'fundamental_frequency': float(f0_estimate),
            'duration': len(audio) / sample_rate,
            'sample_count': len(audio)
        }
    
    def compare_synthesis_quality(self, text: str, 
                                quality_levels: List[str] = None) -> Dict[str, Dict[str, Any]]:
        """Compare synthesis across different quality levels"""
        if quality_levels is None:
            quality_levels = ['low', 'medium', 'high', 'ultra']
        
        results = {}
        
        for quality in quality_levels:
            try:
                config = {'quality': getattr(voirs_ffi.Quality, quality.upper())}
                result = self.synthesize_with_features(text, config)
                results[quality] = result
            except Exception as e:
                warnings.warn(f"Failed to synthesize with quality '{quality}': {e}")
                results[quality] = None
        
        return results
    
    def synthesize_dataset(self, texts: List[str], 
                         metadata: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Create a dataset of synthesized audio with metadata"""
        if metadata and len(metadata) != len(texts):
            raise ValueError("Metadata length must match texts length")
        
        dataset = {
            'audio_data': [],
            'sample_rates': [],
            'features': [],
            'texts': texts,
            'metadata': metadata or [{}] * len(texts)
        }
        
        for i, text in enumerate(texts):
            try:
                result = self.synthesize_with_features(text)
                dataset['audio_data'].append(result['audio'])
                dataset['sample_rates'].append(result['sample_rate'])
                dataset['features'].append(result['features'])
            except Exception as e:
                warnings.warn(f"Failed to process text {i}: {e}")
                dataset['audio_data'].append(np.array([]))
                dataset['sample_rates'].append(0)
                dataset['features'].append({})
        
        # Create feature matrix
        feature_names = list(dataset['features'][0].keys()) if dataset['features'][0] else []
        feature_matrix = np.array([
            [features.get(name, 0) for name in feature_names] 
            for features in dataset['features']
        ])
        
        dataset['feature_matrix'] = feature_matrix
        dataset['feature_names'] = feature_names
        
        return dataset
    
    def analyze_prosody(self, audio: np.ndarray, sample_rate: int,
                       frame_length: int = 2048, hop_length: int = 512) -> Dict[str, np.ndarray]:
        """Analyze prosodic features frame by frame"""
        # Frame-based analysis
        n_frames = (len(audio) - frame_length) // hop_length + 1
        
        rms_frames = np.zeros(n_frames)
        f0_frames = np.zeros(n_frames)
        spectral_centroid_frames = np.zeros(n_frames)
        
        for i in range(n_frames):
            start_idx = i * hop_length
            end_idx = start_idx + frame_length
            frame = audio[start_idx:end_idx]
            
            if len(frame) == frame_length:
                # RMS energy
                rms_frames[i] = np.sqrt(np.mean(frame ** 2))
                
                # Spectral centroid
                fft_frame = scipy.fft.fft(frame)
                magnitude = np.abs(fft_frame[:frame_length//2])
                freqs = scipy.fft.fftfreq(frame_length, 1/sample_rate)[:frame_length//2]
                
                if np.sum(magnitude) > 0:
                    spectral_centroid_frames[i] = np.sum(freqs * magnitude) / np.sum(magnitude)
                
                # F0 estimation using autocorrelation
                autocorr = np.correlate(frame, frame, mode='full')
                autocorr = autocorr[autocorr.size // 2:]
                
                min_period = int(sample_rate / 800)
                max_period = int(sample_rate / 80)
                
                if len(autocorr) > max_period:
                    autocorr_segment = autocorr[min_period:max_period]
                    if len(autocorr_segment) > 0 and np.max(autocorr_segment) > 0:
                        f0_period = np.argmax(autocorr_segment) + min_period
                        f0_frames[i] = sample_rate / f0_period
        
        # Time axis
        time_frames = np.arange(n_frames) * hop_length / sample_rate
        
        return {
            'time': time_frames,
            'rms_energy': rms_frames,
            'fundamental_frequency': f0_frames,
            'spectral_centroid': spectral_centroid_frames
        }

# Example usage and analysis functions
def scientific_analysis_example():
    """Comprehensive example of scientific analysis with VoiRS"""
    # Initialize VoiRS
    voirs = VoiRSNumPyInterface()
    if not voirs.initialize():
        print("Failed to initialize VoiRS")
        return
    
    # Test texts for analysis
    test_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Python is a powerful programming language for data science.",
        "Machine learning algorithms require careful hyperparameter tuning.",
        "Signal processing techniques are fundamental to audio analysis."
    ]
    
    print("1. Basic Synthesis and Feature Extraction")
    print("=" * 50)
    
    for i, text in enumerate(test_texts):
        result = voirs.synthesize_with_features(text)
        features = result['features']
        
        print(f"\nText {i+1}: {text}")
        print(f"Duration: {features['duration']:.2f}s")
        print(f"RMS Energy: {features['rms_energy']:.4f}")
        print(f"F0: {features['fundamental_frequency']:.1f} Hz")
        print(f"Spectral Centroid: {features['spectral_centroid']:.1f} Hz")
    
    print("\n\n2. Quality Comparison Analysis")
    print("=" * 50)
    
    comparison_text = "This is a test for quality comparison analysis."
    quality_comparison = voirs.compare_synthesis_quality(comparison_text)
    
    print(f"Text: {comparison_text}")
    print("\nQuality\tDuration\tRMS\tF0\tCentroid")
    print("-" * 50)
    
    for quality, result in quality_comparison.items():
        if result:
            f = result['features']
            print(f"{quality:7s}\t{f['duration']:.2f}s\t{f['rms_energy']:.3f}\t"
                  f"{f['fundamental_frequency']:.0f}Hz\t{f['spectral_centroid']:.0f}Hz")
    
    print("\n\n3. Dataset Creation and Analysis")
    print("=" * 50)
    
    dataset = voirs.synthesize_dataset(test_texts)
    feature_matrix = dataset['feature_matrix']
    feature_names = dataset['feature_names']
    
    print(f"Dataset shape: {feature_matrix.shape}")
    print(f"Features: {feature_names}")
    
    # Statistical analysis
    print("\nFeature Statistics:")
    print("-" * 30)
    for i, name in enumerate(feature_names):
        values = feature_matrix[:, i]
        print(f"{name:20s}: mean={np.mean(values):.3f}, std={np.std(values):.3f}")
    
    # Correlation analysis
    correlation_matrix = np.corrcoef(feature_matrix.T)
    print(f"\nFeature Correlation Matrix Shape: {correlation_matrix.shape}")
    
    return {
        'voirs_interface': voirs,
        'test_results': quality_comparison,
        'dataset': dataset,
        'correlation_matrix': correlation_matrix
    }

if __name__ == "__main__":
    results = scientific_analysis_example()
```

## Pandas Data Processing

### Text-to-Speech Data Pipeline

#### Pandas Integration (voirs_pandas.py)

```python
import pandas as pd
import numpy as np
import voirs_ffi
from typing import Dict, List, Optional, Any, Callable
import warnings
import time
from pathlib import Path
import json

class VoiRSPandasProcessor:
    """Pandas-integrated VoiRS processing for large-scale text-to-speech workflows"""
    
    def __init__(self):
        self.engine = voirs_ffi.Engine()
        self.initialized = False
        self.processing_stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'total_duration': 0.0,
            'processing_time': 0.0
        }
    
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """Initialize VoiRS engine"""
        default_config = voirs_ffi.SynthesisConfig(
            quality=voirs_ffi.Quality.HIGH,
            thread_count=4
        )
        
        self.initialized = self.engine.initialize(default_config)
        return self.initialized
    
    def process_dataframe(self, df: pd.DataFrame, 
                         text_column: str,
                         config_columns: Optional[Dict[str, str]] = None,
                         output_columns: Optional[Dict[str, str]] = None) -> pd.DataFrame:
        """
        Process a DataFrame with text-to-speech synthesis
        
        Args:
            df: Input DataFrame
            text_column: Column containing text to synthesize
            config_columns: Mapping of synthesis config to column names
            output_columns: Mapping of output names to column names
        
        Returns:
            DataFrame with synthesis results
        """
        if not self.initialized:
            raise RuntimeError("VoiRS engine not initialized")
        
        if text_column not in df.columns:
            raise ValueError(f"Text column '{text_column}' not found in DataFrame")
        
        # Default output columns
        default_outputs = {
            'audio_data': 'audio_data',
            'duration': 'synthesis_duration',
            'success': 'synthesis_success',
            'error': 'synthesis_error',
            'sample_rate': 'sample_rate'
        }
        
        if output_columns:
            default_outputs.update(output_columns)
        
        # Initialize result columns
        result_df = df.copy()
        for col_name in default_outputs.values():
            result_df[col_name] = None
        
        # Process each row
        start_time = time.time()
        
        for idx, row in df.iterrows():
            text = str(row[text_column])
            
            if pd.isna(text) or not text.strip():
                result_df.loc[idx, default_outputs['success']] = False
                result_df.loc[idx, default_outputs['error']] = "Empty text"
                self.processing_stats['failed'] += 1
                continue
            
            # Build synthesis config from row data
            synthesis_config = self._build_config_from_row(row, config_columns)
            
            try:
                # Perform synthesis
                result = self.engine.synthesize(text, synthesis_config)
                
                if result.success:
                    result_df.loc[idx, default_outputs['audio_data']] = result.audio_data
                    result_df.loc[idx, default_outputs['duration']] = result.duration
                    result_df.loc[idx, default_outputs['sample_rate']] = result.sample_rate
                    result_df.loc[idx, default_outputs['success']] = True
                    
                    self.processing_stats['successful'] += 1
                    self.processing_stats['total_duration'] += result.duration
                else:
                    result_df.loc[idx, default_outputs['success']] = False
                    result_df.loc[idx, default_outputs['error']] = "Synthesis failed"
                    self.processing_stats['failed'] += 1
                    
            except Exception as e:
                result_df.loc[idx, default_outputs['success']] = False
                result_df.loc[idx, default_outputs['error']] = str(e)
                self.processing_stats['failed'] += 1
            
            self.processing_stats['total_processed'] += 1
        
        self.processing_stats['processing_time'] = time.time() - start_time
        
        return result_df
    
    def _build_config_from_row(self, row: pd.Series, 
                              config_columns: Optional[Dict[str, str]]) -> voirs_ffi.SynthesisConfig:
        """Build synthesis config from DataFrame row"""
        config = voirs_ffi.SynthesisConfig()
        
        if config_columns:
            for config_param, column_name in config_columns.items():
                if column_name in row and not pd.isna(row[column_name]):
                    value = row[column_name]
                    
                    # Handle specific parameter types
                    if config_param == 'quality':
                        if isinstance(value, str):
                            config.quality = getattr(voirs_ffi.Quality, value.upper())
                    elif config_param == 'output_format':
                        if isinstance(value, str):
                            config.output_format = getattr(voirs_ffi.Format, value.upper())
                    else:
                        setattr(config, config_param, value)
        
        return config
    
    def analyze_synthesis_results(self, df: pd.DataFrame,
                                success_column: str = 'synthesis_success',
                                duration_column: str = 'synthesis_duration',
                                text_column: str = 'text') -> Dict[str, Any]:
        """Analyze synthesis results in DataFrame"""
        if success_column not in df.columns:
            raise ValueError(f"Success column '{success_column}' not found")
        
        successful_df = df[df[success_column] == True]
        failed_df = df[df[success_column] == False]
        
        analysis = {
            'total_items': len(df),
            'successful': len(successful_df),
            'failed': len(failed_df),
            'success_rate': len(successful_df) / len(df) if len(df) > 0 else 0,
        }
        
        if duration_column in df.columns and len(successful_df) > 0:
            durations = successful_df[duration_column].dropna()
            analysis['duration_stats'] = {
                'mean': durations.mean(),
                'median': durations.median(),
                'std': durations.std(),
                'min': durations.min(),
                'max': durations.max(),
                'total': durations.sum()
            }
        
        if text_column in df.columns:
            # Text length analysis
            text_lengths = df[text_column].str.len()
            word_counts = df[text_column].str.split().str.len()
            
            analysis['text_stats'] = {
                'avg_char_length': text_lengths.mean(),
                'avg_word_count': word_counts.mean(),
                'char_length_range': (text_lengths.min(), text_lengths.max()),
                'word_count_range': (word_counts.min(), word_counts.max())
            }
            
            # Success rate by text length
            if len(successful_df) > 0 and len(failed_df) > 0:
                success_lengths = successful_df[text_column].str.len()
                failed_lengths = failed_df[text_column].str.len()
                
                analysis['length_analysis'] = {
                    'avg_successful_length': success_lengths.mean(),
                    'avg_failed_length': failed_lengths.mean(),
                    'length_correlation': text_lengths.corr(df[success_column].astype(int))
                }
        
        return analysis
    
    def create_synthesis_report(self, df: pd.DataFrame,
                              output_file: Optional[str] = None) -> Dict[str, Any]:
        """Create comprehensive synthesis report"""
        analysis = self.analyze_synthesis_results(df)
        
        report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'processing_stats': self.processing_stats.copy(),
            'synthesis_analysis': analysis,
            'configuration': {
                'voirs_version': voirs_ffi.get_version(),
                'dataframe_shape': df.shape,
                'columns': list(df.columns)
            }
        }
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
        
        return report
    
    def batch_process_with_grouping(self, df: pd.DataFrame,
                                  text_column: str,
                                  group_column: str,
                                  config_per_group: Optional[Dict[str, Dict[str, Any]]] = None) -> pd.DataFrame:
        """Process DataFrame with different configurations per group"""
        results = []
        
        for group_value, group_df in df.groupby(group_column):
            print(f"Processing group: {group_value} ({len(group_df)} items)")
            
            # Get config for this group
            group_config_cols = None
            if config_per_group and group_value in config_per_group:
                group_config_cols = config_per_group[group_value]
            
            # Process group
            group_result = self.process_dataframe(
                group_df, 
                text_column, 
                config_columns=group_config_cols
            )
            
            results.append(group_result)
        
        return pd.concat(results, ignore_index=True)

# Example usage functions
def create_sample_dataset() -> pd.DataFrame:
    """Create sample dataset for demonstration"""
    np.random.seed(42)
    
    # Sample texts from different domains
    texts = [
        "Welcome to our customer service line.",
        "The weather today is sunny with a high of 75 degrees.",
        "Please enter your account number followed by the pound key.",
        "Thank you for choosing our premium subscription service.",
        "Your order has been processed and will ship within 24 hours.",
        "Press 1 for English, press 2 for Spanish.",
        "The system will now transfer you to a representative.",
        "We apologize for the delay in processing your request.",
        "Your meeting is scheduled for tomorrow at 2 PM.",
        "Please hold while we connect you to the next available agent."
    ]
    
    # Create DataFrame with synthesis parameters
    data = []
    voices = ['default', 'male_young', 'female_young', 'male_deep', 'female_warm']
    qualities = ['medium', 'high', 'ultra']
    speeds = [0.8, 0.9, 1.0, 1.1, 1.2]
    
    for i, text in enumerate(texts):
        data.append({
            'id': f'sample_{i:03d}',
            'text': text,
            'domain': ['customer_service', 'weather', 'ivr', 'notifications', 'meetings'][i % 5],
            'voice_id': np.random.choice(voices),
            'quality': np.random.choice(qualities),
            'speed': np.random.choice(speeds),
            'priority': np.random.choice(['low', 'medium', 'high']),
            'created_at': pd.Timestamp.now() - pd.Timedelta(days=np.random.randint(0, 30))
        })
    
    return pd.DataFrame(data)

def pandas_processing_example():
    """Comprehensive example of pandas-based processing"""
    print("VoiRS Pandas Integration Example")
    print("=" * 50)
    
    # Initialize processor
    processor = VoiRSPandasProcessor()
    if not processor.initialize():
        print("Failed to initialize VoiRS processor")
        return
    
    # Create sample dataset
    df = create_sample_dataset()
    print(f"Created dataset with {len(df)} samples")
    print(f"Columns: {list(df.columns)}")
    
    # Define configuration mapping
    config_mapping = {
        'voice_id': 'voice_id',
        'quality': 'quality',
        'speed': 'speed'
    }
    
    print("\n1. Processing DataFrame")
    print("-" * 30)
    
    # Process the DataFrame
    result_df = processor.process_dataframe(
        df, 
        text_column='text',
        config_columns=config_mapping
    )
    
    # Display results
    successful = result_df['synthesis_success'].sum()
    total = len(result_df)
    print(f"Processed: {total} items")
    print(f"Successful: {successful}")
    print(f"Failed: {total - successful}")
    print(f"Success rate: {successful/total*100:.1f}%")
    
    print("\n2. Analysis by Domain")
    print("-" * 30)
    
    # Group analysis by domain
    domain_analysis = result_df.groupby('domain').agg({
        'synthesis_success': ['count', 'sum', 'mean'],
        'synthesis_duration': ['mean', 'sum'],
        'text': lambda x: x.str.len().mean()
    }).round(3)
    
    print(domain_analysis)
    
    print("\n3. Quality vs Performance Analysis")
    print("-" * 30)
    
    # Analysis by quality level
    quality_analysis = result_df[result_df['synthesis_success']].groupby('quality').agg({
        'synthesis_duration': ['mean', 'std', 'count'],
        'text': lambda x: x.str.len().mean()
    }).round(3)
    
    print(quality_analysis)
    
    print("\n4. Processing Statistics")
    print("-" * 30)
    
    stats = processor.processing_stats
    print(f"Total processed: {stats['total_processed']}")
    print(f"Processing time: {stats['processing_time']:.2f}s")
    print(f"Average time per item: {stats['processing_time']/stats['total_processed']:.3f}s")
    print(f"Total audio duration: {stats['total_duration']:.2f}s")
    print(f"Real-time factor: {stats['total_duration']/stats['processing_time']:.2f}x")
    
    # Create comprehensive report
    report = processor.create_synthesis_report(result_df, 'synthesis_report.json')
    print(f"\nReport saved to: synthesis_report.json")
    
    return {
        'original_data': df,
        'processed_data': result_df,
        'processor': processor,
        'report': report
    }

if __name__ == "__main__":
    results = pandas_processing_example()
```

## MATLAB Integration

### MATLAB MEX Interface

#### MEX Wrapper (voirs_matlab.c)

```c
#include "mex.h"
#include "matrix.h"
#include <string.h>
#include <stdlib.h>

// VoiRS FFI function declarations
extern void* voirs_pipeline_create(const struct VoiRSSynthesisConfig* config);
extern void* voirs_synthesize(void* pipeline, const char* text, const struct VoiRSSynthesisConfig* config);
extern void voirs_synthesis_result_destroy(void* result);
extern void voirs_pipeline_destroy(void* pipeline);
extern const void* voirs_synthesis_result_get_audio_data(void* result);
extern int32_t voirs_synthesis_result_get_audio_size(void* result);
extern int32_t voirs_synthesis_result_get_sample_rate(void* result);
extern int32_t voirs_synthesis_result_get_channels(void* result);

struct VoiRSSynthesisConfig {
    int32_t quality;
    float speed;
    float volume;
    int32_t output_format;
    const char* voice_id;
};

static void* global_pipeline = NULL;

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    // Check for proper number of inputs
    if (nrhs < 1) {
        mexErrMsgIdAndTxt("VoiRS:invalidNumInputs", "At least one input required.");
    }
    
    // Get command string
    if (!mxIsChar(prhs[0])) {
        mexErrMsgIdAndTxt("VoiRS:invalidInput", "First input must be a command string.");
    }
    
    char* command = mxArrayToString(prhs[0]);
    
    if (strcmp(command, "initialize") == 0) {
        // Initialize VoiRS pipeline
        if (global_pipeline != NULL) {
            voirs_pipeline_destroy(global_pipeline);
        }
        
        struct VoiRSSynthesisConfig config;
        config.quality = 2; // High quality
        config.speed = 1.0f;
        config.volume = 1.0f;
        config.output_format = 0; // WAV
        config.voice_id = "default";
        
        // Override with MATLAB parameters if provided
        if (nrhs > 1 && mxIsStruct(prhs[1])) {
            const mxArray* config_struct = prhs[1];
            
            mxArray* quality_field = mxGetField(config_struct, 0, "quality");
            if (quality_field && mxIsNumeric(quality_field)) {
                config.quality = (int32_t)mxGetScalar(quality_field);
            }
            
            mxArray* speed_field = mxGetField(config_struct, 0, "speed");
            if (speed_field && mxIsNumeric(speed_field)) {
                config.speed = (float)mxGetScalar(speed_field);
            }
            
            mxArray* volume_field = mxGetField(config_struct, 0, "volume");
            if (volume_field && mxIsNumeric(volume_field)) {
                config.volume = (float)mxGetScalar(volume_field);
            }
        }
        
        global_pipeline = voirs_pipeline_create(&config);
        
        if (nlhs > 0) {
            plhs[0] = mxCreateLogicalScalar(global_pipeline != NULL);
        }
        
    } else if (strcmp(command, "synthesize") == 0) {
        // Synthesize text
        if (global_pipeline == NULL) {
            mexErrMsgIdAndTxt("VoiRS:notInitialized", "VoiRS not initialized. Call with 'initialize' first.");
        }
        
        if (nrhs < 2 || !mxIsChar(prhs[1])) {
            mexErrMsgIdAndTxt("VoiRS:invalidInput", "Second input must be text string.");
        }
        
        char* text = mxArrayToString(prhs[1]);
        
        // Synthesis configuration (optional)
        struct VoiRSSynthesisConfig config;
        config.quality = 2;
        config.speed = 1.0f;
        config.volume = 1.0f;
        config.output_format = 0;
        config.voice_id = "default";
        
        if (nrhs > 2 && mxIsStruct(prhs[2])) {
            // Override config with provided parameters
            const mxArray* config_struct = prhs[2];
            
            mxArray* voice_field = mxGetField(config_struct, 0, "voice_id");
            if (voice_field && mxIsChar(voice_field)) {
                config.voice_id = mxArrayToString(voice_field);
            }
            
            mxArray* speed_field = mxGetField(config_struct, 0, "speed");
            if (speed_field && mxIsNumeric(speed_field)) {
                config.speed = (float)mxGetScalar(speed_field);
            }
        }
        
        // Perform synthesis
        void* result = voirs_synthesize(global_pipeline, text, &config);
        
        if (result != NULL) {
            // Get audio data
            const void* audio_data = voirs_synthesis_result_get_audio_data(result);
            int32_t audio_size = voirs_synthesis_result_get_audio_size(result);
            int32_t sample_rate = voirs_synthesis_result_get_sample_rate(result);
            int32_t channels = voirs_synthesis_result_get_channels(result);
            
            // Convert to MATLAB array (assuming 16-bit integers)
            int32_t num_samples = audio_size / (sizeof(int16_t) * channels);
            
            mwSize dims[2] = {num_samples, channels};
            plhs[0] = mxCreateNumericArray(2, dims, mxINT16_CLASS, mxREAL);
            
            // Copy data
            memcpy(mxGetData(plhs[0]), audio_data, audio_size);
            
            // Return sample rate as second output
            if (nlhs > 1) {
                plhs[1] = mxCreateDoubleScalar((double)sample_rate);
            }
            
            // Return duration as third output
            if (nlhs > 2) {
                double duration = (double)num_samples / (double)sample_rate;
                plhs[2] = mxCreateDoubleScalar(duration);
            }
            
            voirs_synthesis_result_destroy(result);
        } else {
            mexErrMsgIdAndTxt("VoiRS:synthesisFailed", "Text synthesis failed.");
        }
        
        mxFree(text);
        
    } else if (strcmp(command, "cleanup") == 0) {
        // Cleanup resources
        if (global_pipeline != NULL) {
            voirs_pipeline_destroy(global_pipeline);
            global_pipeline = NULL;
        }
        
        if (nlhs > 0) {
            plhs[0] = mxCreateLogicalScalar(1);
        }
        
    } else {
        mexErrMsgIdAndTxt("VoiRS:invalidCommand", "Unknown command. Use 'initialize', 'synthesize', or 'cleanup'.");
    }
    
    mxFree(command);
}

// MEX exit function - cleanup on MEX clear
void mexAtExit(void) {
    if (global_pipeline != NULL) {
        voirs_pipeline_destroy(global_pipeline);
        global_pipeline = NULL;
    }
}
```

#### MATLAB Interface Class (VoiRS.m)

```matlab
classdef VoiRS < handle
    % VoiRS - MATLAB interface for VoiRS text-to-speech synthesis
    %
    % This class provides a MATLAB interface to the VoiRS FFI library
    % for high-quality text-to-speech synthesis.
    %
    % Example:
    %   voirs = VoiRS();
    %   voirs.initialize();
    %   [audio, fs] = voirs.synthesize('Hello, MATLAB world!');
    %   sound(audio, fs);
    
    properties (Access = private)
        initialized = false;
    end
    
    properties (Constant)
        % Quality levels
        QUALITY_LOW = 0;
        QUALITY_MEDIUM = 1;
        QUALITY_HIGH = 2;
        QUALITY_ULTRA = 3;
        
        % Default configuration
        DEFAULT_CONFIG = struct(...
            'quality', 2, ...
            'speed', 1.0, ...
            'volume', 1.0 ...
        );
    end
    
    methods
        function obj = VoiRS()
            % Constructor
            % Load MEX function if not already loaded
            if ~exist('voirs_matlab', 'file')
                error('VoiRS:mexNotFound', 'voirs_matlab MEX function not found. Please compile the MEX interface.');
            end
        end
        
        function success = initialize(obj, config)
            % Initialize VoiRS engine
            %
            % Inputs:
            %   config (optional) - Structure with configuration parameters:
            %     .quality - Quality level (0-3, default: 2)
            %     .speed   - Speech speed (0.1-3.0, default: 1.0)
            %     .volume  - Volume level (0.0-2.0, default: 1.0)
            %
            % Outputs:
            %   success - Logical indicating initialization success
            
            if nargin < 2
                config = obj.DEFAULT_CONFIG;
            else
                config = obj.mergeConfig(obj.DEFAULT_CONFIG, config);
            end
            
            try
                success = voirs_matlab('initialize', config);
                obj.initialized = success;
                
                if success
                    fprintf('VoiRS initialized successfully\n');
                else
                    warning('VoiRS:initFailed', 'VoiRS initialization failed');
                end
            catch ME
                warning('VoiRS:initError', 'VoiRS initialization error: %s', ME.message);
                success = false;
                obj.initialized = false;
            end
        end
        
        function [audio_data, sample_rate, duration] = synthesize(obj, text, config)
            % Synthesize text to speech
            %
            % Inputs:
            %   text   - String to synthesize
            %   config (optional) - Synthesis configuration structure:
            %     .voice_id - Voice identifier (default: 'default')
            %     .speed    - Speech speed (0.1-3.0, default: 1.0)
            %     .quality  - Quality level (0-3, default: 2)
            %
            % Outputs:
            %   audio_data  - Audio signal as int16 array
            %   sample_rate - Sample rate in Hz
            %   duration    - Duration in seconds
            
            if ~obj.initialized
                error('VoiRS:notInitialized', 'VoiRS not initialized. Call initialize() first.');
            end
            
            if ~ischar(text) && ~isstring(text)
                error('VoiRS:invalidInput', 'Text input must be a string or char array.');
            end
            
            % Convert string to char if necessary
            if isstring(text)
                text = char(text);
            end
            
            % Use default config if not provided
            if nargin < 3
                config = struct();
            end
            
            try
                [audio_data, sample_rate, duration] = voirs_matlab('synthesize', text, config);
            catch ME
                error('VoiRS:synthesisError', 'Synthesis failed: %s', ME.message);
            end
        end
        
        function audio_normalized = synthesizeNormalized(obj, text, config)
            % Synthesize text and return normalized floating-point audio
            %
            % Inputs:
            %   text   - String to synthesize
            %   config - Optional synthesis configuration
            %
            % Outputs:
            %   audio_normalized - Normalized audio signal (range: -1 to 1)
            
            [audio_int16, ~, ~] = obj.synthesize(text, config);
            audio_normalized = double(audio_int16) / 32768.0;
        end
        
        function results = synthesizeBatch(obj, texts, config)
            % Synthesize multiple texts in batch
            %
            % Inputs:
            %   texts  - Cell array of strings
            %   config - Optional synthesis configuration
            %
            % Outputs:
            %   results - Structure array with fields:
            %     .text        - Original text
            %     .audio_data  - Synthesized audio
            %     .sample_rate - Sample rate
            %     .duration    - Duration in seconds
            %     .success     - Logical success flag
            %     .error       - Error message (if failed)
            
            if ~iscell(texts)
                error('VoiRS:invalidInput', 'Texts input must be a cell array of strings.');
            end
            
            n_texts = length(texts);
            results(n_texts) = struct();
            
            if nargin < 3
                config = struct();
            end
            
            fprintf('Processing %d texts...\n', n_texts);
            
            for i = 1:n_texts
                results(i).text = texts{i};
                results(i).success = false;
                
                try
                    [audio, fs, duration] = obj.synthesize(texts{i}, config);
                    results(i).audio_data = audio;
                    results(i).sample_rate = fs;
                    results(i).duration = duration;
                    results(i).success = true;
                    results(i).error = '';
                    
                    fprintf('  ✓ %d/%d: %s\n', i, n_texts, texts{i}(1:min(50, length(texts{i}))));
                    
                catch ME
                    results(i).error = ME.message;
                    fprintf('  ✗ %d/%d: %s (Error: %s)\n', i, n_texts, ...
                           texts{i}(1:min(50, length(texts{i}))), ME.message);
                end
            end
            
            successful = sum([results.success]);
            fprintf('Batch processing complete: %d/%d successful\n', successful, n_texts);
        end
        
        function analysis = analyzeAudio(obj, audio_data, sample_rate)
            % Analyze synthesized audio
            %
            % Inputs:
            %   audio_data  - Audio signal
            %   sample_rate - Sample rate in Hz
            %
            % Outputs:
            %   analysis - Structure with analysis results
            
            % Convert to double precision
            if isa(audio_data, 'int16')
                audio_double = double(audio_data) / 32768.0;
            else
                audio_double = double(audio_data);
            end
            
            % Basic statistics
            analysis.duration = length(audio_double) / sample_rate;
            analysis.sample_rate = sample_rate;
            analysis.num_samples = length(audio_double);
            analysis.rms_energy = sqrt(mean(audio_double.^2));
            analysis.peak_amplitude = max(abs(audio_double));
            
            % Zero crossing rate
            zero_crossings = sum(diff(sign(audio_double)) ~= 0);
            analysis.zero_crossing_rate = zero_crossings / length(audio_double);
            
            % Spectral analysis
            N = length(audio_double);
            frequencies = (0:N-1) * sample_rate / N;
            fft_data = fft(audio_double);
            magnitude_spectrum = abs(fft_data(1:floor(N/2)));
            power_spectrum = magnitude_spectrum.^2;
            
            % Spectral centroid
            positive_freqs = frequencies(1:floor(N/2));
            analysis.spectral_centroid = sum(positive_freqs .* magnitude_spectrum') / sum(magnitude_spectrum);
            
            % Spectral rolloff (95% of energy)
            cumulative_energy = cumsum(power_spectrum);
            total_energy = cumulative_energy(end);
            rolloff_idx = find(cumulative_energy >= 0.95 * total_energy, 1);
            analysis.spectral_rolloff = positive_freqs(rolloff_idx);
            
            % Fundamental frequency estimation (simple autocorrelation)
            autocorr = xcorr(audio_double, audio_double);
            autocorr = autocorr((length(autocorr)+1)/2:end); % Take positive lags only
            
            % Look for peaks in reasonable F0 range (80-800 Hz)
            min_period = round(sample_rate / 800);
            max_period = round(sample_rate / 80);
            
            if length(autocorr) > max_period
                [~, peak_idx] = max(autocorr(min_period:max_period));
                f0_period = peak_idx + min_period - 1;
                analysis.fundamental_frequency = sample_rate / f0_period;
            else
                analysis.fundamental_frequency = NaN;
            end
        end
        
        function visualizeAudio(obj, audio_data, sample_rate, analysis_flag)
            % Visualize synthesized audio
            %
            % Inputs:
            %   audio_data    - Audio signal
            %   sample_rate   - Sample rate in Hz
            %   analysis_flag - Include spectral analysis (default: true)
            
            if nargin < 4
                analysis_flag = true;
            end
            
            % Convert to double precision
            if isa(audio_data, 'int16')
                audio_double = double(audio_data) / 32768.0;
            else
                audio_double = double(audio_data);
            end
            
            time_axis = (0:length(audio_double)-1) / sample_rate;
            
            figure('Name', 'VoiRS Audio Analysis', 'Position', [100 100 1200 800]);
            
            if analysis_flag
                % Multi-panel plot
                subplot(2, 2, 1);
                plot(time_axis, audio_double);
                title('Waveform');
                xlabel('Time (s)');
                ylabel('Amplitude');
                grid on;
                
                % Spectrogram
                subplot(2, 2, 2);
                spectrogram(audio_double, hamming(256), 128, 512, sample_rate, 'yaxis');
                title('Spectrogram');
                
                % Frequency spectrum
                subplot(2, 2, 3);
                N = length(audio_double);
                frequencies = (0:N-1) * sample_rate / N;
                fft_data = fft(audio_double);
                magnitude_spectrum = abs(fft_data);
                
                plot(frequencies(1:floor(N/2)), 20*log10(magnitude_spectrum(1:floor(N/2))));
                title('Frequency Spectrum');
                xlabel('Frequency (Hz)');
                ylabel('Magnitude (dB)');
                grid on;
                
                % Audio analysis
                subplot(2, 2, 4);
                analysis = obj.analyzeAudio(audio_data, sample_rate);
                
                analysis_text = sprintf(...
                    'Duration: %.2f s\nRMS Energy: %.4f\nSpectral Centroid: %.1f Hz\nF0 Estimate: %.1f Hz', ...
                    analysis.duration, analysis.rms_energy, ...
                    analysis.spectral_centroid, analysis.fundamental_frequency);
                
                text(0.1, 0.7, analysis_text, 'FontSize', 12, 'FontFamily', 'monospace');
                axis off;
                title('Audio Statistics');
                
            else
                % Simple waveform plot
                plot(time_axis, audio_double);
                title('Synthesized Audio Waveform');
                xlabel('Time (s)');
                ylabel('Amplitude');
                grid on;
            end
        end
        
        function success = cleanup(obj)
            % Cleanup VoiRS resources
            %
            % Outputs:
            %   success - Logical indicating cleanup success
            
            try
                success = voirs_matlab('cleanup');
                obj.initialized = false;
                fprintf('VoiRS cleanup completed\n');
            catch ME
                warning('VoiRS:cleanupError', 'Cleanup error: %s', ME.message);
                success = false;
            end
        end
        
        function delete(obj)
            % Destructor - ensure cleanup
            if obj.initialized
                obj.cleanup();
            end
        end
    end
    
    methods (Access = private)
        function merged = mergeConfig(~, default_config, user_config)
            % Merge user configuration with defaults
            merged = default_config;
            
            if isstruct(user_config)
                fields = fieldnames(user_config);
                for i = 1:length(fields)
                    merged.(fields{i}) = user_config.(fields{i});
                end
            end
        end
    end
    
    methods (Static)
        function demo()
            % Run VoiRS demonstration
            fprintf('VoiRS MATLAB Demo\n');
            fprintf('================\n\n');
            
            % Initialize VoiRS
            voirs = VoiRS();
            if ~voirs.initialize()
                error('Failed to initialize VoiRS');
            end
            
            % Demo texts
            demo_texts = {
                'Hello, MATLAB world!',
                'This is a demonstration of VoiRS text-to-speech synthesis.',
                'MATLAB provides excellent tools for scientific computing.',
                'The integration allows for seamless audio processing workflows.'
            };
            
            fprintf('1. Single synthesis demo:\n');
            [audio, fs] = voirs.synthesize(demo_texts{1});
            fprintf('   Synthesized: "%s"\n', demo_texts{1});
            fprintf('   Duration: %.2f seconds\n', length(audio)/fs);
            
            % Play audio (if possible)
            try
                sound(audio, fs);
                fprintf('   Playing audio...\n');
                pause(length(audio)/fs + 0.5);
            catch
                fprintf('   (Audio playback not available)\n');
            end
            
            fprintf('\n2. Batch synthesis demo:\n');
            results = voirs.synthesizeBatch(demo_texts);
            
            fprintf('\n3. Audio analysis demo:\n');
            analysis = voirs.analyzeAudio(audio, fs);
            fprintf('   RMS Energy: %.4f\n', analysis.rms_energy);
            fprintf('   Spectral Centroid: %.1f Hz\n', analysis.spectral_centroid);
            fprintf('   F0 Estimate: %.1f Hz\n', analysis.fundamental_frequency);
            
            fprintf('\n4. Visualization demo:\n');
            voirs.visualizeAudio(audio, fs);
            
            % Cleanup
            voirs.cleanup();
            fprintf('\nDemo completed!\n');
        end
    end
end
```

## R Integration

### R Package Interface

#### R Package Description (DESCRIPTION)

```
Package: voirsR
Type: Package
Title: R Interface for VoiRS Text-to-Speech Synthesis
Version: 0.1.0
Author: VoiRS Team
Maintainer: VoiRS Team <support@voirs.ai>
Description: Provides R bindings for the VoiRS FFI library, enabling high-quality
    text-to-speech synthesis directly from R. Supports batch processing, audio analysis,
    and integration with R data analysis workflows.
License: MIT
Encoding: UTF-8
LazyData: true
Depends: R (>= 3.5.0)
Imports: 
    methods,
    audio,
    signal,
    tuneR,
    dplyr,
    ggplot2,
    jsonlite
Suggests: 
    knitr,
    rmarkdown,
    testthat
VignetteBuilder: knitr
SystemRequirements: VoiRS FFI library
RoxygenNote: 7.2.0
```

#### R Interface Implementation (R/voirs.R)

```r
#' VoiRS R Interface
#' 
#' R bindings for VoiRS text-to-speech synthesis
#' 
#' @docType package
#' @name voirsR
#' @useDynLib voirsR
NULL

# Quality levels enumeration
VOIRS_QUALITY <- list(
  LOW = 0L,
  MEDIUM = 1L,
  HIGH = 2L,
  ULTRA = 3L
)

# Format enumeration
VOIRS_FORMAT <- list(
  WAV = 0L,
  MP3 = 1L,
  FLAC = 2L
)

#' Initialize VoiRS Engine
#' 
#' Initialize the VoiRS text-to-speech engine with specified configuration.
#' 
#' @param quality Integer quality level (0-3). Default: 2 (HIGH)
#' @param speed Numeric speech speed (0.1-3.0). Default: 1.0
#' @param volume Numeric volume level (0.0-2.0). Default: 1.0
#' @param thread_count Integer number of threads. Default: 2
#' 
#' @return Logical indicating initialization success
#' @export
#' 
#' @examples
#' \dontrun{
#' # Initialize with default settings
#' voirs_init()
#' 
#' # Initialize with custom quality
#' voirs_init(quality = VOIRS_QUALITY$ULTRA)
#' }
voirs_init <- function(quality = VOIRS_QUALITY$HIGH, 
                       speed = 1.0, 
                       volume = 1.0, 
                       thread_count = 2L) {
  
  if (!is.numeric(quality) || quality < 0 || quality > 3) {
    stop("Quality must be an integer between 0 and 3")
  }
  
  if (!is.numeric(speed) || speed < 0.1 || speed > 3.0) {
    stop("Speed must be between 0.1 and 3.0")
  }
  
  if (!is.numeric(volume) || volume < 0.0 || volume > 2.0) {
    stop("Volume must be between 0.0 and 2.0")
  }
  
  result <- .Call("voirs_r_init", 
                  as.integer(quality),
                  as.numeric(speed),
                  as.numeric(volume),
                  as.integer(thread_count))
  
  return(as.logical(result))
}

#' Synthesize Text to Speech
#' 
#' Convert text to synthesized speech audio.
#' 
#' @param text Character string to synthesize
#' @param voice_id Character voice identifier. Default: "default"
#' @param quality Integer quality level (0-3). Default: current setting
#' @param speed Numeric speech speed (0.1-3.0). Default: current setting
#' @param volume Numeric volume level (0.0-2.0). Default: current setting
#' @param format Integer output format (0=WAV, 1=MP3, 2=FLAC). Default: 0
#' 
#' @return List with components:
#'   \item{success}{Logical indicating synthesis success}
#'   \item{audio_data}{Raw vector containing audio data}
#'   \item{sample_rate}{Integer sample rate in Hz}
#'   \item{duration}{Numeric duration in seconds}
#'   \item{error}{Character error message (if failed)}
#' 
#' @export
#' 
#' @examples
#' \dontrun{
#' # Initialize VoiRS
#' voirs_init()
#' 
#' # Synthesize text
#' result <- voirs_synthesize("Hello, R world!")
#' 
#' if (result$success) {
#'   cat("Synthesis successful, duration:", result$duration, "seconds\n")
#' }
#' }
voirs_synthesize <- function(text, 
                            voice_id = "default",
                            quality = NULL,
                            speed = NULL,
                            volume = NULL,
                            format = VOIRS_FORMAT$WAV) {
  
  if (!is.character(text) || length(text) != 1) {
    stop("Text must be a single character string")
  }
  
  if (nchar(text) == 0) {
    stop("Text cannot be empty")
  }
  
  if (nchar(text) > 10000) {
    stop("Text too long (maximum 10,000 characters)")
  }
  
  # Use current settings if not specified
  if (is.null(quality)) quality <- -1L
  if (is.null(speed)) speed <- -1.0
  if (is.null(volume)) volume <- -1.0
  
  result <- .Call("voirs_r_synthesize",
                  text,
                  voice_id,
                  as.integer(quality),
                  as.numeric(speed),
                  as.numeric(volume),
                  as.integer(format))
  
  return(result)
}

#' Batch Synthesis
#' 
#' Synthesize multiple texts in batch with progress tracking.
#' 
#' @param texts Character vector of texts to synthesize
#' @param voice_id Character voice identifier. Default: "default"
#' @param quality Integer quality level. Default: current setting
#' @param speed Numeric speech speed. Default: current setting
#' @param volume Numeric volume level. Default: current setting
#' @param format Integer output format. Default: 0 (WAV)
#' @param progress Logical whether to show progress. Default: TRUE
#' 
#' @return Data frame with columns:
#'   \item{index}{Integer index of text}
#'   \item{text}{Character original text}
#'   \item{success}{Logical synthesis success}
#'   \item{duration}{Numeric duration in seconds}
#'   \item{sample_rate}{Integer sample rate}
#'   \item{audio_data}{List column with raw audio data}
#'   \item{error}{Character error message}
#' 
#' @export
#' 
#' @examples
#' \dontrun{
#' texts <- c("First sentence.", "Second sentence.", "Third sentence.")
#' results <- voirs_batch_synthesize(texts)
#' 
#' # Summary
#' cat("Successful:", sum(results$success), "out of", nrow(results), "\n")
#' }
voirs_batch_synthesize <- function(texts,
                                  voice_id = "default",
                                  quality = NULL,
                                  speed = NULL,
                                  volume = NULL,
                                  format = VOIRS_FORMAT$WAV,
                                  progress = TRUE) {
  
  if (!is.character(texts)) {
    stop("Texts must be a character vector")
  }
  
  if (length(texts) == 0) {
    stop("Texts vector cannot be empty")
  }
  
  if (length(texts) > 1000) {
    stop("Too many texts (maximum 1,000)")
  }
  
  n_texts <- length(texts)
  results <- data.frame(
    index = 1:n_texts,
    text = texts,
    success = logical(n_texts),
    duration = numeric(n_texts),
    sample_rate = integer(n_texts),
    audio_data = vector("list", n_texts),
    error = character(n_texts),
    stringsAsFactors = FALSE
  )
  
  if (progress) {
    cat("Processing", n_texts, "texts...\n")
  }
  
  for (i in 1:n_texts) {
    if (progress && i %% 10 == 0) {
      cat("  Progress:", i, "/", n_texts, "\n")
    }
    
    result <- voirs_synthesize(texts[i], voice_id, quality, speed, volume, format)
    
    results$success[i] <- result$success
    results$duration[i] <- if (result$success) result$duration else 0
    results$sample_rate[i] <- if (result$success) result$sample_rate else 0
    results$audio_data[[i]] <- if (result$success) result$audio_data else raw(0)
    results$error[i] <- if (!result$success) result$error else ""
  }
  
  if (progress) {
    successful <- sum(results$success)
    cat("Batch complete:", successful, "/", n_texts, "successful\n")
    cat("Total duration:", sum(results$duration), "seconds\n")
  }
  
  return(results)
}

#' Analyze Audio Features
#' 
#' Extract acoustic features from synthesized audio.
#' 
#' @param audio_data Raw vector containing audio data
#' @param sample_rate Integer sample rate in Hz
#' 
#' @return List with acoustic features:
#'   \item{duration}{Numeric duration in seconds}
#'   \item{rms_energy}{Numeric RMS energy}
#'   \item{zero_crossing_rate}{Numeric zero crossing rate}
#'   \item{spectral_centroid}{Numeric spectral centroid in Hz}
#'   \item{spectral_rolloff}{Numeric spectral rolloff frequency}
#'   \item{fundamental_frequency}{Numeric F0 estimate in Hz}
#' 
#' @export
#' 
#' @examples
#' \dontrun{
#' result <- voirs_synthesize("Hello world")
#' if (result$success) {
#'   features <- voirs_analyze_audio(result$audio_data, result$sample_rate)
#'   print(features)
#' }
#' }
voirs_analyze_audio <- function(audio_data, sample_rate) {
  if (!is.raw(audio_data)) {
    stop("audio_data must be a raw vector")
  }
  
  if (!is.numeric(sample_rate) || sample_rate <= 0) {
    stop("sample_rate must be a positive number")
  }
  
  features <- .Call("voirs_r_analyze_audio", audio_data, as.integer(sample_rate))
  return(features)
}

#' Convert Audio to Wave Object
#' 
#' Convert VoiRS audio data to tuneR Wave object for further processing.
#' 
#' @param audio_data Raw vector containing audio data
#' @param sample_rate Integer sample rate in Hz
#' @param bit_depth Integer bit depth (16 or 32). Default: 16
#' 
#' @return tuneR Wave object
#' 
#' @export
#' 
#' @examples
#' \dontrun{
#' result <- voirs_synthesize("Convert to Wave object")
#' if (result$success) {
#'   wave_obj <- voirs_to_wave(result$audio_data, result$sample_rate)
#'   tuneR::play(wave_obj)
#' }
#' }
voirs_to_wave <- function(audio_data, sample_rate, bit_depth = 16) {
  if (!requireNamespace("tuneR", quietly = TRUE)) {
    stop("tuneR package required for this function")
  }
  
  if (!is.raw(audio_data)) {
    stop("audio_data must be a raw vector")
  }
  
  # Convert raw data to numeric
  if (bit_depth == 16) {
    # Convert raw bytes to 16-bit integers
    audio_int <- as.integer(readBin(audio_data, "integer", 
                                   size = 2, signed = TRUE,
                                   n = length(audio_data) / 2,
                                   endian = "little"))
  } else {
    stop("Only 16-bit audio currently supported")
  }
  
  # Create Wave object
  wave_obj <- tuneR::Wave(left = audio_int, 
                         samp.rate = sample_rate,
                         bit = bit_depth)
  
  return(wave_obj)
}

#' Visualize Audio Analysis
#' 
#' Create visualization plots for synthesized audio analysis.
#' 
#' @param audio_data Raw vector containing audio data
#' @param sample_rate Integer sample rate in Hz
#' @param title Character plot title. Default: "VoiRS Audio Analysis"
#' 
#' @return ggplot2 plot object
#' 
#' @export
#' 
#' @examples
#' \dontrun{
#' result <- voirs_synthesize("Visualize this audio")
#' if (result$success) {
#'   plot <- voirs_plot_audio(result$audio_data, result$sample_rate)
#'   print(plot)
#' }
#' }
voirs_plot_audio <- function(audio_data, sample_rate, title = "VoiRS Audio Analysis") {
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("ggplot2 package required for this function")
  }
  
  if (!requireNamespace("signal", quietly = TRUE)) {
    stop("signal package required for this function")
  }
  
  # Convert to numeric waveform
  audio_int <- as.integer(readBin(audio_data, "integer", 
                                 size = 2, signed = TRUE,
                                 n = length(audio_data) / 2,
                                 endian = "little"))
  
  audio_norm <- audio_int / 32768.0
  n_samples <- length(audio_norm)
  time_axis <- (0:(n_samples-1)) / sample_rate
  
  # Create data frame for plotting
  plot_data <- data.frame(
    time = time_axis,
    amplitude = audio_norm
  )
  
  # Create waveform plot
  p <- ggplot2::ggplot(plot_data, ggplot2::aes(x = time, y = amplitude)) +
    ggplot2::geom_line(color = "blue", size = 0.5) +
    ggplot2::labs(
      title = title,
      x = "Time (seconds)",
      y = "Amplitude"
    ) +
    ggplot2::theme_minimal() +
    ggplot2::theme(
      plot.title = ggplot2::element_text(hjust = 0.5, size = 14),
      axis.title = ggplot2::element_text(size = 12),
      axis.text = ggplot2::element_text(size = 10)
    )
  
  return(p)
}

#' Create Synthesis Report
#' 
#' Generate comprehensive report from batch synthesis results.
#' 
#' @param results Data frame from voirs_batch_synthesize
#' @param output_file Character output file path (optional)
#' 
#' @return List with report data
#' 
#' @export
#' 
#' @examples
#' \dontrun{
#' texts <- c("First.", "Second.", "Third.")
#' results <- voirs_batch_synthesize(texts)
#' report <- voirs_create_report(results, "synthesis_report.json")
#' }
voirs_create_report <- function(results, output_file = NULL) {
  if (!requireNamespace("dplyr", quietly = TRUE)) {
    stop("dplyr package required for this function")
  }
  
  if (!is.data.frame(results)) {
    stop("results must be a data frame")
  }
  
  # Basic statistics
  total_items <- nrow(results)
  successful_items <- sum(results$success)
  failed_items <- total_items - successful_items
  success_rate <- successful_items / total_items
  
  # Duration statistics (for successful syntheses)
  successful_results <- results[results$success, ]
  
  duration_stats <- list(
    total_duration = sum(successful_results$duration),
    mean_duration = mean(successful_results$duration),
    median_duration = median(successful_results$duration),
    min_duration = min(successful_results$duration),
    max_duration = max(successful_results$duration)
  )
  
  # Text length analysis
  text_lengths <- nchar(results$text)
  word_counts <- sapply(strsplit(results$text, "\\s+"), length)
  
  text_stats <- list(
    mean_char_length = mean(text_lengths),
    mean_word_count = mean(word_counts),
    char_length_range = range(text_lengths),
    word_count_range = range(word_counts)
  )
  
  # Error analysis
  error_summary <- table(results$error[!results$success])
  
  # Create comprehensive report
  report <- list(
    timestamp = Sys.time(),
    summary = list(
      total_items = total_items,
      successful_items = successful_items,
      failed_items = failed_items,
      success_rate = success_rate
    ),
    duration_statistics = duration_stats,
    text_statistics = text_stats,
    error_summary = as.list(error_summary),
    detailed_results = results
  )
  
  # Save to file if requested
  if (!is.null(output_file)) {
    if (!requireNamespace("jsonlite", quietly = TRUE)) {
      stop("jsonlite package required for JSON output")
    }
    
    # Convert to JSON-friendly format
    json_report <- report
    json_report$detailed_results$audio_data <- NULL  # Remove raw data
    
    jsonlite::write_json(json_report, output_file, pretty = TRUE, auto_unbox = TRUE)
    cat("Report saved to:", output_file, "\n")
  }
  
  return(report)
}

#' Cleanup VoiRS Resources
#' 
#' Clean up VoiRS engine resources.
#' 
#' @return Logical indicating cleanup success
#' @export
#' 
#' @examples
#' \dontrun{
#' voirs_cleanup()
#' }
voirs_cleanup <- function() {
  result <- .Call("voirs_r_cleanup")
  return(as.logical(result))
}

# Package unload hook
.onUnload <- function(libpath) {
  try(voirs_cleanup(), silent = TRUE)
}
```

These scientific computing integration examples demonstrate how VoiRS FFI can be seamlessly integrated into research workflows, providing powerful text-to-speech capabilities for data analysis, experimentation, and computational research across multiple platforms and programming environments.
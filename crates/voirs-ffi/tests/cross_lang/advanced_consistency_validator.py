#!/usr/bin/env python3
"""
Advanced Consistency Validator for VoiRS Cross-Language Bindings.

This module provides sophisticated validation of consistency across
language bindings, including audio quality analysis, feature parity
testing, and behavioral consistency validation.
"""

import sys
import os
import time
import math
import hashlib
import tempfile
import subprocess
import json
import statistics
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
import threading
from concurrent.futures import ThreadPoolExecutor

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

@dataclass
class AudioFingerprint:
    """Audio fingerprint for cross-binding comparison."""
    duration: float
    sample_rate: int
    channels: int
    rms_energy: float
    spectral_centroid: float
    zero_crossing_rate: float
    peak_amplitude: float
    silence_ratio: float
    dynamic_range: float
    frequency_content_hash: str

@dataclass
class ConsistencyTestResult:
    """Result of a consistency test."""
    test_name: str
    binding_results: Dict[str, Any]
    consistency_score: float
    discrepancies: List[str]
    audio_fingerprints: Dict[str, AudioFingerprint]
    timing_analysis: Dict[str, Any]
    feature_parity: Dict[str, bool]
    error_analysis: Dict[str, Any]

class AudioAnalyzer:
    """Advanced audio analysis for consistency validation."""
    
    @staticmethod
    def analyze_audio_samples(samples: List[float], sample_rate: int) -> AudioFingerprint:
        """Analyze audio samples and generate fingerprint."""
        if not samples:
            return AudioAnalyzer._empty_fingerprint()
        
        # Basic properties
        duration = len(samples) / sample_rate
        channels = 1  # Assuming mono for simplicity
        
        # Convert to numpy array for analysis
        try:
            import numpy as np
            audio = np.array(samples, dtype=np.float32)
        except ImportError:
            # Fallback to basic analysis without numpy
            return AudioAnalyzer._basic_analysis(samples, sample_rate, duration, channels)
        
        # Energy analysis
        rms_energy = np.sqrt(np.mean(audio ** 2))
        peak_amplitude = np.max(np.abs(audio))
        
        # Silence detection
        silence_threshold = 0.01 * peak_amplitude
        silence_samples = np.sum(np.abs(audio) < silence_threshold)
        silence_ratio = silence_samples / len(audio)
        
        # Dynamic range
        dynamic_range = 20 * np.log10(peak_amplitude / (rms_energy + 1e-10))
        
        # Zero crossing rate
        zero_crossings = np.sum(np.diff(np.sign(audio)) != 0)
        zero_crossing_rate = zero_crossings / len(audio)
        
        # Spectral analysis (simplified)
        try:
            # Simple spectral centroid approximation
            fft = np.fft.fft(audio[:min(len(audio), 2048)])
            magnitude = np.abs(fft[:len(fft)//2])
            freqs = np.fft.fftfreq(len(fft), 1/sample_rate)[:len(fft)//2]
            spectral_centroid = np.sum(freqs * magnitude) / (np.sum(magnitude) + 1e-10)
            
            # Frequency content hash (for consistency checking)
            magnitude_normalized = magnitude / (np.max(magnitude) + 1e-10)
            frequency_content_hash = hashlib.md5(magnitude_normalized.tobytes()).hexdigest()[:16]
            
        except Exception:
            spectral_centroid = 0.0
            frequency_content_hash = "unknown"
        
        return AudioFingerprint(
            duration=duration,
            sample_rate=sample_rate,
            channels=channels,
            rms_energy=float(rms_energy),
            spectral_centroid=float(spectral_centroid),
            zero_crossing_rate=float(zero_crossing_rate),
            peak_amplitude=float(peak_amplitude),
            silence_ratio=float(silence_ratio),
            dynamic_range=float(dynamic_range),
            frequency_content_hash=frequency_content_hash
        )
    
    @staticmethod
    def _empty_fingerprint() -> AudioFingerprint:
        """Return empty fingerprint for error cases."""
        return AudioFingerprint(
            duration=0.0, sample_rate=0, channels=0,
            rms_energy=0.0, spectral_centroid=0.0, zero_crossing_rate=0.0,
            peak_amplitude=0.0, silence_ratio=1.0, dynamic_range=0.0,
            frequency_content_hash="empty"
        )
    
    @staticmethod
    def _basic_analysis(samples: List[float], sample_rate: int, 
                       duration: float, channels: int) -> AudioFingerprint:
        """Basic audio analysis without numpy."""
        if not samples:
            return AudioAnalyzer._empty_fingerprint()
        
        # Basic statistics
        rms_energy = math.sqrt(sum(s ** 2 for s in samples) / len(samples))
        peak_amplitude = max(abs(s) for s in samples)
        
        # Zero crossing rate
        zero_crossings = sum(1 for i in range(1, len(samples)) 
                           if (samples[i] >= 0) != (samples[i-1] >= 0))
        zero_crossing_rate = zero_crossings / len(samples)
        
        # Silence detection
        silence_threshold = 0.01 * peak_amplitude
        silence_samples = sum(1 for s in samples if abs(s) < silence_threshold)
        silence_ratio = silence_samples / len(samples)
        
        # Dynamic range
        dynamic_range = 20 * math.log10(peak_amplitude / (rms_energy + 1e-10))
        
        # Simple hash for consistency
        sample_hash = hashlib.md5(str(samples[:100]).encode()).hexdigest()[:16]
        
        return AudioFingerprint(
            duration=duration,
            sample_rate=sample_rate,
            channels=channels,
            rms_energy=rms_energy,
            spectral_centroid=0.0,  # Not calculated in basic mode
            zero_crossing_rate=zero_crossing_rate,
            peak_amplitude=peak_amplitude,
            silence_ratio=silence_ratio,
            dynamic_range=dynamic_range,
            frequency_content_hash=sample_hash
        )
    
    @staticmethod
    def compare_fingerprints(fp1: AudioFingerprint, fp2: AudioFingerprint) -> float:
        """Compare two audio fingerprints and return similarity score (0-1)."""
        if not fp1 or not fp2:
            return 0.0
        
        # Weights for different aspects
        weights = {
            'duration': 0.15,
            'sample_rate': 0.10,
            'channels': 0.05,
            'rms_energy': 0.20,
            'spectral_centroid': 0.15,
            'zero_crossing_rate': 0.10,
            'peak_amplitude': 0.15,
            'dynamic_range': 0.10
        }
        
        similarity_score = 0.0
        
        # Duration similarity
        if fp1.duration > 0 and fp2.duration > 0:
            duration_ratio = min(fp1.duration, fp2.duration) / max(fp1.duration, fp2.duration)
            similarity_score += weights['duration'] * duration_ratio
        
        # Sample rate exact match
        if fp1.sample_rate == fp2.sample_rate:
            similarity_score += weights['sample_rate']
        
        # Channels exact match
        if fp1.channels == fp2.channels:
            similarity_score += weights['channels']
        
        # Energy similarity
        if fp1.rms_energy > 0 and fp2.rms_energy > 0:
            energy_ratio = min(fp1.rms_energy, fp2.rms_energy) / max(fp1.rms_energy, fp2.rms_energy)
            similarity_score += weights['rms_energy'] * energy_ratio
        
        # Spectral centroid similarity
        if fp1.spectral_centroid > 0 and fp2.spectral_centroid > 0:
            centroid_diff = abs(fp1.spectral_centroid - fp2.spectral_centroid)
            max_centroid = max(fp1.spectral_centroid, fp2.spectral_centroid)
            centroid_similarity = max(0, 1 - centroid_diff / max_centroid)
            similarity_score += weights['spectral_centroid'] * centroid_similarity
        
        # Zero crossing rate similarity
        zcr_diff = abs(fp1.zero_crossing_rate - fp2.zero_crossing_rate)
        zcr_similarity = max(0, 1 - zcr_diff)
        similarity_score += weights['zero_crossing_rate'] * zcr_similarity
        
        # Peak amplitude similarity
        if fp1.peak_amplitude > 0 and fp2.peak_amplitude > 0:
            peak_ratio = min(fp1.peak_amplitude, fp2.peak_amplitude) / max(fp1.peak_amplitude, fp2.peak_amplitude)
            similarity_score += weights['peak_amplitude'] * peak_ratio
        
        # Dynamic range similarity
        dr_diff = abs(fp1.dynamic_range - fp2.dynamic_range)
        dr_similarity = max(0, 1 - dr_diff / 60)  # 60dB max difference
        similarity_score += weights['dynamic_range'] * dr_similarity
        
        return min(1.0, similarity_score)

class AdvancedConsistencyValidator:
    """Advanced validator for cross-language binding consistency."""
    
    def __init__(self):
        self.bindings = {}
        self.test_scenarios = []
        self.results = []
        self._setup_bindings()
        self._setup_test_scenarios()
    
    def _setup_bindings(self):
        """Setup available bindings for testing."""
        # Python bindings
        try:
            import voirs_ffi as voirs
            self.bindings['python'] = {
                'module': voirs,
                'available': True,
                'type': 'python',
                'synthesize_func': self._synthesize_python,
                'features': self._detect_python_features(voirs)
            }
        except ImportError:
            self.bindings['python'] = {
                'module': None,
                'available': False,
                'type': 'python',
                'error': 'Module not found'
            }
        
        # C API bindings
        self.bindings['c_api'] = {
            'available': False,
            'type': 'c_api',
            'error': 'C API testing requires compiled library'
        }
        
        # Node.js bindings
        self.bindings['nodejs'] = {
            'available': self._check_nodejs_availability(),
            'type': 'nodejs',
            'synthesize_func': self._synthesize_nodejs,
            'features': self._detect_nodejs_features() if self._check_nodejs_availability() else {}
        }
        
        # WebAssembly bindings
        self.bindings['wasm'] = {
            'available': False,
            'type': 'wasm',
            'error': 'WASM testing requires browser environment'
        }
    
    def _setup_test_scenarios(self):
        """Setup comprehensive test scenarios."""
        self.test_scenarios = [
            # Basic text synthesis
            {
                'name': 'basic_short_text',
                'text': 'Hello world',
                'config': {},
                'expected_features': ['synthesis']
            },
            {
                'name': 'basic_medium_text', 
                'text': 'This is a medium length test sentence for consistency validation.',
                'config': {},
                'expected_features': ['synthesis']
            },
            {
                'name': 'basic_long_text',
                'text': 'This is a much longer test sentence that contains multiple clauses and should provide sufficient audio content for detailed consistency analysis across different language bindings.',
                'config': {},
                'expected_features': ['synthesis']
            },
            
            # Different configurations
            {
                'name': 'config_speaking_rate_slow',
                'text': 'Testing slower speaking rate configuration.',
                'config': {'speaking_rate': 0.8},
                'expected_features': ['synthesis', 'config']
            },
            {
                'name': 'config_speaking_rate_fast',
                'text': 'Testing faster speaking rate configuration.',
                'config': {'speaking_rate': 1.5},
                'expected_features': ['synthesis', 'config']
            },
            {
                'name': 'config_pitch_shift_low',
                'text': 'Testing lower pitch shift configuration.',
                'config': {'pitch_shift': -2.0},
                'expected_features': ['synthesis', 'config']
            },
            {
                'name': 'config_pitch_shift_high',
                'text': 'Testing higher pitch shift configuration.',
                'config': {'pitch_shift': 2.0},
                'expected_features': ['synthesis', 'config']
            },
            
            # Quality levels
            {
                'name': 'quality_low',
                'text': 'Testing low quality synthesis.',
                'config': {'quality': 'low'},
                'expected_features': ['synthesis', 'quality_control']
            },
            {
                'name': 'quality_high',
                'text': 'Testing high quality synthesis.',
                'config': {'quality': 'high'},
                'expected_features': ['synthesis', 'quality_control']
            },
            
            # Special characters and internationalization
            {
                'name': 'special_characters',
                'text': 'Testing special characters: @#$%^&*()_+-={}[]|;:",./<>?',
                'config': {},
                'expected_features': ['synthesis', 'special_chars']
            },
            {
                'name': 'unicode_content',
                'text': 'Testing Unicode: café naïve résumé 日本語 中文 العربية',
                'config': {},
                'expected_features': ['synthesis', 'unicode']
            },
            
            # SSML testing
            {
                'name': 'ssml_basic',
                'text': '<speak>This is <emphasis level="strong">basic SSML</emphasis> testing.</speak>',
                'config': {'format': 'ssml'},
                'expected_features': ['ssml']
            },
            {
                'name': 'ssml_complex',
                'text': '<speak><p>Complex SSML with <prosody rate="slow">rate changes</prosody> and <break time="500ms"/> pauses.</p></speak>',
                'config': {'format': 'ssml'},
                'expected_features': ['ssml', 'prosody', 'breaks']
            },
            
            # Error conditions
            {
                'name': 'error_empty_text',
                'text': '',
                'config': {},
                'expected_features': ['error_handling'],
                'expect_error': True
            },
            {
                'name': 'error_invalid_config',
                'text': 'Testing invalid configuration.',
                'config': {'speaking_rate': -1, 'quality': 'invalid'},
                'expected_features': ['error_handling'],
                'expect_error': True
            }
        ]
    
    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run comprehensive cross-language consistency validation."""
        print("Running Advanced Cross-Language Consistency Validation")
        print("=" * 65)
        
        validation_results = {
            'metadata': {
                'timestamp': time.time(),
                'total_scenarios': len(self.test_scenarios),
                'available_bindings': [name for name, info in self.bindings.items() if info.get('available', False)]
            },
            'binding_info': self._get_binding_info(),
            'scenario_results': {},
            'consistency_analysis': {},
            'feature_parity_matrix': {},
            'recommendations': []
        }
        
        available_bindings = validation_results['metadata']['available_bindings']
        
        if len(available_bindings) < 2:
            validation_results['status'] = 'insufficient_bindings'
            validation_results['message'] = f"Need at least 2 bindings for consistency validation. Only {len(available_bindings)} available."
            return validation_results
        
        print(f"Testing {len(available_bindings)} bindings across {len(self.test_scenarios)} scenarios...")
        
        # Run each test scenario
        for scenario in self.test_scenarios:
            scenario_name = scenario['name']
            print(f"\n--- Testing scenario: {scenario_name} ---")
            
            try:
                result = self._run_scenario_validation(scenario, available_bindings)
                validation_results['scenario_results'][scenario_name] = result
                
                if result.consistency_score >= 0.95:
                    print(f"✓ {scenario_name}: Highly consistent ({result.consistency_score:.3f})")
                elif result.consistency_score >= 0.85:
                    print(f"~ {scenario_name}: Mostly consistent ({result.consistency_score:.3f})")
                else:
                    print(f"✗ {scenario_name}: Inconsistent ({result.consistency_score:.3f})")
                    if result.discrepancies:
                        for discrepancy in result.discrepancies[:3]:  # Show first 3
                            print(f"    - {discrepancy}")
                
            except Exception as e:
                print(f"✗ {scenario_name}: Failed with error: {e}")
                validation_results['scenario_results'][scenario_name] = {
                    'error': str(e),
                    'consistency_score': 0.0
                }
        
        # Analyze overall consistency
        validation_results['consistency_analysis'] = self._analyze_overall_consistency(
            validation_results['scenario_results']
        )
        
        # Generate feature parity matrix
        validation_results['feature_parity_matrix'] = self._generate_feature_parity_matrix(
            available_bindings
        )
        
        # Generate recommendations
        validation_results['recommendations'] = self._generate_recommendations(
            validation_results
        )
        
        return validation_results
    
    def _run_scenario_validation(self, scenario: Dict[str, Any], 
                                available_bindings: List[str]) -> ConsistencyTestResult:
        """Run validation for a single scenario."""
        binding_results = {}
        audio_fingerprints = {}
        timing_analysis = {}
        feature_parity = {}
        error_analysis = {}
        
        # Execute scenario on each binding
        for binding_name in available_bindings:
            binding_info = self.bindings[binding_name]
            
            try:
                start_time = time.time()
                result = self._execute_scenario(binding_name, binding_info, scenario)
                end_time = time.time()
                
                execution_time = (end_time - start_time) * 1000  # ms
                timing_analysis[binding_name] = {
                    'execution_time_ms': execution_time,
                    'success': True
                }
                
                # Analyze audio if successful
                if 'audio' in result and result['audio'] and 'samples' in result['audio']:
                    samples = result['audio']['samples']
                    sample_rate = result['audio'].get('sample_rate', 22050)
                    
                    fingerprint = AudioAnalyzer.analyze_audio_samples(samples, sample_rate)
                    audio_fingerprints[binding_name] = fingerprint
                
                # Check feature support
                feature_parity[binding_name] = self._check_feature_support(
                    binding_name, binding_info, scenario
                )
                
                binding_results[binding_name] = {
                    'success': True,
                    'result': result,
                    'execution_time_ms': execution_time
                }
                
            except Exception as e:
                timing_analysis[binding_name] = {
                    'execution_time_ms': 0,
                    'success': False,
                    'error': str(e)
                }
                
                error_analysis[binding_name] = {
                    'error_type': type(e).__name__,
                    'error_message': str(e),
                    'expected_error': scenario.get('expect_error', False)
                }
                
                binding_results[binding_name] = {
                    'success': False,
                    'error': str(e)
                }
        
        # Calculate consistency score
        consistency_score = self._calculate_consistency_score(
            binding_results, audio_fingerprints, timing_analysis, scenario
        )
        
        # Identify discrepancies
        discrepancies = self._identify_discrepancies(
            binding_results, audio_fingerprints, timing_analysis
        )
        
        return ConsistencyTestResult(
            test_name=scenario['name'],
            binding_results=binding_results,
            consistency_score=consistency_score,
            discrepancies=discrepancies,
            audio_fingerprints=audio_fingerprints,
            timing_analysis=timing_analysis,
            feature_parity=feature_parity,
            error_analysis=error_analysis
        )
    
    def _execute_scenario(self, binding_name: str, binding_info: Dict[str, Any], 
                         scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a test scenario on a specific binding."""
        text = scenario['text']
        config = scenario.get('config', {})
        
        if binding_name == 'python':
            return self._synthesize_python(binding_info, text, config)
        elif binding_name == 'nodejs':
            return self._synthesize_nodejs(text, config)
        elif binding_name == 'c_api':
            return self._synthesize_c_api(binding_info, text, config)
        elif binding_name == 'wasm':
            return self._synthesize_wasm(text, config)
        else:
            raise ValueError(f"Unknown binding: {binding_name}")
    
    def _synthesize_python(self, binding_info: Dict[str, Any], 
                          text: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize using Python bindings."""
        voirs = binding_info['module']
        
        if config.get('format') == 'ssml':
            if hasattr(voirs, 'VoirsPipeline'):
                pipeline = voirs.VoirsPipeline()
                if hasattr(pipeline, 'synthesize_ssml'):
                    audio = pipeline.synthesize_ssml(text)
                else:
                    raise NotImplementedError("SSML not supported in Python bindings")
            else:
                raise RuntimeError("VoirsPipeline not found in Python bindings")
        else:
            pipeline = voirs.VoirsPipeline()
            
            # Apply configuration if supported
            if config and hasattr(pipeline, 'synthesize_with_config'):
                audio = pipeline.synthesize_with_config(text, config)
            else:
                audio = pipeline.synthesize(text)
        
        # Extract audio data
        if hasattr(audio, 'samples_as_list'):
            samples = audio.samples_as_list()
        elif hasattr(audio, 'samples'):
            samples = list(audio.samples) if hasattr(audio.samples, '__iter__') else []
        else:
            samples = []
        
        return {
            'audio': {
                'samples': samples,
                'sample_rate': getattr(audio, 'sample_rate', 22050),
                'channels': getattr(audio, 'channels', 1),
                'duration': getattr(audio, 'duration', len(samples) / 22050.0)
            },
            'config_applied': config,
            'binding': 'python'
        }
    
    def _synthesize_nodejs(self, text: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize using Node.js bindings."""
        # Create temporary Node.js script
        script_content = f'''
const {{ VoirsPipeline }} = require('../../index.js');

async function synthesize() {{
    try {{
        const pipeline = new VoirsPipeline();
        const text = {json.dumps(text)};
        const config = {json.dumps(config)};
        
        let audio;
        if (config.format === 'ssml') {{
            if (pipeline.synthesizeSsml) {{
                audio = await pipeline.synthesizeSsml(text);
            }} else {{
                throw new Error('SSML not supported');
            }}
        }} else if (Object.keys(config).length > 0) {{
            if (pipeline.synthesizeWithConfig) {{
                audio = await pipeline.synthesizeWithConfig(text, config);
            }} else {{
                audio = await pipeline.synthesize(text);
            }}
        }} else {{
            audio = await pipeline.synthesize(text);
        }}
        
        console.log(JSON.stringify({{
            audio: {{
                samples: Array.from(audio.samples || []),
                sample_rate: audio.sampleRate || 22050,
                channels: audio.channels || 1,
                duration: audio.duration || 0
            }},
            config_applied: config,
            binding: 'nodejs'
        }}));
    }} catch (error) {{
        console.error(JSON.stringify({{error: error.message}}));
        process.exit(1);
    }}
}}

synthesize();
'''
        
        # Write and execute script
        with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
            f.write(script_content)
            script_path = f.name
        
        try:
            result = subprocess.run(
                ['node', script_path],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=os.path.dirname(os.path.abspath(__file__))
            )
            
            if result.returncode != 0:
                error_info = json.loads(result.stderr) if result.stderr else {"error": "Unknown error"}
                raise RuntimeError(f"Node.js synthesis failed: {error_info.get('error')}")
            
            return json.loads(result.stdout)
            
        except subprocess.TimeoutExpired:
            raise RuntimeError("Node.js synthesis timed out")
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse Node.js output: {e}")
        finally:
            try:
                os.unlink(script_path)
            except:
                pass
    
    def _synthesize_c_api(self, binding_info: Dict[str, Any], 
                         text: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize using C API bindings."""
        raise NotImplementedError("C API synthesis requires compiled library")
    
    def _synthesize_wasm(self, text: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize using WASM bindings."""
        raise NotImplementedError("WASM synthesis requires browser environment")
    
    def _calculate_consistency_score(self, binding_results: Dict[str, Any],
                                   audio_fingerprints: Dict[str, AudioFingerprint],
                                   timing_analysis: Dict[str, Any],
                                   scenario: Dict[str, Any]) -> float:
        """Calculate overall consistency score for a scenario."""
        if len(binding_results) < 2:
            return 0.0
        
        successful_bindings = [
            name for name, result in binding_results.items()
            if result.get('success', False)
        ]
        
        if len(successful_bindings) < 2:
            # Check if all bindings failed consistently for error scenarios
            if scenario.get('expect_error', False):
                failed_bindings = [
                    name for name, result in binding_results.items()
                    if not result.get('success', False)
                ]
                if len(failed_bindings) == len(binding_results):
                    return 1.0  # Consistent failure is good for error scenarios
            return 0.0
        
        scores = []
        
        # Audio consistency (if available)
        if len(audio_fingerprints) >= 2:
            fingerprint_pairs = [
                (fp1, fp2) for i, fp1 in enumerate(audio_fingerprints.values())
                for fp2 in list(audio_fingerprints.values())[i+1:]
            ]
            
            audio_similarities = [
                AudioAnalyzer.compare_fingerprints(fp1, fp2)
                for fp1, fp2 in fingerprint_pairs
            ]
            
            if audio_similarities:
                scores.append(statistics.mean(audio_similarities))
        
        # Timing consistency
        execution_times = [
            timing['execution_time_ms'] for timing in timing_analysis.values()
            if timing.get('success', False)
        ]
        
        if len(execution_times) >= 2:
            # Calculate coefficient of variation for timing
            mean_time = statistics.mean(execution_times)
            if mean_time > 0:
                std_time = statistics.stdev(execution_times) if len(execution_times) > 1 else 0
                cv = std_time / mean_time
                timing_consistency = max(0, 1 - cv)  # Lower CV = higher consistency
                scores.append(timing_consistency)
        
        # Basic result consistency
        if len(successful_bindings) >= 2:
            # Check if all successful bindings returned similar basic properties
            sample_rates = []
            channels = []
            durations = []
            
            for binding in successful_bindings:
                result = binding_results[binding]['result']
                if 'audio' in result:
                    audio = result['audio']
                    sample_rates.append(audio.get('sample_rate', 0))
                    channels.append(audio.get('channels', 0))
                    durations.append(audio.get('duration', 0))
            
            # Sample rate consistency
            if sample_rates and len(set(sample_rates)) == 1:
                scores.append(1.0)
            elif sample_rates:
                scores.append(0.5)  # Partial credit for having sample rates
            
            # Channel consistency
            if channels and len(set(channels)) == 1:
                scores.append(1.0)
            elif channels:
                scores.append(0.5)
            
            # Duration consistency (within 10% tolerance)
            if len(durations) >= 2:
                max_duration = max(durations)
                min_duration = min(durations)
                if max_duration > 0:
                    duration_ratio = min_duration / max_duration
                    scores.append(duration_ratio)
        
        return statistics.mean(scores) if scores else 0.0
    
    def _identify_discrepancies(self, binding_results: Dict[str, Any],
                              audio_fingerprints: Dict[str, AudioFingerprint],
                              timing_analysis: Dict[str, Any]) -> List[str]:
        """Identify specific discrepancies between bindings."""
        discrepancies = []
        
        successful_bindings = [
            name for name, result in binding_results.items()
            if result.get('success', False)
        ]
        
        if len(successful_bindings) < 2:
            discrepancies.append("Insufficient successful bindings for comparison")
            return discrepancies
        
        # Check timing discrepancies
        execution_times = {
            name: timing['execution_time_ms']
            for name, timing in timing_analysis.items()
            if timing.get('success', False)
        }
        
        if len(execution_times) >= 2:
            times = list(execution_times.values())
            max_time = max(times)
            min_time = min(times)
            
            if max_time > min_time * 3:  # 3x difference threshold
                slowest = max(execution_times, key=execution_times.get)
                fastest = min(execution_times, key=execution_times.get)
                discrepancies.append(
                    f"Large timing difference: {slowest} ({execution_times[slowest]:.1f}ms) "
                    f"vs {fastest} ({execution_times[fastest]:.1f}ms)"
                )
        
        # Check audio property discrepancies
        if len(audio_fingerprints) >= 2:
            # Sample rate discrepancies
            sample_rates = {name: fp.sample_rate for name, fp in audio_fingerprints.items()}
            unique_rates = set(sample_rates.values())
            if len(unique_rates) > 1:
                discrepancies.append(f"Sample rate mismatch: {dict(sample_rates)}")
            
            # Duration discrepancies (>10% difference)
            durations = {name: fp.duration for name, fp in audio_fingerprints.items()}
            if len(durations) >= 2:
                min_duration = min(durations.values())
                max_duration = max(durations.values())
                if max_duration > min_duration * 1.1:
                    discrepancies.append(f"Duration mismatch: {dict(durations)}")
            
            # Energy level discrepancies
            energies = {name: fp.rms_energy for name, fp in audio_fingerprints.items()}
            if len(energies) >= 2:
                energy_values = list(energies.values())
                if max(energy_values) > min(energy_values) * 2:  # 2x difference
                    discrepancies.append(f"Energy level mismatch: {dict(energies)}")
        
        return discrepancies
    
    def _detect_python_features(self, voirs_module) -> Dict[str, bool]:
        """Detect available features in Python bindings."""
        features = {}
        
        if hasattr(voirs_module, 'VoirsPipeline'):
            pipeline_class = voirs_module.VoirsPipeline
            
            # Basic synthesis
            features['synthesis'] = hasattr(pipeline_class, 'synthesize')
            
            # SSML support
            features['ssml'] = hasattr(pipeline_class, 'synthesize_ssml')
            
            # Configuration support
            features['config'] = hasattr(pipeline_class, 'synthesize_with_config')
            
            # Streaming support
            features['streaming'] = hasattr(pipeline_class, 'synthesize_streaming')
            
            # Callback support
            features['callbacks'] = hasattr(pipeline_class, 'synthesize_with_callbacks')
            
            # Voice management
            features['voice_management'] = (
                hasattr(pipeline_class, 'list_voices') and
                hasattr(pipeline_class, 'set_voice')
            )
            
            # Quality control
            features['quality_control'] = hasattr(pipeline_class, 'set_quality')
            
            # Audio format support
            if hasattr(pipeline_class, 'synthesize'):
                try:
                    # Try to create instance to test further
                    pipeline = pipeline_class()
                    features['audio_formats'] = hasattr(pipeline, 'supported_formats')
                except:
                    features['audio_formats'] = False
        
        return features
    
    def _detect_nodejs_features(self) -> Dict[str, bool]:
        """Detect available features in Node.js bindings."""
        # This would require executing Node.js code to detect features
        # For now, return basic feature set
        return {
            'synthesis': True,
            'ssml': True,
            'config': True,
            'streaming': False,
            'callbacks': False,
            'voice_management': False,
            'quality_control': False,
            'audio_formats': False
        }
    
    def _check_nodejs_availability(self) -> bool:
        """Check if Node.js bindings are available."""
        try:
            result = subprocess.run(['node', '--version'], capture_output=True)
            if result.returncode == 0:
                # Check if index.js exists
                index_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'index.js')
                return os.path.exists(index_path)
        except:
            pass
        return False
    
    def _check_feature_support(self, binding_name: str, binding_info: Dict[str, Any],
                              scenario: Dict[str, Any]) -> Dict[str, bool]:
        """Check which features are supported by a binding for a scenario."""
        expected_features = scenario.get('expected_features', [])
        binding_features = binding_info.get('features', {})
        
        feature_support = {}
        for feature in expected_features:
            feature_support[feature] = binding_features.get(feature, False)
        
        return feature_support
    
    def _analyze_overall_consistency(self, scenario_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze overall consistency across all scenarios."""
        analysis = {
            'overall_score': 0.0,
            'scenario_scores': {},
            'consistency_distribution': {},
            'problem_scenarios': [],
            'excellent_scenarios': []
        }
        
        scores = []
        for scenario_name, result in scenario_results.items():
            if isinstance(result, dict) and 'consistency_score' in result:
                score = result['consistency_score']
                scores.append(score)
                analysis['scenario_scores'][scenario_name] = score
                
                if score < 0.7:
                    analysis['problem_scenarios'].append({
                        'name': scenario_name,
                        'score': score,
                        'issues': result.get('discrepancies', [])
                    })
                elif score >= 0.95:
                    analysis['excellent_scenarios'].append({
                        'name': scenario_name,
                        'score': score
                    })
        
        if scores:
            analysis['overall_score'] = statistics.mean(scores)
            
            # Distribution analysis
            high_consistency = sum(1 for s in scores if s >= 0.9)
            medium_consistency = sum(1 for s in scores if 0.7 <= s < 0.9)
            low_consistency = sum(1 for s in scores if s < 0.7)
            
            analysis['consistency_distribution'] = {
                'high': high_consistency,
                'medium': medium_consistency,
                'low': low_consistency,
                'total': len(scores)
            }
        
        return analysis
    
    def _generate_feature_parity_matrix(self, available_bindings: List[str]) -> Dict[str, Any]:
        """Generate feature parity matrix across bindings."""
        matrix = {
            'bindings': available_bindings,
            'features': {},
            'parity_score': 0.0
        }
        
        # Collect all features across bindings
        all_features = set()
        for binding_name in available_bindings:
            binding_features = self.bindings[binding_name].get('features', {})
            all_features.update(binding_features.keys())
        
        # Create feature matrix
        for feature in all_features:
            matrix['features'][feature] = {}
            for binding_name in available_bindings:
                binding_features = self.bindings[binding_name].get('features', {})
                matrix['features'][feature][binding_name] = binding_features.get(feature, False)
        
        # Calculate parity score
        total_feature_binding_pairs = len(all_features) * len(available_bindings)
        if total_feature_binding_pairs > 0:
            supported_pairs = sum(
                sum(binding_support.values())
                for binding_support in matrix['features'].values()
            )
            matrix['parity_score'] = supported_pairs / total_feature_binding_pairs
        
        return matrix
    
    def _generate_recommendations(self, validation_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        consistency_analysis = validation_results.get('consistency_analysis', {})
        overall_score = consistency_analysis.get('overall_score', 0.0)
        
        if overall_score < 0.8:
            recommendations.append(
                f"Overall consistency score is low ({overall_score:.3f}). "
                "Review binding implementations for discrepancies."
            )
        
        problem_scenarios = consistency_analysis.get('problem_scenarios', [])
        if problem_scenarios:
            recommendations.append(
                f"Found {len(problem_scenarios)} problematic scenarios. "
                f"Focus on: {', '.join([s['name'] for s in problem_scenarios[:3]])}"
            )
        
        feature_parity = validation_results.get('feature_parity_matrix', {})
        parity_score = feature_parity.get('parity_score', 1.0)
        
        if parity_score < 0.8:
            recommendations.append(
                f"Feature parity is low ({parity_score:.3f}). "
                "Consider implementing missing features across bindings."
            )
        
        available_bindings = validation_results.get('metadata', {}).get('available_bindings', [])
        if len(available_bindings) < 3:
            recommendations.append(
                "Consider building additional language bindings to improve test coverage."
            )
        
        return recommendations
    
    def _get_binding_info(self) -> Dict[str, Any]:
        """Get comprehensive binding information."""
        info = {}
        for name, binding in self.bindings.items():
            info[name] = {
                'available': binding.get('available', False),
                'type': binding.get('type', 'unknown'),
                'features': binding.get('features', {}),
                'error': binding.get('error', None)
            }
            
            if binding.get('available', False) and 'module' in binding:
                try:
                    info[name]['version'] = getattr(binding['module'], '__version__', 'unknown')
                except:
                    info[name]['version'] = 'unknown'
        
        return info

def main():
    """Main entry point for advanced consistency validation."""
    print("VoiRS Advanced Cross-Language Consistency Validator")
    print("=" * 60)
    
    validator = AdvancedConsistencyValidator()
    
    try:
        results = validator.run_comprehensive_validation()
        
        # Print summary
        metadata = results.get('metadata', {})
        print(f"\nValidation Summary:")
        print(f"  Scenarios tested: {metadata.get('total_scenarios', 0)}")
        print(f"  Available bindings: {len(metadata.get('available_bindings', []))}")
        
        consistency_analysis = results.get('consistency_analysis', {})
        overall_score = consistency_analysis.get('overall_score', 0.0)
        print(f"  Overall consistency: {overall_score:.3f}")
        
        distribution = consistency_analysis.get('consistency_distribution', {})
        if distribution:
            print(f"  High consistency scenarios: {distribution.get('high', 0)}")
            print(f"  Medium consistency scenarios: {distribution.get('medium', 0)}")
            print(f"  Low consistency scenarios: {distribution.get('low', 0)}")
        
        # Print recommendations
        recommendations = results.get('recommendations', [])
        if recommendations:
            print(f"\nRecommendations:")
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec}")
        
        # Save detailed results
        output_file = f"advanced_consistency_report_{int(time.time())}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nDetailed results saved to: {output_file}")
        
        if overall_score >= 0.9:
            print(f"\n🎉 Excellent consistency across language bindings!")
            return 0
        elif overall_score >= 0.8:
            print(f"\n✅ Good consistency with minor discrepancies.")
            return 0
        else:
            print(f"\n⚠️  Significant consistency issues detected.")
            return 1
    
    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
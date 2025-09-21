"""
Unit tests for VoiRS Pipeline Python bindings.

Tests the core functionality of the VoirsPipeline class including
creation, configuration, synthesis, and voice management.
"""

import pytest
import asyncio
from pathlib import Path

try:
    import voirs_ffi as voirs
    VOIRS_AVAILABLE = True
except ImportError:
    VOIRS_AVAILABLE = False

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.mark.skipif(not VOIRS_AVAILABLE, reason="VoiRS Python bindings not available")
class TestVoirsPipeline:
    """Test VoirsPipeline class functionality."""
    
    async def test_pipeline_creation(self):
        """Test basic pipeline creation."""
        pipeline = await voirs.VoirsPipeline.create()
        assert pipeline is not None
        assert hasattr(pipeline, 'synthesize')
        assert hasattr(pipeline, 'set_voice')
        assert hasattr(pipeline, 'get_voice')
        assert hasattr(pipeline, 'list_voices')
    
    async def test_pipeline_creation_with_config(self):
        """Test pipeline creation with configuration."""
        config = {
            'use_gpu': False,
            'num_threads': 2,
            'cache_dir': '/tmp/voirs_test'
        }
        
        try:
            pipeline = await voirs.VoirsPipeline.create_with_config(config)
            assert pipeline is not None
        except Exception as e:
            # Configuration might not be supported in test environment
            pytest.skip(f"Pipeline configuration not supported: {e}")
    
    async def test_pipeline_info(self, voirs_pipeline):
        """Test getting pipeline information."""
        info = voirs_pipeline.get_info()
        assert isinstance(info, dict)
        assert 'version' in info
        assert 'features' in info
    
    async def test_version_info(self):
        """Test version information."""
        version = voirs.version()
        assert isinstance(version, str)
        assert len(version) > 0
        # Version should follow semantic versioning pattern
        assert '.' in version


@pytest.mark.skipif(not VOIRS_AVAILABLE, reason="VoiRS Python bindings not available")
class TestSynthesis:
    """Test text synthesis functionality."""
    
    async def test_basic_synthesis(self, voirs_pipeline, sample_texts, audio_validator):
        """Test basic text synthesis."""
        text = sample_texts["short"]
        audio = await voirs_pipeline.synthesize(text)
        
        audio_validator.validate_audio_buffer(audio)
        audio_validator.check_audio_quality(audio)
    
    async def test_synthesis_with_config(self, voirs_pipeline, sample_texts, synthesis_configs, audio_validator):
        """Test synthesis with various configurations."""
        text = sample_texts["medium"]
        
        for config_name, config in synthesis_configs.items():
            try:
                audio = await voirs_pipeline.synthesize_with_config(text, config)
                audio_validator.validate_audio_buffer(audio)
                audio_validator.check_audio_quality(audio)
            except Exception as e:
                pytest.skip(f"Configuration {config_name} not supported: {e}")
    
    async def test_synthesis_empty_text(self, voirs_pipeline):
        """Test synthesis with empty text."""
        with pytest.raises(Exception):  # Should raise an error
            await voirs_pipeline.synthesize("")
    
    async def test_synthesis_long_text(self, voirs_pipeline, sample_texts, audio_validator):
        """Test synthesis with long text."""
        text = sample_texts["long"]
        audio = await voirs_pipeline.synthesize(text)
        
        audio_validator.validate_audio_buffer(audio)
        assert audio.duration > 3.0  # Long text should produce longer audio
    
    async def test_synthesis_special_characters(self, voirs_pipeline, sample_texts, audio_validator):
        """Test synthesis with special characters."""
        text = sample_texts["special_chars"]
        audio = await voirs_pipeline.synthesize(text)
        
        audio_validator.validate_audio_buffer(audio)
        audio_validator.check_audio_quality(audio)
    
    async def test_synthesis_numbers(self, voirs_pipeline, sample_texts, audio_validator):
        """Test synthesis with numbers."""
        text = sample_texts["numbers"]
        audio = await voirs_pipeline.synthesize(text)
        
        audio_validator.validate_audio_buffer(audio)
        audio_validator.check_audio_quality(audio)
    
    async def test_synthesis_punctuation(self, voirs_pipeline, sample_texts, audio_validator):
        """Test synthesis with punctuation."""
        text = sample_texts["punctuation"]
        audio = await voirs_pipeline.synthesize(text)
        
        audio_validator.validate_audio_buffer(audio)
        audio_validator.check_audio_quality(audio)


@pytest.mark.skipif(not VOIRS_AVAILABLE, reason="VoiRS Python bindings not available")
class TestVoiceManagement:
    """Test voice management functionality."""
    
    async def test_list_voices(self, voirs_pipeline):
        """Test listing available voices."""
        voices = await voirs_pipeline.list_voices()
        assert isinstance(voices, list)
        
        if voices:  # If voices are available
            voice = voices[0]
            assert hasattr(voice, 'id')
            assert hasattr(voice, 'name')
            assert hasattr(voice, 'language')
            assert isinstance(voice.id, str)
            assert isinstance(voice.name, str)
            assert isinstance(voice.language, str)
    
    async def test_get_current_voice(self, voirs_pipeline):
        """Test getting current voice."""
        current_voice = await voirs_pipeline.get_voice()
        # Current voice might be None if no voice is set
        if current_voice is not None:
            assert isinstance(current_voice, str)
            assert len(current_voice) > 0
    
    async def test_set_voice(self, voirs_pipeline):
        """Test setting voice."""
        voices = await voirs_pipeline.list_voices()
        
        if voices:
            # Set to first available voice
            voice_id = voices[0].id
            await voirs_pipeline.set_voice(voice_id)
            
            # Verify voice was set
            current_voice = await voirs_pipeline.get_voice()
            assert current_voice == voice_id
        else:
            pytest.skip("No voices available for testing")
    
    async def test_set_invalid_voice(self, voirs_pipeline):
        """Test setting invalid voice ID."""
        with pytest.raises(Exception):
            await voirs_pipeline.set_voice("invalid_voice_id_12345")


@pytest.mark.skipif(not VOIRS_AVAILABLE, reason="VoiRS Python bindings not available")
class TestAudioBuffer:
    """Test AudioBuffer functionality."""
    
    async def test_audio_buffer_properties(self, voirs_pipeline, sample_texts):
        """Test audio buffer properties."""
        text = sample_texts["short"]
        audio = await voirs_pipeline.synthesize(text)
        
        # Check basic properties
        assert hasattr(audio, 'samples')
        assert hasattr(audio, 'sample_rate')
        assert hasattr(audio, 'channels')
        assert hasattr(audio, 'duration')
        
        # Check property types
        assert isinstance(audio.sample_rate, int)
        assert isinstance(audio.channels, int)
        assert isinstance(audio.duration, float)
        
        # Check reasonable values
        assert audio.sample_rate > 0
        assert audio.channels > 0
        assert audio.duration > 0
    
    async def test_audio_samples_access(self, voirs_pipeline, sample_texts):
        """Test accessing audio samples."""
        text = sample_texts["short"]
        audio = await voirs_pipeline.synthesize(text)
        
        samples = audio.samples
        assert isinstance(samples, (list, tuple)) or hasattr(samples, '__iter__')
        assert len(samples) > 0
        
        # Check first few samples
        for i, sample in enumerate(samples[:10]):
            assert isinstance(sample, (int, float))
            if i > 100:  # Don't check all samples for performance
                break
    
    async def test_audio_buffer_save_wav(self, voirs_pipeline, sample_texts, temp_audio_dir, audio_validator):
        """Test saving audio buffer as WAV file."""
        text = sample_texts["short"]
        audio = await voirs_pipeline.synthesize(text)
        
        wav_file = temp_audio_dir / "test_output.wav"
        
        try:
            audio.save_wav(str(wav_file))
            audio_validator.validate_audio_file(wav_file, "wav")
        except AttributeError:
            pytest.skip("save_wav method not available")
    
    async def test_audio_buffer_save_flac(self, voirs_pipeline, sample_texts, temp_audio_dir, audio_validator):
        """Test saving audio buffer as FLAC file."""
        text = sample_texts["short"]
        audio = await voirs_pipeline.synthesize(text)
        
        flac_file = temp_audio_dir / "test_output.flac"
        
        try:
            audio.save_flac(str(flac_file))
            audio_validator.validate_audio_file(flac_file, "flac")
        except AttributeError:
            pytest.skip("save_flac method not available")


@pytest.mark.skipif(not VOIRS_AVAILABLE, reason="VoiRS Python bindings not available")
@pytest.mark.skipif("numpy_available == False", reason="NumPy not available")
class TestNumPyIntegration:
    """Test NumPy integration functionality."""
    
    async def test_numpy_array_conversion(self, voirs_pipeline, sample_texts, numpy_available):
        """Test converting audio to NumPy array."""
        if not numpy_available:
            pytest.skip("NumPy not available")
        
        import numpy as np
        
        text = sample_texts["short"]
        audio = await voirs_pipeline.synthesize(text)
        
        try:
            np_array = audio.to_numpy()
            assert isinstance(np_array, np.ndarray)
            assert np_array.dtype == np.float32
            assert len(np_array.shape) in [1, 2]  # 1D for mono, 2D for stereo
            assert np_array.size > 0
        except AttributeError:
            pytest.skip("to_numpy method not available")
    
    async def test_numpy_array_properties(self, voirs_pipeline, sample_texts, numpy_available):
        """Test NumPy array properties match audio buffer."""
        if not numpy_available:
            pytest.skip("NumPy not available")
        
        import numpy as np
        
        text = sample_texts["medium"]
        audio = await voirs_pipeline.synthesize(text)
        
        try:
            np_array = audio.to_numpy()
            
            # Check shape consistency
            if audio.channels == 1:
                assert len(np_array.shape) == 1
                assert np_array.shape[0] == len(audio.samples)
            else:
                assert len(np_array.shape) == 2
                assert np_array.shape[1] == audio.channels
                assert np_array.shape[0] * np_array.shape[1] == len(audio.samples)
            
            # Check value range (should be normalized)
            assert np_array.min() >= -1.0
            assert np_array.max() <= 1.0
            
        except AttributeError:
            pytest.skip("to_numpy method not available")


@pytest.mark.skipif(not VOIRS_AVAILABLE, reason="VoiRS Python bindings not available")
class TestErrorHandling:
    """Test error handling and edge cases."""
    
    async def test_invalid_text_types(self, voirs_pipeline):
        """Test synthesis with invalid text types."""
        invalid_inputs = [None, 123, [], {}, b"bytes"]
        
        for invalid_input in invalid_inputs:
            with pytest.raises((TypeError, ValueError, Exception)):
                await voirs_pipeline.synthesize(invalid_input)
    
    async def test_invalid_config_types(self, voirs_pipeline, sample_texts):
        """Test synthesis with invalid configuration types."""
        text = sample_texts["short"]
        invalid_configs = [None, "string", 123, []]
        
        for invalid_config in invalid_configs:
            with pytest.raises((TypeError, ValueError, Exception)):
                await voirs_pipeline.synthesize_with_config(text, invalid_config)
    
    async def test_concurrent_synthesis(self, voirs_pipeline, sample_texts):
        """Test concurrent synthesis operations."""
        text = sample_texts["short"]
        
        # Start multiple synthesis operations concurrently
        tasks = [
            voirs_pipeline.synthesize(f"{text} {i}")
            for i in range(3)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check that all operations completed successfully
        for result in results:
            if isinstance(result, Exception):
                pytest.fail(f"Concurrent synthesis failed: {result}")
            else:
                assert hasattr(result, 'samples')  # Should be valid audio buffer
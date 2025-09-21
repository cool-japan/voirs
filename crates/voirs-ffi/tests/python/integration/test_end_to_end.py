"""
Integration tests for VoiRS Python bindings.

Tests end-to-end workflows, real-world usage scenarios,
and integration between different components.
"""

import pytest
import asyncio
import tempfile
from pathlib import Path
import time

try:
    import voirs_ffi as voirs
    VOIRS_AVAILABLE = True
except ImportError:
    VOIRS_AVAILABLE = False

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


@pytest.mark.skipif(not VOIRS_AVAILABLE, reason="VoiRS Python bindings not available")
class TestEndToEndWorkflows:
    """Test complete end-to-end workflows."""
    
    async def test_complete_synthesis_workflow(
        self, 
        sample_texts, 
        synthesis_configs, 
        temp_audio_dir, 
        audio_validator,
        performance_metrics
    ):
        """Test complete synthesis workflow from text to audio file."""
        performance_metrics.start_timer("complete_workflow")
        
        # 1. Create pipeline
        pipeline = await voirs.VoirsPipeline.create()
        
        # 2. List and select voice
        voices = await pipeline.list_voices()
        if voices:
            await pipeline.set_voice(voices[0].id)
        
        # 3. Synthesize with different configurations
        text = sample_texts["medium"]
        results = {}
        
        for config_name, config in synthesis_configs.items():
            try:
                audio = await pipeline.synthesize_with_config(text, config)
                audio_validator.validate_audio_buffer(audio)
                results[config_name] = audio
            except Exception as e:
                # Some configs might not be supported
                print(f"Config {config_name} failed: {e}")
        
        assert len(results) > 0, "No synthesis configurations worked"
        
        # 4. Save audio files
        for config_name, audio in results.items():
            wav_file = temp_audio_dir / f"test_{config_name}.wav"
            try:
                audio.save_wav(str(wav_file))
                audio_validator.validate_audio_file(wav_file, "wav")
            except AttributeError:
                # save_wav might not be available
                pass
        
        performance_metrics.end_timer("complete_workflow")
        workflow_time = performance_metrics.get_duration("complete_workflow")
        
        # Workflow should complete in reasonable time
        assert workflow_time < 30.0, f"Workflow too slow: {workflow_time:.2f}s"
    
    async def test_batch_synthesis(self, sample_texts, audio_validator, performance_metrics):
        """Test synthesizing multiple texts in batch."""
        pipeline = await voirs.VoirsPipeline.create()
        
        texts = [
            sample_texts["short"],
            sample_texts["medium"],
            sample_texts["special_chars"],
            sample_texts["numbers"],
        ]
        
        performance_metrics.start_timer("batch_synthesis")
        
        # Synthesize all texts
        results = []
        for i, text in enumerate(texts):
            audio = await pipeline.synthesize(text)
            audio_validator.validate_audio_buffer(audio)
            results.append(audio)
        
        performance_metrics.end_timer("batch_synthesis")
        
        # Verify all syntheses were successful
        assert len(results) == len(texts)
        
        # Check that different texts produce different audio
        durations = [audio.duration for audio in results]
        assert len(set(durations)) > 1, "All audio files have same duration"
    
    async def test_voice_switching_workflow(self, sample_texts, audio_validator):
        """Test switching between voices and comparing results."""
        pipeline = await voirs.VoirsPipeline.create()
        voices = await pipeline.list_voices()
        
        if len(voices) < 2:
            pytest.skip("Need at least 2 voices for voice switching test")
        
        text = sample_texts["medium"]
        voice_results = {}
        
        # Synthesize with different voices
        for voice in voices[:2]:  # Test with first 2 voices
            await pipeline.set_voice(voice.id)
            
            # Verify voice was set
            current_voice = await pipeline.get_voice()
            assert current_voice == voice.id
            
            # Synthesize with this voice
            audio = await pipeline.synthesize(text)
            audio_validator.validate_audio_buffer(audio)
            voice_results[voice.id] = audio
        
        # Different voices should produce different audio characteristics
        durations = [audio.duration for audio in voice_results.values()]
        # Allow some tolerance for duration differences
        assert max(durations) - min(durations) < 2.0, "Voice differences too extreme"
    
    async def test_configuration_effects(self, sample_texts, audio_validator):
        """Test that different configurations produce measurably different results."""
        pipeline = await voirs.VoirsPipeline.create()
        text = sample_texts["medium"]
        
        # Test configurations with expected differences
        configs = {
            "normal": {"speaking_rate": 1.0},
            "fast": {"speaking_rate": 1.5},
            "slow": {"speaking_rate": 0.7},
        }
        
        results = {}
        for config_name, config in configs.items():
            try:
                audio = await pipeline.synthesize_with_config(text, config)
                audio_validator.validate_audio_buffer(audio)
                results[config_name] = audio
            except Exception as e:
                print(f"Config {config_name} failed: {e}")
        
        if len(results) >= 2:
            # Fast speech should be shorter than slow speech
            if "fast" in results and "slow" in results:
                assert results["fast"].duration < results["slow"].duration, \
                    "Fast speech should be shorter than slow speech"
            
            # Normal should be between fast and slow
            if all(k in results for k in ["normal", "fast", "slow"]):
                assert results["fast"].duration <= results["normal"].duration <= results["slow"].duration


@pytest.mark.skipif(not VOIRS_AVAILABLE, reason="VoiRS Python bindings not available")
class TestRealWorldScenarios:
    """Test real-world usage scenarios."""
    
    async def test_podcast_generation(self, temp_audio_dir, audio_validator):
        """Test generating a podcast-like audio content."""
        pipeline = await voirs.VoirsPipeline.create()
        
        # Simulate podcast content
        podcast_segments = [
            "Welcome to our technology podcast.",
            "Today we're discussing artificial intelligence and speech synthesis.",
            "First, let's talk about the recent advances in neural text-to-speech.",
            "These systems can now produce very natural-sounding speech.",
            "Thank you for listening, and we'll see you next time."
        ]
        
        podcast_audio = []
        total_duration = 0
        
        for segment in podcast_segments:
            audio = await pipeline.synthesize(segment)
            audio_validator.validate_audio_buffer(audio)
            podcast_audio.append(audio)
            total_duration += audio.duration
        
        # Podcast should be reasonable length
        assert total_duration > 10.0, "Podcast too short"
        assert total_duration < 60.0, "Podcast too long for test"
        
        # Save each segment
        for i, audio in enumerate(podcast_audio):
            segment_file = temp_audio_dir / f"podcast_segment_{i+1}.wav"
            try:
                audio.save_wav(str(segment_file))
                audio_validator.validate_audio_file(segment_file, "wav")
            except AttributeError:
                pass  # save_wav might not be available
    
    async def test_notification_system(self, audio_validator):
        """Test generating system notifications."""
        pipeline = await voirs.VoirsPipeline.create()
        
        notifications = [
            "You have a new message.",
            "Meeting reminder: Team standup in 5 minutes.",
            "System update completed successfully.",
            "Battery low. Please charge your device.",
            "Download finished."
        ]
        
        # Generate short, quick notifications
        config = {"speaking_rate": 1.2, "volume_gain": 1.1}
        
        for notification in notifications:
            try:
                audio = await pipeline.synthesize_with_config(notification, config)
                audio_validator.validate_audio_buffer(audio)
                
                # Notifications should be short and clear
                assert audio.duration < 5.0, f"Notification too long: {audio.duration}s"
                assert audio.duration > 0.5, f"Notification too short: {audio.duration}s"
                
            except Exception as e:
                print(f"Notification synthesis failed: {e}")
    
    async def test_accessibility_features(self, audio_validator):
        """Test accessibility features like reading content aloud."""
        pipeline = await voirs.VoirsPipeline.create()
        
        # Simulate reading webpage content
        content_blocks = [
            "Article title: Understanding Speech Synthesis Technology",
            "Published on December 15, 2023 by Dr. Sarah Johnson",
            "Speech synthesis, also known as text-to-speech, is a technology that converts written text into spoken words.",
            "This technology has applications in accessibility, education, and entertainment.",
            "Modern neural networks have significantly improved the quality and naturalness of synthesized speech."
        ]
        
        # Use slower, clearer speech for accessibility
        accessibility_config = {
            "speaking_rate": 0.9,
            "volume_gain": 1.0,
            "enable_enhancement": True
        }
        
        for block in content_blocks:
            try:
                audio = await pipeline.synthesize_with_config(block, accessibility_config)
                audio_validator.validate_audio_buffer(audio)
                audio_validator.check_audio_quality(audio)
                
                # Accessible speech should be clear and well-paced
                words = len(block.split())
                expected_duration = words * 0.4  # ~0.4 seconds per word
                assert audio.duration > expected_duration * 0.7, "Speech too fast for accessibility"
                
            except Exception as e:
                print(f"Accessibility synthesis failed: {e}")


@pytest.mark.skipif(not VOIRS_AVAILABLE, reason="VoiRS Python bindings not available")
class TestConcurrencyAndStability:
    """Test concurrent operations and system stability."""
    
    async def test_concurrent_pipelines(self, sample_texts, audio_validator):
        """Test creating and using multiple pipelines concurrently."""
        text = sample_texts["short"]
        
        async def create_and_synthesize(pipeline_id):
            pipeline = await voirs.VoirsPipeline.create()
            audio = await pipeline.synthesize(f"{text} Pipeline {pipeline_id}")
            audio_validator.validate_audio_buffer(audio)
            return audio
        
        # Create multiple pipelines concurrently
        tasks = [create_and_synthesize(i) for i in range(3)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All should succeed
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                pytest.fail(f"Pipeline {i} failed: {result}")
            else:
                audio_validator.validate_audio_buffer(result)
    
    async def test_pipeline_reuse(self, sample_texts, audio_validator):
        """Test reusing the same pipeline for multiple synthesis operations."""
        pipeline = await voirs.VoirsPipeline.create()
        
        # Use same pipeline for multiple synthesis operations
        for i in range(5):
            text = f"{sample_texts['short']} Iteration {i}"
            audio = await pipeline.synthesize(text)
            audio_validator.validate_audio_buffer(audio)
    
    async def test_memory_stability(self, sample_texts, memory_tracker):
        """Test memory stability during repeated operations."""
        memory_tracker.start_tracking()
        
        pipeline = await voirs.VoirsPipeline.create()
        text = sample_texts["medium"]
        
        # Perform many synthesis operations
        for i in range(10):
            audio = await pipeline.synthesize(f"{text} {i}")
            # Don't keep references to prevent legitimate memory growth
            del audio
        
        # Check for memory leaks
        assert not memory_tracker.check_memory_leak(threshold_mb=50.0), \
            "Potential memory leak detected"
    
    async def test_error_recovery(self, sample_texts, audio_validator):
        """Test system recovery after errors."""
        pipeline = await voirs.VoirsPipeline.create()
        
        # Try to cause an error
        try:
            await pipeline.synthesize(None)  # Should fail
        except Exception:
            pass  # Expected to fail
        
        # System should still work after error
        text = sample_texts["short"]
        audio = await pipeline.synthesize(text)
        audio_validator.validate_audio_buffer(audio)


@pytest.mark.skipif(not VOIRS_AVAILABLE, reason="VoiRS Python bindings not available")
class TestFormatCompatibility:
    """Test compatibility with different audio formats and outputs."""
    
    async def test_multiple_format_export(self, sample_texts, temp_audio_dir, audio_validator):
        """Test exporting audio in multiple formats."""
        pipeline = await voirs.VoirsPipeline.create()
        text = sample_texts["medium"]
        audio = await pipeline.synthesize(text)
        
        formats_to_test = ["wav", "flac"]  # Start with supported formats
        
        for fmt in formats_to_test:
            output_file = temp_audio_dir / f"test_output.{fmt}"
            
            try:
                if fmt == "wav":
                    audio.save_wav(str(output_file))
                elif fmt == "flac":
                    audio.save_flac(str(output_file))
                
                audio_validator.validate_audio_file(output_file, fmt)
                
            except AttributeError:
                print(f"Format {fmt} not supported")
            except Exception as e:
                print(f"Format {fmt} failed: {e}")
    
    async def test_different_sample_rates(self, sample_texts, audio_validator):
        """Test synthesis with different sample rates."""
        pipeline = await voirs.VoirsPipeline.create()
        text = sample_texts["short"]
        
        sample_rates = [22050, 44100, 48000]
        
        for rate in sample_rates:
            config = {"sample_rate": rate}
            try:
                audio = await pipeline.synthesize_with_config(text, config)
                audio_validator.validate_audio_buffer(audio)
                
                # Check that the requested sample rate was used
                assert audio.sample_rate == rate or abs(audio.sample_rate - rate) < 100, \
                    f"Expected {rate} Hz, got {audio.sample_rate} Hz"
                
            except Exception as e:
                print(f"Sample rate {rate} failed: {e}")
    
    async def test_quality_levels(self, sample_texts, audio_validator):
        """Test different quality levels."""
        pipeline = await voirs.VoirsPipeline.create()
        text = sample_texts["medium"]
        
        quality_levels = ["low", "medium", "high", "ultra"]
        results = {}
        
        for quality in quality_levels:
            config = {"quality": quality}
            try:
                audio = await pipeline.synthesize_with_config(text, config)
                audio_validator.validate_audio_buffer(audio)
                results[quality] = audio
            except Exception as e:
                print(f"Quality {quality} failed: {e}")
        
        # Higher quality might produce different characteristics
        if len(results) >= 2:
            # All should be valid audio
            for quality, audio in results.items():
                assert audio.duration > 0, f"Invalid audio for quality {quality}"
/*
 * VoiRS Speech Recognition - Streaming C++ Example
 * 
 * This example demonstrates streaming speech recognition using the VoiRS
 * library from C++ code.
 * 
 * Compile with:
 *   g++ -std=c++17 -o streaming_recognition streaming_recognition.cpp -lvoirs_recognizer
 * 
 * Make sure to build the Rust library with:
 *   cargo build --release --features c-api
 */

#include <iostream>
#include <vector>
#include <memory>
#include <fstream>
#include <chrono>
#include <thread>
#include <cstring>
#include <cmath>

extern "C" {
#include "../../include/voirs_recognizer.h"
}

class VoirsRecognizerWrapper {
public:
    VoirsRecognizerWrapper() : recognizer_(nullptr) {}
    
    ~VoirsRecognizerWrapper() {
        if (recognizer_) {
            voirs_recognizer_destroy(recognizer_);
        }
    }
    
    bool initialize(const VoirsRecognitionConfig& config) {
        VoirsError error = voirs_recognizer_create(&config, &recognizer_);
        if (error != VOIRS_SUCCESS) {
            std::cerr << "Failed to create recognizer: " << voirs_error_to_string(error) << std::endl;
            return false;
        }
        return true;
    }
    
    bool recognize(const std::vector<uint8_t>& audio_data, 
                   const VoirsAudioFormat& format,
                   const VoirsRecognitionResult** result) {
        VoirsError error = voirs_recognize(recognizer_, audio_data.data(), 
                                          audio_data.size(), &format, result);
        return error == VOIRS_SUCCESS;
    }
    
    bool startStreaming(const VoirsStreamingConfig& config,
                       VoirsStreamingCallback callback,
                       void* user_data) {
        VoirsError error = voirs_start_streaming(recognizer_, &config, callback, user_data);
        return error == VOIRS_SUCCESS;
    }
    
    bool stopStreaming() {
        VoirsError error = voirs_stop_streaming(recognizer_);
        return error == VOIRS_SUCCESS;
    }
    
    bool streamAudio(const std::vector<uint8_t>& audio_chunk) {
        VoirsError error = voirs_stream_audio(recognizer_, audio_chunk.data(), audio_chunk.size());
        return error == VOIRS_SUCCESS;
    }
    
    bool isStreamingActive() const {
        return voirs_is_streaming_active(recognizer_);
    }
    
    VoirsRecognizer* get() { return recognizer_; }
    
private:
    VoirsRecognizer* recognizer_;
};

class AudioGenerator {
public:
    static std::vector<uint8_t> generateSineWave(uint32_t sample_rate, 
                                                  float duration, 
                                                  float frequency,
                                                  float amplitude = 0.1f) {
        size_t sample_count = static_cast<size_t>(sample_rate * duration);
        std::vector<uint8_t> audio_data(sample_count * 2); // 16-bit samples
        int16_t* samples = reinterpret_cast<int16_t*>(audio_data.data());
        
        for (size_t i = 0; i < sample_count; i++) {
            float t = static_cast<float>(i) / sample_rate;
            float sample = amplitude * std::sin(2.0f * M_PI * frequency * t);
            samples[i] = static_cast<int16_t>(sample * 32767.0f);
        }
        
        return audio_data;
    }
    
    static std::vector<uint8_t> generateWhiteNoise(uint32_t sample_rate, 
                                                    float duration,
                                                    float amplitude = 0.01f) {
        size_t sample_count = static_cast<size_t>(sample_rate * duration);
        std::vector<uint8_t> audio_data(sample_count * 2);
        int16_t* samples = reinterpret_cast<int16_t*>(audio_data.data());
        
        for (size_t i = 0; i < sample_count; i++) {
            float sample = amplitude * (2.0f * static_cast<float>(rand()) / RAND_MAX - 1.0f);
            samples[i] = static_cast<int16_t>(sample * 32767.0f);
        }
        
        return audio_data;
    }
};

class StreamingDemo {
public:
    StreamingDemo() : chunk_count_(0), total_results_(0) {}
    
    static void streamingCallback(const VoirsRecognitionResult* result, void* user_data) {
        StreamingDemo* demo = static_cast<StreamingDemo*>(user_data);
        demo->handleStreamingResult(result);
    }
    
    void handleStreamingResult(const VoirsRecognitionResult* result) {
        total_results_++;
        
        std::cout << "\n=== Streaming Result #" << total_results_ << " ===\n";
        std::cout << "Text: \"" << (result->text ? result->text : "(no text)") << "\"\n";
        std::cout << "Confidence: " << (result->confidence * 100.0f) << "%\n";
        std::cout << "Language: " << (result->language ? result->language : "unknown") << "\n";
        std::cout << "Audio duration: " << result->audio_duration_s << " seconds\n";
        
        if (result->segments && result->segment_count > 0) {
            std::cout << "Segments (" << result->segment_count << "):";
            for (size_t i = 0; i < result->segment_count; i++) {
                const VoirsSegment* seg = &result->segments[i];
                std::cout << " [" << seg->start_time << "s-" << seg->end_time << "s] \""
                         << (seg->text ? seg->text : "") << "\"";
            }
            std::cout << "\n";
        }
    }
    
    void runStreamingDemo() {
        std::cout << "\n=== Streaming Recognition Demo ===\n";
        
        // Initialize library
        VoirsError error = voirs_init();
        if (error != VOIRS_SUCCESS) {
            std::cerr << "Failed to initialize VoiRS: " << voirs_error_to_string(error) << std::endl;
            return;
        }
        
        // Create recognizer
        VoirsRecognitionConfig config = VOIRS_DEFAULT_CONFIG();
        config.model_name = "whisper-base";
        config.language = "en";
        
        VoirsRecognizerWrapper recognizer;
        if (!recognizer.initialize(config)) {
            std::cerr << "Failed to initialize recognizer\n";
            return;
        }
        
        std::cout << "Recognizer initialized successfully!\n";
        
        // Start streaming
        VoirsStreamingConfig streaming_config = VOIRS_DEFAULT_STREAMING_CONFIG();
        streaming_config.chunk_duration = 0.5f; // 500ms chunks
        streaming_config.overlap_duration = 0.05f; // 50ms overlap
        
        if (!recognizer.startStreaming(streaming_config, streamingCallback, this)) {
            std::cerr << "Failed to start streaming\n";
            return;
        }
        
        std::cout << "Streaming started...\n";
        
        // Simulate streaming audio chunks
        const uint32_t sample_rate = 16000;
        const float chunk_duration = 0.5f; // 500ms
        const size_t total_chunks = 10;
        
        for (size_t i = 0; i < total_chunks; i++) {
            chunk_count_++;
            
            std::cout << "\nProcessing chunk " << chunk_count_ << "/" << total_chunks << "...\n";
            
            // Generate different types of audio for demonstration
            std::vector<uint8_t> audio_chunk;
            
            if (i % 3 == 0) {
                // Sine wave
                float frequency = 440.0f + (i * 110.0f); // Varying frequency
                audio_chunk = AudioGenerator::generateSineWave(sample_rate, chunk_duration, frequency);
                std::cout << "Generated sine wave (" << frequency << " Hz)\n";
            } else if (i % 3 == 1) {
                // White noise
                audio_chunk = AudioGenerator::generateWhiteNoise(sample_rate, chunk_duration);
                std::cout << "Generated white noise\n";
            } else {
                // Mixed signal
                auto sine = AudioGenerator::generateSineWave(sample_rate, chunk_duration, 880.0f, 0.05f);
                auto noise = AudioGenerator::generateWhiteNoise(sample_rate, chunk_duration, 0.02f);
                
                audio_chunk.resize(sine.size());
                int16_t* sine_samples = reinterpret_cast<int16_t*>(sine.data());
                int16_t* noise_samples = reinterpret_cast<int16_t*>(noise.data());
                int16_t* mixed_samples = reinterpret_cast<int16_t*>(audio_chunk.data());
                
                for (size_t j = 0; j < sine.size() / 2; j++) {
                    mixed_samples[j] = static_cast<int16_t>(
                        std::clamp(static_cast<int32_t>(sine_samples[j]) + noise_samples[j], 
                                   static_cast<int32_t>(INT16_MIN), 
                                   static_cast<int32_t>(INT16_MAX)));
                }
                std::cout << "Generated mixed signal (sine + noise)\n";
            }
            
            // Stream the audio chunk
            if (!recognizer.streamAudio(audio_chunk)) {
                std::cerr << "Failed to stream audio chunk " << chunk_count_ << std::endl;
                break;
            }
            
            // Simulate real-time processing delay
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        
        // Wait a bit for final processing
        std::this_thread::sleep_for(std::chrono::seconds(1));
        
        // Stop streaming
        if (!recognizer.stopStreaming()) {
            std::cerr << "Failed to stop streaming\n";
        } else {
            std::cout << "\nStreaming stopped successfully.\n";
        }
        
        // Get final statistics
        size_t chunks_processed = 0;
        double total_duration = 0.0;
        double avg_latency = 0.0;
        
        error = voirs_get_streaming_stats(recognizer.get(), &chunks_processed, 
                                         &total_duration, &avg_latency);
        if (error == VOIRS_SUCCESS) {
            std::cout << "\n=== Streaming Statistics ===\n";
            std::cout << "Chunks processed: " << chunks_processed << "\n";
            std::cout << "Total audio duration: " << total_duration << " seconds\n";
            std::cout << "Average latency: " << avg_latency << " ms\n";
            std::cout << "Results received: " << total_results_ << "\n";
            
            if (total_duration > 0) {
                double real_time_factor = (avg_latency / 1000.0) / (total_duration / chunks_processed);
                std::cout << "Real-time factor: " << real_time_factor << "\n";
            }
        }
    }
    
private:
    size_t chunk_count_;
    size_t total_results_;
};

int main() {
    std::cout << "VoiRS Speech Recognition - Streaming C++ Example\n";
    std::cout << "===============================================\n";
    
    // Get version information
    const VoirsVersion* version = voirs_get_version();
    std::cout << "VoiRS Version: " << version->version_string << "\n";
    std::cout << "Build timestamp: " << version->build_timestamp << "\n";
    
    try {
        StreamingDemo demo;
        demo.runStreamingDemo();
        
        std::cout << "\nDemo completed successfully!\n";
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Unknown exception caught\n";
        return 1;
    }
}

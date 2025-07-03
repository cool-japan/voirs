#!/usr/bin/env node

/**
 * Example usage of VoiRS Node.js bindings
 * 
 * This example demonstrates basic text-to-speech synthesis using the VoiRS library.
 */

const { VoirsPipeline, synthesizeStreaming } = require('../index.js');
const fs = require('fs');
const path = require('path');

async function basicExample() {
    console.log('🎤 VoiRS Node.js Example - Basic Synthesis');
    
    try {
        // Create a new VoiRS pipeline
        const pipeline = new VoirsPipeline({
            useGpu: false,
            numThreads: 4,
            device: 'cpu'
        });
        
        console.log('📋 Pipeline Info:', pipeline.getInfo());
        
        // List available voices
        console.log('🎭 Listing available voices...');
        const voices = await pipeline.listVoices();
        console.log(`Found ${voices.length} voices:`);
        voices.forEach(voice => {
            console.log(`  - ${voice.name} (${voice.id}) - ${voice.language} [${voice.quality}]`);
        });
        
        // Set a voice if available
        if (voices.length > 0) {
            await pipeline.setVoice(voices[0].id);
            console.log(`🎵 Set voice to: ${voices[0].name}`);
        }
        
        // Synthesize some text
        const text = "Hello from VoiRS! This is a demonstration of text-to-speech synthesis using Node.js bindings.";
        console.log(`📝 Synthesizing: "${text}"`);
        
        const audioResult = await pipeline.synthesize(text, {
            speakingRate: 1.0,
            pitchShift: 0.0,
            volumeGain: 0.0,
            enableEnhancement: true,
            outputFormat: 'wav',
            sampleRate: 22050,
            quality: 'high'
        });
        
        console.log(`🔊 Generated audio: ${audioResult.duration.toFixed(2)}s, ${audioResult.sampleRate}Hz, ${audioResult.channels} channel(s)`);
        console.log(`📊 Audio data size: ${audioResult.samples.length} bytes`);
        
        // Save to file
        const outputPath = path.join(__dirname, 'output.wav');
        fs.writeFileSync(outputPath, audioResult.samples);
        console.log(`💾 Audio saved to: ${outputPath}`);
        
    } catch (error) {
        console.error('❌ Error:', error.message);
        process.exit(1);
    }
}

async function streamingExample() {
    console.log('\n🌊 VoiRS Node.js Example - Streaming Synthesis');
    
    try {
        const pipeline = new VoirsPipeline({
            useGpu: false,
            numThreads: 4
        });
        
        const text = "This is a streaming synthesis example. You should receive audio data in real-time chunks as it's being processed.";
        console.log(`📝 Streaming synthesis: "${text}"`);
        
        let chunkCount = 0;
        let totalBytes = 0;
        
        await synthesizeStreaming(
            pipeline,
            text,
            // Chunk callback
            (chunk) => {
                chunkCount++;
                totalBytes += chunk.length;
                console.log(`📦 Received chunk ${chunkCount}: ${chunk.length} bytes`);
            },
            // Progress callback
            (progress) => {
                console.log(`📈 Progress: ${(progress * 100).toFixed(1)}%`);
            },
            // Synthesis options
            {
                quality: 'medium',
                speakingRate: 1.2
            }
        );
        
        console.log(`✅ Streaming complete! Received ${chunkCount} chunks, ${totalBytes} total bytes`);
        
    } catch (error) {
        console.error('❌ Streaming Error:', error.message);
        process.exit(1);
    }
}

async function callbackExample() {
    console.log('\n📞 VoiRS Node.js Example - Synthesis with Callbacks');
    
    try {
        const pipeline = new VoirsPipeline();
        
        const text = "This example demonstrates synthesis with progress and error callbacks for better user experience.";
        console.log(`📝 Synthesizing with callbacks: "${text}"`);
        
        const audioResult = await pipeline.synthesizeWithCallbacks(
            text,
            {
                speakingRate: 0.9,
                quality: 'high'
            },
            // Progress callback
            (progress) => {
                const percentage = (progress * 100).toFixed(1);
                process.stdout.write(`\r📈 Progress: ${percentage}%`);
            },
            // Error callback
            (error) => {
                console.error('\n❌ Synthesis Error:', error);
            }
        );
        
        console.log(`\n✅ Synthesis complete: ${audioResult.duration.toFixed(2)}s audio generated`);
        
    } catch (error) {
        console.error('❌ Callback Example Error:', error.message);
        process.exit(1);
    }
}

async function ssmlExample() {
    console.log('\n🏷️  VoiRS Node.js Example - SSML Synthesis');
    
    try {
        const pipeline = new VoirsPipeline();
        
        const ssml = `
        <speak>
            <p>This is an example of <emphasis level="strong">SSML synthesis</emphasis>.</p>
            <break time="500ms"/>
            <p>SSML allows you to control <prosody rate="slow">speaking rate</prosody>, 
            <prosody pitch="high">pitch</prosody>, and <prosody volume="loud">volume</prosody>.</p>
            <p>You can also add <break time="1s"/> pauses between words.</p>
        </speak>
        `;
        
        console.log('📝 Synthesizing SSML content...');
        
        const audioResult = await pipeline.synthesizeSsml(ssml);
        
        console.log(`✅ SSML synthesis complete: ${audioResult.duration.toFixed(2)}s audio generated`);
        
        // Save SSML result
        const outputPath = path.join(__dirname, 'ssml_output.wav');
        fs.writeFileSync(outputPath, audioResult.samples);
        console.log(`💾 SSML audio saved to: ${outputPath}`);
        
    } catch (error) {
        console.error('❌ SSML Example Error:', error.message);
        process.exit(1);
    }
}

// Run all examples
async function runAllExamples() {
    console.log('🚀 VoiRS Node.js Bindings Examples\n');
    
    await basicExample();
    await streamingExample();
    await callbackExample();
    await ssmlExample();
    
    console.log('\n🎉 All examples completed successfully!');
}

// Run examples if this file is executed directly
if (require.main === module) {
    runAllExamples().catch(error => {
        console.error('❌ Example failed:', error);
        process.exit(1);
    });
}

module.exports = {
    basicExample,
    streamingExample,
    callbackExample,
    ssmlExample,
    runAllExamples
};
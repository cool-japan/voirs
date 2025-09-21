# VoiRS WebAssembly Integration

This directory contains a complete WebAssembly integration example for VoiRS, enabling speech synthesis directly in web browsers.

## 🌐 What is this?

The VoiRS WebAssembly demo showcases how to run neural text-to-speech synthesis entirely in a web browser using WebAssembly (WASM). This enables:

- **Client-side speech synthesis** - No server required
- **Real-time audio generation** - Fast synthesis with low latency
- **Cross-platform compatibility** - Works in any modern browser
- **Privacy-focused** - All processing happens locally
- **Offline capability** - Works without internet connection (after initial load)

## 🚀 Quick Start

### Prerequisites

1. **Rust with WebAssembly support:**
   ```bash
   # Install Rust if not already installed
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   
   # Add WebAssembly target
   rustup target add wasm32-unknown-unknown
   ```

2. **wasm-pack (WebAssembly build tool):**
   ```bash
   curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh
   ```

### Build and Run

1. **Build the WebAssembly demo:**
   ```bash
   cd examples
   ./build_wasm_demo.sh
   ```

2. **Start the demo server:**
   ```bash
   cd wasm_demo_output
   ./serve.sh
   ```

3. **Open your browser** and navigate to: `http://localhost:8000`

4. **Try the demo:**
   - Enter text in the input field
   - Click "Synthesize Speech"
   - Listen to the generated audio!

## 📁 Files Overview

- **`wasm_integration_example.rs`** - Main WebAssembly bindings and synthesis logic
- **`wasm_demo.html`** - Interactive demo web page
- **`build_wasm_demo.sh`** - Build script for compiling to WebAssembly
- **`WASM_README.md`** - This documentation file

## 🛠️ Technical Details

### Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Browser   │    │   WebAssembly   │    │   VoiRS Core    │
│                 │    │    Module       │    │                 │
│ ┌─────────────┐ │    │ ┌─────────────┐ │    │ ┌─────────────┐ │
│ │   HTML/JS   │ │◄──►│ │ WASM Wrapper│ │◄──►│ │ G2P + TTS   │ │
│ │   Web UI    │ │    │ │  Functions  │ │    │ │  Pipeline   │ │
│ └─────────────┘ │    │ └─────────────┘ │    │ └─────────────┘ │
│                 │    │                 │    │                 │
│ ┌─────────────┐ │    │ ┌─────────────┐ │    │ ┌─────────────┐ │
│ │ Web Audio   │ │    │ │ Audio Data  │ │    │ │ Audio       │ │
│ │   API       │ │◄───│ │ Conversion  │ │◄───│ │ Generation  │ │
│ └─────────────┘ │    │ └─────────────┘ │    │ └─────────────┘ │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Components

1. **WebAssembly Module** (`WasmVoirsSynthesizer`)
   - Rust-compiled WASM binary
   - Exposes text-to-speech API to JavaScript
   - Handles audio data conversion

2. **JavaScript Bindings**
   - Auto-generated from `wasm-bindgen`
   - Provides Promise-based async API
   - Integrates with Web Audio API

3. **Web Interface**
   - HTML5/CSS3 responsive design
   - Real-time performance metrics
   - Audio playback and download

### Performance Optimization

- **Lightweight Models**: Uses optimized neural models for web deployment
- **Memory Management**: Efficient memory usage with garbage collection
- **Streaming Processing**: Chunk-based processing for responsive UI
- **Caching**: Intelligent caching of compiled models

## 🎯 Features

### Core Functionality
- ✅ **Real-time text-to-speech synthesis**
- ✅ **Multiple voice examples and presets**
- ✅ **Audio playback with Web Audio API**
- ✅ **WAV file download capability**
- ✅ **Performance metrics and monitoring**

### Technical Features
- ✅ **WebAssembly compilation from Rust**
- ✅ **Browser-compatible audio processing**
- ✅ **Responsive web interface**
- ✅ **Error handling and user feedback**
- ✅ **Cross-browser compatibility**

### Demo Features
- ✅ **Interactive text input**
- ✅ **Example text suggestions**
- ✅ **Real-time synthesis metrics**
- ✅ **Audio visualization**
- ✅ **Performance monitoring**

## 🔧 Development

### Building for Different Modes

```bash
# Development build (faster compilation, larger size)
./build_wasm_demo.sh --dev

# Release build (optimized, smaller size)
./build_wasm_demo.sh --release
```

### Customizing the Demo

1. **Modify the UI** - Edit `wasm_demo.html`
2. **Add features** - Extend `wasm_integration_example.rs`
3. **Change models** - Update model configuration in the Rust code
4. **Styling** - Customize CSS in the HTML file

### Integration in Your Project

```javascript
// Import the generated WebAssembly module
import init, { WasmVoirsSynthesizer } from './pkg/voirs_examples.js';

// Initialize WebAssembly
await init();

// Create synthesizer
const synthesizer = await new WasmVoirsSynthesizer();

// Synthesize speech
const audioData = await synthesizer.synthesize('Hello, WebAssembly!');

// Create Web Audio buffer
const audioContext = new AudioContext();
const audioBuffer = await synthesizer.create_audio_buffer(text, audioContext);

// Play audio
const source = audioContext.createBufferSource();
source.buffer = audioBuffer;
source.connect(audioContext.destination);
source.start();
```

## 🐛 Troubleshooting

### Common Issues

1. **CORS Errors**
   - **Problem**: Opening HTML file directly in browser
   - **Solution**: Use HTTP server (provided scripts)

2. **WebAssembly Load Errors**
   - **Problem**: WASM module not found or corrupted
   - **Solution**: Rebuild with `./build_wasm_demo.sh`

3. **Audio Not Playing**
   - **Problem**: Browser audio permissions or Web Audio API issues
   - **Solution**: Check browser permissions, try different browser

4. **Large File Sizes**
   - **Problem**: WASM binary too large for deployment
   - **Solution**: Use `--release` build mode, enable compression

### Browser Compatibility

| Browser | Version | WebAssembly | Web Audio | Status |
|---------|---------|-------------|-----------|---------|
| Chrome  | 57+     | ✅          | ✅        | ✅ Full Support |
| Firefox | 52+     | ✅          | ✅        | ✅ Full Support |
| Safari  | 11+     | ✅          | ✅        | ✅ Full Support |
| Edge    | 16+     | ✅          | ✅        | ✅ Full Support |

### Performance Tips

1. **Use Release Builds** for production deployment
2. **Enable Compression** (gzip/brotli) on your web server
3. **Cache WASM Files** with appropriate HTTP headers
4. **Preload Models** for faster synthesis times

## 📚 Additional Resources

- [WebAssembly Official Site](https://webassembly.org/)
- [wasm-bindgen Book](https://rustwasm.github.io/wasm-bindgen/)
- [Web Audio API Documentation](https://developer.mozilla.org/en-US/docs/Web/API/Web_Audio_API)
- [VoiRS Documentation](../README.md)

## 🤝 Contributing

Contributions to improve the WebAssembly integration are welcome! Areas for enhancement:

- **Performance optimizations**
- **Additional voice models**
- **UI/UX improvements**
- **Mobile browser support**
- **Advanced audio features**

## 📄 License

This WebAssembly integration follows the same license as the main VoiRS project.

---

**Happy synthesizing in the browser! 🎵🌐**
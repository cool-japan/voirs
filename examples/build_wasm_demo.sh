#!/bin/bash
#
# VoiRS WebAssembly Demo Build Script
#
# This script builds the VoiRS WebAssembly demo and sets up the necessary files
# for running the demo in a web browser.
#
# Prerequisites:
# - Rust with wasm32-unknown-unknown target installed
# - wasm-pack installed (curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh)
#
# Usage:
# ./build_wasm_demo.sh [--dev|--release]

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
EXAMPLES_DIR="$SCRIPT_DIR"
BUILD_MODE="${1:-release}"
OUTPUT_DIR="$EXAMPLES_DIR/wasm_demo_output"

echo "🌐 VoiRS WebAssembly Demo Builder"
echo "================================="
echo "Script directory: $SCRIPT_DIR"
echo "Project root: $PROJECT_ROOT"
echo "Build mode: $BUILD_MODE"
echo "Output directory: $OUTPUT_DIR"
echo

# Check prerequisites
check_prerequisites() {
    echo "🔍 Checking prerequisites..."
    
    # Check if Rust is installed
    if ! command -v rustc &> /dev/null; then
        echo "❌ Error: Rust is not installed. Please install Rust from https://rustup.rs/"
        exit 1
    fi
    
    # Check if wasm32-unknown-unknown target is installed
    if ! rustup target list --installed | grep -q wasm32-unknown-unknown; then
        echo "📦 Installing wasm32-unknown-unknown target..."
        rustup target add wasm32-unknown-unknown
    fi
    
    # Check if wasm-pack is installed
    if ! command -v wasm-pack &> /dev/null; then
        echo "❌ Error: wasm-pack is not installed."
        echo "   Install with: curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh"
        exit 1
    fi
    
    echo "✅ Prerequisites check completed"
    echo
}

# Build the WebAssembly module
build_wasm() {
    echo "🔨 Building WebAssembly module..."
    
    cd "$EXAMPLES_DIR"
    
    # Determine build flags
    local build_flags=""
    if [ "$BUILD_MODE" = "dev" ] || [ "$BUILD_MODE" = "debug" ]; then
        build_flags="--dev"
        echo "   Building in development mode (larger size, faster compilation)"
    else
        build_flags="--release"
        echo "   Building in release mode (optimized for size and performance)"
    fi
    
    # Build with wasm-pack
    echo "   Running: wasm-pack build --target web --out-dir wasm_pkg $build_flags"
    wasm-pack build --target web --out-dir wasm_pkg $build_flags
    
    if [ $? -eq 0 ]; then
        echo "✅ WebAssembly build completed successfully"
    else
        echo "❌ WebAssembly build failed"
        exit 1
    fi
    
    echo
}

# Set up demo files
setup_demo() {
    echo "📁 Setting up demo files..."
    
    # Create output directory
    mkdir -p "$OUTPUT_DIR"
    
    # Copy WASM package files
    if [ -d "$EXAMPLES_DIR/wasm_pkg" ]; then
        echo "   Copying WebAssembly package files..."
        cp -r "$EXAMPLES_DIR/wasm_pkg"/* "$OUTPUT_DIR/"
    else
        echo "❌ Error: WebAssembly package directory not found"
        exit 1
    fi
    
    # Copy HTML demo file
    if [ -f "$EXAMPLES_DIR/wasm_demo.html" ]; then
        echo "   Copying HTML demo file..."
        cp "$EXAMPLES_DIR/wasm_demo.html" "$OUTPUT_DIR/index.html"
    else
        echo "❌ Error: HTML demo file not found"
        exit 1
    fi
    
    # Create a simple HTTP server script
    cat > "$OUTPUT_DIR/serve.py" << 'EOF'
#!/usr/bin/env python3
"""
Simple HTTP server for serving the VoiRS WebAssembly demo.
This server sets the correct MIME types for WebAssembly files.
"""
import http.server
import socketserver
import sys
from pathlib import Path

class WasmHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cross-Origin-Embedder-Policy', 'require-corp')
        self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
        super().end_headers()
    
    def guess_type(self, path):
        mime_type, encoding = super().guess_type(path)
        if path.endswith('.wasm'):
            return 'application/wasm'
        return mime_type, encoding

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    
    print(f"🌐 Starting VoiRS WebAssembly Demo Server")
    print(f"📍 Server running at: http://localhost:{port}")
    print(f"📁 Serving from: {Path.cwd()}")
    print(f"🚀 Open your browser and navigate to the URL above")
    print(f"⏹️  Press Ctrl+C to stop the server")
    print()
    
    with socketserver.TCPServer(("", port), WasmHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n🛑 Server stopped")
EOF
    chmod +x "$OUTPUT_DIR/serve.py"
    
    # Create a shell script for serving
    cat > "$OUTPUT_DIR/serve.sh" << 'EOF'
#!/bin/bash
# Simple HTTP server for the VoiRS WebAssembly demo
PORT=${1:-8000}
echo "🌐 Starting VoiRS WebAssembly Demo Server on port $PORT"
echo "📍 Open your browser at: http://localhost:$PORT"
echo "⏹️  Press Ctrl+C to stop"
echo
if command -v python3 &> /dev/null; then
    python3 serve.py $PORT
elif command -v python &> /dev/null; then
    python serve.py $PORT
else
    echo "❌ Python not found. Please install Python to run the server."
    echo "   Alternative: Use any HTTP server to serve this directory"
    exit 1
fi
EOF
    chmod +x "$OUTPUT_DIR/serve.sh"
    
    # Create README for the demo
    cat > "$OUTPUT_DIR/README.md" << 'EOF'
# VoiRS WebAssembly Demo

This directory contains the compiled VoiRS WebAssembly demo.

## Quick Start

1. **Start the demo server:**
   ```bash
   ./serve.sh
   ```
   Or with Python directly:
   ```bash
   python3 serve.py
   ```

2. **Open your browser** and navigate to: `http://localhost:8000`

3. **Try the demo:**
   - Enter text in the input field
   - Click "Synthesize Speech" 
   - Listen to the generated audio!

## Files

- `index.html` - Main demo page
- `voirs_examples.js` - JavaScript bindings for WebAssembly
- `voirs_examples_bg.wasm` - Compiled WebAssembly module
- `serve.py` / `serve.sh` - HTTP server scripts
- `README.md` - This file

## Requirements

- Modern web browser with WebAssembly support
- HTTP server (Python scripts provided)

## Troubleshooting

- **CORS errors:** Make sure you're serving the files via HTTP server, not opening `index.html` directly
- **Audio not playing:** Check browser permissions for audio playback
- **WebAssembly errors:** Ensure your browser supports WebAssembly (most modern browsers do)

## Technical Details

- **Runtime:** WebAssembly compiled from Rust
- **Audio:** Web Audio API
- **Models:** Optimized neural TTS models
- **Performance:** Real-time synthesis capability

Enjoy experimenting with VoiRS in your browser! 🎵
EOF
    
    echo "✅ Demo files setup completed"
    echo
}

# Show file sizes and information
show_info() {
    echo "📊 Build Information:"
    echo "--------------------"
    
    if [ -f "$OUTPUT_DIR/voirs_examples_bg.wasm" ]; then
        local wasm_size=$(du -h "$OUTPUT_DIR/voirs_examples_bg.wasm" | cut -f1)
        echo "   WebAssembly module size: $wasm_size"
    fi
    
    echo "   Build mode: $BUILD_MODE"
    echo "   Output directory: $OUTPUT_DIR"
    echo "   Files generated:"
    ls -la "$OUTPUT_DIR" | grep -E "\.(wasm|js|html)$" | awk '{printf "     - %s (%s)\n", $9, $5}'
    echo
}

# Main execution
main() {
    echo "Starting VoiRS WebAssembly demo build process..."
    echo
    
    check_prerequisites
    build_wasm
    setup_demo
    show_info
    
    echo "🎉 VoiRS WebAssembly Demo Build Complete!"
    echo "========================================="
    echo
    echo "📁 Demo files are ready in: $OUTPUT_DIR"
    echo
    echo "🚀 To run the demo:"
    echo "   cd $OUTPUT_DIR"
    echo "   ./serve.sh"
    echo "   # Then open http://localhost:8000 in your browser"
    echo
    echo "🌐 The demo includes:"
    echo "   ✅ Interactive text-to-speech interface"
    echo "   ✅ Real-time audio generation"
    echo "   ✅ Performance metrics"
    echo "   ✅ Audio download capability"
    echo
    echo "Happy synthesizing! 🎵"
}

# Handle script arguments
case "${1:-}" in
    -h|--help)
        echo "VoiRS WebAssembly Demo Builder"
        echo
        echo "Usage: $0 [--dev|--release|--help]"
        echo
        echo "Options:"
        echo "  --dev, --debug    Build in development mode (faster compilation)"
        echo "  --release         Build in release mode (optimized, default)"
        echo "  --help, -h        Show this help message"
        echo
        exit 0
        ;;
    --dev|--debug)
        BUILD_MODE="dev"
        main
        ;;
    --release|"")
        BUILD_MODE="release"
        main
        ;;
    *)
        echo "❌ Unknown option: $1"
        echo "   Use --help for usage information"
        exit 1
        ;;
esac
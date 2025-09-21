#!/bin/bash
set -e

echo "Building VoiRS Recognizer WASM module..."

# Check for wasm-pack
if ! command -v wasm-pack &> /dev/null; then
    echo "wasm-pack is required. Install with: cargo install wasm-pack"
    exit 1
fi

# Create output directory
mkdir -p pkg

# Build for web (browser)
echo "Building for web/browser..."
wasm-pack build --target web --out-dir pkg/web --features wasm --no-default-features -- --features "whisper,analysis,wasm"

# Build for Node.js
echo "Building for Node.js..."
wasm-pack build --target nodejs --out-dir pkg/nodejs --features wasm --no-default-features -- --features "whisper,analysis,wasm"

# Build for bundlers (webpack, etc.)
echo "Building for bundlers..."
wasm-pack build --target bundler --out-dir pkg/bundler --features wasm --no-default-features -- --features "whisper,analysis,wasm"

# Build for no-modules (legacy browsers)
echo "Building for no-modules..."
wasm-pack build --target no-modules --out-dir pkg/no-modules --features wasm --no-default-features -- --features "whisper,analysis,wasm"

echo "WASM builds completed!"
echo "Output directories:"
echo "  - pkg/web/ - For modern browsers with ES modules"
echo "  - pkg/nodejs/ - For Node.js applications"
echo "  - pkg/bundler/ - For webpack/rollup/other bundlers"
echo "  - pkg/no-modules/ - For legacy browsers without module support"

# Copy additional files
if [ -f "README_WASM.md" ]; then
    cp README_WASM.md pkg/web/
    cp README_WASM.md pkg/nodejs/
    cp README_WASM.md pkg/bundler/
    cp README_WASM.md pkg/no-modules/
fi

echo "WASM module build complete!"

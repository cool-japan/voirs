# VoiRS Python Documentation

This directory contains comprehensive documentation for the VoiRS Python bindings.

## Documentation Structure

### Core References
- [**API Reference**](api_reference.md) - Complete class and function documentation
- [**Class Reference**](class_reference.md) - Detailed class documentation with examples
- [**Quick Start Guide**](quick_start.md) - Get started quickly with VoiRS

### Tutorials and Guides
- [**Tutorial Notebooks**](tutorials/) - Interactive Jupyter notebooks
- [**Performance Guide**](performance_guide.md) - Optimization and performance tips
- [**Integration Examples**](integration_examples.md) - Real-world integration scenarios

### Advanced Topics
- [**Configuration Guide**](configuration.md) - Advanced configuration options
- [**Error Handling**](error_handling.md) - Exception handling and debugging
- [**Memory Management**](memory_management.md) - Best practices for memory usage

## Getting Started

```python
# Quick example
from voirs_ffi import VoirsPipeline

# Create pipeline
pipeline = VoirsPipeline()

# Synthesize text
audio = pipeline.synthesize("Hello, world!")

# Save to file
audio.save("output.wav")
```

## Installation

```bash
pip install voirs-ffi
```

## Requirements

- Python 3.8+
- NumPy (optional, for advanced audio processing)
- CUDA (optional, for GPU acceleration)

## Support

For issues and questions, please visit:
- [GitHub Issues](https://github.com/cool-japan/voirs/issues)
- [API Documentation](https://docs.voirs.dev/python/)
- [Community Forum](https://forum.voirs.dev/)
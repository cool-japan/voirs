# VoiRS Python Tutorials

This directory contains interactive Jupyter notebooks that demonstrate various aspects of using VoiRS Python bindings.

## Available Tutorials

### 1. [Basic Usage](01_basic_usage.ipynb)
**Level: Beginner**

Learn the fundamentals of VoiRS Python bindings:
- Installation and setup
- Basic text-to-speech synthesis
- Working with voices
- Audio processing basics
- Configuration options
- Error handling
- Performance optimization

**Prerequisites:**
- Python 3.8+
- VoiRS Python bindings installed
- Jupyter notebook environment

### 2. [Advanced Features](02_advanced_features.ipynb) *(Coming Soon)*
**Level: Intermediate**

Explore advanced VoiRS features:
- SSML markup support
- Streaming synthesis
- Custom voice training
- Audio effects and filters
- Real-time processing
- Integration with audio libraries

### 3. [Performance Optimization](03_performance_optimization.ipynb) *(Coming Soon)*
**Level: Advanced**

Deep dive into performance optimization:
- GPU acceleration setup
- Memory management
- Batch processing
- Caching strategies
- Profiling and benchmarking
- Production deployment tips

### 4. [Web Integration](04_web_integration.ipynb) *(Coming Soon)*
**Level: Intermediate**

Build web applications with VoiRS:
- Flask/FastAPI integration
- RESTful API development
- WebSocket streaming
- JavaScript client examples
- Authentication and rate limiting

### 5. [Machine Learning Integration](05_ml_integration.ipynb) *(Coming Soon)*
**Level: Advanced**

Integrate VoiRS with ML workflows:
- Data pipeline integration
- Feature extraction from audio
- Quality prediction models
- Automated voice selection
- A/B testing frameworks

## Running the Tutorials

### Prerequisites

```bash
# Install VoiRS Python bindings
pip install voirs-ffi

# Install notebook dependencies
pip install jupyter numpy matplotlib pandas
```

### Starting Jupyter

```bash
# Navigate to tutorials directory
cd docs/python/tutorials

# Start Jupyter notebook
jupyter notebook
```

### Using in Google Colab

You can also run these tutorials in Google Colab:

1. Upload the notebook file to Google Drive
2. Open with Google Colab
3. Install VoiRS in the first cell:
   ```python
   !pip install voirs-ffi
   ```

## Tutorial Structure

Each tutorial follows a consistent structure:

1. **Introduction**: Overview and learning objectives
2. **Prerequisites**: Required knowledge and setup
3. **Code Examples**: Interactive code with explanations
4. **Exercises**: Hands-on practice opportunities
5. **Summary**: Key takeaways and next steps
6. **Resources**: Links to additional documentation

## Common Issues and Solutions

### Audio Playback Issues

If you encounter audio playback issues in notebooks:

```python
# Install additional audio dependencies
!pip install sounddevice
!pip install IPython

# Use IPython audio widget
from IPython.display import Audio
Audio(audio.samples, rate=audio.sample_rate)
```

### GPU Not Detected

If GPU acceleration isn't working:

```python
# Check CUDA availability
from voirs_ffi import check_compatibility
info = check_compatibility()
print(f"GPU available: {info['gpu']}")

# Install CUDA toolkit if needed
# Follow instructions at: https://docs.voirs.dev/installation/gpu
```

### Memory Issues

For large datasets or long sessions:

```python
# Use memory-efficient settings
pipeline = VoirsPipeline.with_config(
    use_gpu=True,
    num_threads=2,
    quality="medium"
)

# Clean up after processing
import gc
del audio
gc.collect()
```

## Contributing

We welcome contributions to the tutorial collection! To add a new tutorial:

1. Create a new notebook following the naming convention: `##_tutorial_name.ipynb`
2. Follow the standard tutorial structure
3. Include comprehensive code examples and explanations
4. Add appropriate error handling and fallbacks
5. Update this README with tutorial information
6. Submit a pull request

### Tutorial Guidelines

- **Code Quality**: Use clear, well-commented code
- **Error Handling**: Include proper error handling for all examples
- **Accessibility**: Ensure tutorials work in different environments
- **Performance**: Include performance considerations and tips
- **Documentation**: Add comprehensive docstrings and markdown explanations

## Support

If you encounter issues with the tutorials:

1. Check the [Common Issues](#common-issues-and-solutions) section
2. Review the [VoiRS Documentation](https://docs.voirs.dev/)
3. Visit the [GitHub Issues](https://github.com/cool-japan/voirs/issues) page
4. Join our [Community Forum](https://forum.voirs.dev/)

## License

These tutorials are provided under the MIT License, same as the VoiRS project.
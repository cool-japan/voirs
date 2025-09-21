# VoiRS Evaluation Python Bindings

Python bindings for the VoiRS speech evaluation system, providing seamless integration with Python scientific computing tools including NumPy, SciPy, Pandas, and Matplotlib.

## 🚀 Quick Start

### Installation

```bash
# Install from PyPI (when available)
pip install voirs-evaluation

# Or install from source
git clone https://github.com/VoiRS/voirs-evaluation
cd voirs-evaluation
pip install -r requirements.txt
pip install .
```

### Basic Usage

```python
import numpy as np
import voirs_evaluation as ve

# Load audio data as NumPy arrays
reference = np.random.randn(16000).astype(np.float32)
degraded = np.random.randn(16000).astype(np.float32)

# Create evaluator
evaluator = ve.PyQualityEvaluator()

# Evaluate quality
result = evaluator.evaluate(reference, degraded, sample_rate=16000)
print(f"PESQ Score: {result.pesq}")
print(f"STOI Score: {result.stoi}")
print(f"MCD Score: {result.mcd}")
```

## 📚 Features

### Core Evaluation Capabilities

- **Quality Metrics**: PESQ, STOI, MCD, MSD with full ITU-T compliance
- **Statistical Analysis**: Paired t-tests, correlation analysis, confidence intervals
- **Pronunciation Assessment**: Phoneme-level accuracy, fluency, prosody evaluation
- **Batch Processing**: Efficient evaluation of multiple audio samples
- **Real-time Monitoring**: Streaming evaluation capabilities

### Scientific Computing Integration

- **NumPy Arrays**: Direct support for NumPy arrays as audio input/output
- **Pandas DataFrames**: Results formatted for easy DataFrame integration
- **SciPy Statistics**: Compatible with SciPy statistical functions
- **Matplotlib**: Ready-to-plot results for visualization

## 🧪 Examples

### Quality Evaluation with Statistical Analysis

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import voirs_evaluation as ve

# Generate test data
sample_rate = 16000
reference = ve.create_sine_wave(440, 1.0, sample_rate)
degraded = ve.add_noise(reference, 0.1)

# Evaluate quality
evaluator = ve.PyQualityEvaluator()
result = evaluator.evaluate(reference, degraded, sample_rate)

# Convert to DataFrame
df = pd.DataFrame([result.to_dict()])
print(df)

# Statistical analysis
analyzer = ve.PyStatisticalAnalyzer()
stats = analyzer.descriptive_stats(degraded.tolist())
print(f"SNR: {stats['mean']:.2f} dB")
```

### Batch Evaluation with Pandas

```python
import numpy as np
import pandas as pd
import voirs_evaluation as ve

# Create test dataset
evaluator = ve.PyQualityEvaluator()
results = []

for snr in [10, 20, 30, 40]:
    reference = ve.create_sine_wave(440, 1.0, 16000)
    degraded = ve.add_noise(reference, 1.0 / (10**(snr/20)))
    
    result = evaluator.evaluate(reference, degraded, 16000)
    result_dict = result.to_dict()
    result_dict['snr_target'] = snr
    results.append(result_dict)

# Create DataFrame
df = pd.DataFrame(results)
print(df[['snr_target', 'pesq', 'stoi', 'overall_score']])

# Plot results
df.plot(x='snr_target', y=['pesq', 'stoi'], kind='line')
plt.xlabel('Target SNR (dB)')
plt.ylabel('Quality Score')
plt.title('Quality vs SNR')
plt.show()
```

### Real-time Monitoring

```python
import numpy as np
import voirs_evaluation as ve

# Simulate real-time audio chunks
chunk_size = 1600  # 100ms at 16kHz
total_chunks = 50

quality_history = []

for i in range(total_chunks):
    # Generate audio chunk
    chunk = ve.create_sine_wave(440 + i*5, 0.1, 16000)
    
    # Add varying noise
    noisy_chunk = ve.add_noise(chunk, 0.1 * (1 + 0.5 * np.sin(i * 0.2)))
    
    # Calculate SNR
    snr = ve.calculate_snr(chunk, noisy_chunk - chunk)
    quality_history.append(snr)
    
    print(f"Chunk {i}: SNR = {snr:.2f} dB")

print(f"Average SNR: {np.mean(quality_history):.2f} dB")
```

### Statistical Testing

```python
import voirs_evaluation as ve

# Compare two systems
system_a_scores = [3.2, 3.5, 3.1, 3.8, 3.4]
system_b_scores = [2.8, 3.0, 2.9, 3.2, 3.1]

analyzer = ve.PyStatisticalAnalyzer()
test_result = analyzer.paired_t_test(system_a_scores, system_b_scores)

print(f"T-statistic: {test_result.statistic:.3f}")
print(f"P-value: {test_result.p_value:.3f}")
print(f"Effect size: {test_result.effect_size:.3f}")
print(f"Significant: {test_result.is_significant()}")
```

### Pronunciation Evaluation

```python
import numpy as np
import voirs_evaluation as ve

# Generate test speech
audio = ve.create_sine_wave(200, 1.0, 16000)  # Simulate speech
reference_text = "Hello world"

# Evaluate pronunciation
evaluator = ve.PyPronunciationEvaluator()
result = evaluator.evaluate(audio, reference_text, 16000)

print(f"Overall Score: {result.overall_score:.3f}")
print(f"Phoneme Accuracy: {result.phoneme_accuracy:.3f}")
print(f"Fluency: {result.fluency_score:.3f}")
print(f"Prosody: {result.prosody_score:.3f}")
```

## 🔧 API Reference

### Classes

#### `PyQualityEvaluator`

Main class for audio quality evaluation.

**Methods:**
- `__init__()`: Create new evaluator
- `evaluate(reference, degraded, sample_rate)`: Evaluate quality metrics
- `evaluate_no_reference(audio, sample_rate)`: No-reference evaluation
- `evaluate_batch(references, degraded_list, sample_rate)`: Batch evaluation

#### `PyStatisticalAnalyzer`

Statistical analysis and testing.

**Methods:**
- `__init__()`: Create new analyzer
- `paired_t_test(group1, group2)`: Perform paired t-test
- `correlation_test(x, y)`: Correlation analysis
- `descriptive_stats(data)`: Calculate descriptive statistics

#### `PyPronunciationEvaluator`

Pronunciation assessment.

**Methods:**
- `__init__()`: Create new evaluator
- `evaluate(audio, reference_text, sample_rate)`: Evaluate pronunciation

### Result Classes

#### `PyQualityResult`

Quality evaluation results.

**Attributes:**
- `pesq`: PESQ score (1.0-4.5)
- `stoi`: STOI score (0.0-1.0)
- `mcd`: MCD score (lower is better)
- `msd`: MSD score (lower is better)
- `overall_score`: Overall quality (0.0-1.0)
- `confidence`: Confidence score (0.0-1.0)

**Methods:**
- `to_dict()`: Convert to dictionary for DataFrame integration

#### `PyStatisticalResult`

Statistical test results.

**Attributes:**
- `statistic`: Test statistic value
- `p_value`: P-value
- `effect_size`: Effect size measure
- `ci_lower`, `ci_upper`: Confidence interval bounds
- `degrees_of_freedom`: Degrees of freedom

**Methods:**
- `is_significant()`: Check if p < 0.05
- `to_dict()`: Convert to dictionary

### Utility Functions

- `create_sine_wave(frequency, duration, sample_rate)`: Generate test sine wave
- `add_noise(audio, noise_level)`: Add white noise to signal
- `calculate_snr(signal, noise)`: Calculate SNR in dB

## 🧑‍💻 Development

### Building from Source

```bash
# Install Rust (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Clone repository
git clone https://github.com/VoiRS/voirs-evaluation
cd voirs-evaluation

# Install Python dependencies
pip install -r requirements.txt

# Build and install
pip install .

# Or for development
pip install -e .
```

### Running Tests

```bash
# Run Rust tests
cargo test --features python

# Run Python tests
pytest tests/

# Run example
python examples/python_integration.py
```

### Development Dependencies

```bash
pip install -r requirements.txt
pip install pytest pytest-benchmark black flake8 mypy
```

## 🔬 Scientific Computing Integration

### NumPy Integration

The Python bindings accept NumPy arrays directly:

```python
import numpy as np
import voirs_evaluation as ve

# Direct NumPy array support
audio = np.random.randn(16000).astype(np.float32)
evaluator = ve.PyQualityEvaluator()
result = evaluator.evaluate_no_reference(audio, 16000)
```

### SciPy Compatibility

Results are compatible with SciPy statistical functions:

```python
import scipy.stats as stats
import voirs_evaluation as ve

# Get evaluation results
scores = [result.overall_score for result in batch_results]

# Use with SciPy
normality = stats.shapiro(scores)
print(f"Normality test p-value: {normality.pvalue}")
```

### Pandas DataFrames

Results can be easily converted to Pandas DataFrames:

```python
import pandas as pd
import voirs_evaluation as ve

results = []
for condition in test_conditions:
    result = evaluator.evaluate(condition.reference, condition.degraded, 16000)
    result_dict = result.to_dict()
    result_dict['condition'] = condition.name
    results.append(result_dict)

df = pd.DataFrame(results)
df.to_csv('evaluation_results.csv')
```

### Matplotlib Visualization

Results are ready for plotting:

```python
import matplotlib.pyplot as plt
import voirs_evaluation as ve

# Plot quality vs SNR
plt.scatter(snr_values, quality_scores)
plt.xlabel('SNR (dB)')
plt.ylabel('Quality Score')
plt.title('Quality vs SNR Relationship')
plt.show()
```

## 📊 Performance

The Python bindings provide efficient access to the Rust evaluation core:

- **Zero-copy NumPy integration**: Direct memory access without copying
- **Batch processing**: Optimized for evaluating multiple samples
- **Multi-threading**: Automatic parallelization for batch operations
- **Memory efficient**: Minimal Python overhead

## 🤝 Contributing

Contributions are welcome! Please see the main repository for contribution guidelines.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔗 Links

- [Main VoiRS Repository](https://github.com/VoiRS/voirs)
- [Documentation](https://docs.voirs.ai)
- [Python Package](https://pypi.org/project/voirs-evaluation/)
- [Issue Tracker](https://github.com/VoiRS/voirs-evaluation/issues)
# VoiRS Singing Examples

This directory contains comprehensive examples demonstrating the Version 3.0.0 research-grade features of the VoiRS Singing synthesis system.

## Available Examples

### 1. Adaptive Learning Demo
**File**: `adaptive_learning_demo.rs`

Demonstrates the adaptive learning system that continuously improves synthesis quality through user feedback.

**Run**:
```bash
cargo run --example adaptive_learning_demo
```

**Features Demonstrated**:
- User feedback collection and processing
- Preference learning from ratings
- Quality metric fine-tuning
- Style adaptation from examples
- Personalized recommendations
- Learning statistics and improvement tracking

**Expected Output**:
- System initialization
- Feedback collection (8 samples)
- Learning statistics
- User preferences with confidence scores
- Personalized recommendations
- Style adaptation results
- Model improvement history

---

### 2. Neural Architecture Search Demo
**File**: `neural_architecture_search_demo.rs`

Demonstrates automatic neural architecture optimization using evolutionary algorithms.

**Run**:
```bash
cargo run --example neural_architecture_search_demo
```

**Features Demonstrated**:
- Neural Architecture Search (NAS) with evolutionary algorithms
- Population initialization and evolution
- Model compression with pruning and quantization
- Hardware-specific optimization (CPU, GPU, Mobile)
- Energy-efficient inference optimization

**Expected Output**:
- NAS configuration
- Architecture search progress
- Top 5 optimal architectures
- Model compression results
- Hardware optimization comparisons
- Energy efficiency metrics

---

### 3. Research Models Demo
**File**: `research_models_demo.rs`

Demonstrates state-of-the-art research models integrated into VoiRS.

**Run**:
```bash
cargo run --example research_models_demo
```

**Features Demonstrated**:
- **Diffusion Transformers**: Multi-step generation with noise schedules
- **Neural Codec Language Models**: Discrete token-based synthesis
- **Flow-Matching Synthesis**: ODE-based continuous generation
- **Score-Based Models**: Probabilistic synthesis with Langevin dynamics
- **Consistency Models**: Single-step fast generation

**Expected Output**:
- Configuration for each model
- Generation time comparisons
- Model characteristics
- Performance benchmarks
- Feature comparison table

---

### 4. Next Generation Features (Legacy)
**File**: `next_gen_features.rs`

Demonstrates Version 2.0.0 features including LLM understanding, cloud deployment, and composition assistance.

**Run**:
```bash
cargo run --example next_gen_features
```

---

## Running All Examples

To build and verify all examples compile correctly:

```bash
cargo build --examples
```

To run a specific example with release optimizations:

```bash
cargo run --example adaptive_learning_demo --release
```

## Requirements

- Rust 1.75+ with async support
- Tokio runtime (included in dependencies)
- Approximately 2-5 seconds execution time per example

## Integration

These examples can be used as templates for integrating VoiRS Singing into your applications. Key patterns demonstrated:

1. **System Initialization**: Creating and configuring synthesis systems
2. **Async Operations**: Using tokio for asynchronous synthesis
3. **Error Handling**: Proper Result type usage
4. **Configuration**: Customizing behavior with config structs
5. **API Usage**: Calling key methods and interpreting results

## Further Documentation

- **API Documentation**: `cargo doc --no-deps --open`
- **Benchmarks**: See `../benches/README.md`
- **Main Documentation**: See `../README.md`
- **TODO**: See `../TODO.md` for implementation details

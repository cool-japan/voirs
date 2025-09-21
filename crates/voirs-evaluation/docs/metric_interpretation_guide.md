# Metric Interpretation Guide

## Overview

This guide explains how to interpret the various evaluation metrics provided by the VoiRS evaluation framework. Understanding these metrics is crucial for making informed decisions about speech synthesis quality and system performance.

## Quality Metrics

### Mean Opinion Score (MOS)

**Range**: 1.0 - 5.0  
**Higher is better**

The MOS is designed to predict human subjective ratings of speech quality.

#### Interpretation:
- **5.0 - 4.5**: Excellent - Indistinguishable from natural speech
- **4.4 - 4.0**: Good - High quality with minor artifacts
- **3.9 - 3.5**: Fair - Acceptable quality for most applications
- **3.4 - 3.0**: Poor - Noticeable quality degradation
- **2.9 - 1.0**: Very Poor - Significant artifacts, limited usability

#### Typical Values:
- **High-end TTS systems**: 4.2 - 4.6
- **Commercial TTS**: 3.8 - 4.2
- **Research systems**: 3.5 - 4.0
- **Basic/old systems**: 2.5 - 3.5

#### Use Cases:
- Overall system quality assessment
- User satisfaction prediction
- System comparison and ranking

### PESQ (Perceptual Evaluation of Speech Quality)

**Range**: -0.5 - 4.5 (practical range: 1.0 - 4.5)  
**Higher is better**

PESQ is an ITU-T standard that correlates well with human perception of speech quality.

#### Interpretation:
- **4.5 - 4.0**: Excellent quality, minimal degradation
- **3.9 - 3.5**: Good quality, slight degradation noticeable
- **3.4 - 3.0**: Fair quality, moderate degradation
- **2.9 - 2.5**: Poor quality, significant degradation
- **< 2.5**: Very poor quality, severe degradation

#### Technical Notes:
- Requires reference audio for comparison
- Sensitive to noise, distortion, and coding artifacts
- Works best with narrowband (8 kHz) or wideband (16 kHz) audio
- Not suitable for music or non-speech signals

#### Use Cases:
- Codec evaluation
- Transmission quality assessment
- Reference-based TTS evaluation

### STOI (Short-Time Objective Intelligibility)

**Range**: 0.0 - 1.0  
**Higher is better**

STOI measures speech intelligibility, focusing on how well speech can be understood.

#### Interpretation:
- **0.95 - 1.0**: Excellent intelligibility, near-perfect understanding
- **0.90 - 0.94**: Very good intelligibility, minimal comprehension issues
- **0.80 - 0.89**: Good intelligibility, mostly understandable
- **0.70 - 0.79**: Fair intelligibility, some comprehension difficulties
- **0.60 - 0.69**: Poor intelligibility, significant comprehension issues
- **< 0.60**: Very poor intelligibility, difficult to understand

#### Factors Affecting STOI:
- Background noise
- Spectral distortion
- Temporal artifacts
- Missing frequency components

#### Use Cases:
- Hearing aid evaluation
- Noise suppression assessment
- Speech enhancement validation
- Accessibility applications

### MCD (Mel-Cepstral Distortion)

**Range**: 0.0 - ∞ (practical range: 0.0 - 15.0)  
**Lower is better**

MCD measures spectral differences between synthesized and reference speech using mel-cepstral features.

#### Interpretation:
- **0.0 - 2.0**: Excellent similarity, nearly identical spectral characteristics
- **2.1 - 4.0**: Good similarity, minor spectral differences
- **4.1 - 6.0**: Fair similarity, noticeable spectral differences
- **6.1 - 8.0**: Poor similarity, significant spectral differences
- **> 8.0**: Very poor similarity, major spectral distortion

#### Technical Notes:
- Requires reference audio with matching phonetic content
- Sensitive to spectral envelope differences
- Less sensitive to temporal alignment issues (when using DTW)
- Commonly used in TTS research

#### Use Cases:
- Voice conversion evaluation
- TTS system comparison
- Spectral similarity assessment

### Naturalness Score

**Range**: 0.0 - 1.0  
**Higher is better**

Measures how natural the synthesized speech sounds based on multiple acoustic factors.

#### Interpretation:
- **0.90 - 1.0**: Very natural, human-like speech characteristics
- **0.80 - 0.89**: Natural, minor unnatural elements
- **0.70 - 0.79**: Somewhat natural, noticeable synthetic qualities
- **0.60 - 0.69**: Unnatural, obvious synthetic characteristics
- **< 0.60**: Very unnatural, robotic or heavily distorted

#### Components Analyzed:
- **Pitch naturalness**: F0 contour smoothness and range
- **Rhythm naturalness**: Energy variation and timing patterns
- **Spectral naturalness**: Formant structure and spectral tilt

#### Use Cases:
- TTS naturalness assessment
- Voice quality evaluation
- User experience optimization

### Intelligibility Score

**Range**: 0.0 - 1.0  
**Higher is better**

No-reference intelligibility assessment based on spectral and temporal clarity.

#### Interpretation:
- **0.90 - 1.0**: Highly intelligible, clear articulation
- **0.80 - 0.89**: Good intelligibility, mostly clear
- **0.70 - 0.79**: Fair intelligibility, some unclear elements
- **0.60 - 0.69**: Poor intelligibility, many unclear elements
- **< 0.60**: Very poor intelligibility, difficult to understand

#### Factors Considered:
- Spectral clarity
- Temporal envelope modulation
- Noise level estimation

## Pronunciation Metrics

### Phoneme Accuracy

**Range**: 0.0 - 1.0  
**Higher is better**

Measures correctness of individual phoneme pronunciation.

#### Interpretation:
- **0.95 - 1.0**: Native-like pronunciation
- **0.85 - 0.94**: Very good pronunciation, minor errors
- **0.75 - 0.84**: Good pronunciation, some noticeable errors
- **0.65 - 0.74**: Acceptable pronunciation, several errors
- **0.50 - 0.64**: Needs improvement, many errors
- **< 0.50**: Poor pronunciation, significant errors

### Fluency Score

**Range**: 0.0 - 1.0  
**Higher is better**

Evaluates speaking rate, rhythm, and temporal consistency.

#### Interpretation:
- **0.90 - 1.0**: Very fluent, natural rhythm and pacing
- **0.80 - 0.89**: Fluent, minor rhythm irregularities
- **0.70 - 0.79**: Somewhat fluent, noticeable rhythm issues
- **0.60 - 0.69**: Dysfluent, significant rhythm problems
- **< 0.60**: Very dysfluent, poor rhythm and pacing

### Prosody Quality

**Range**: 0.0 - 1.0  
**Higher is better**

Assesses stress patterns, intonation, and emphasis.

#### Interpretation:
- **0.90 - 1.0**: Excellent prosody, natural stress and intonation
- **0.80 - 0.89**: Good prosody, minor irregularities
- **0.70 - 0.79**: Fair prosody, some unnatural patterns
- **0.60 - 0.69**: Poor prosody, many unnatural patterns
- **< 0.60**: Very poor prosody, monotonic or highly irregular

## Comparative Analysis Results

### Quality Difference

**Range**: -∞ - +∞ (practical range: -2.0 - +2.0)  
**Positive favors System A, Negative favors System B**

#### Interpretation:
- **> +0.5**: System A significantly better
- **+0.2 - +0.5**: System A moderately better
- **-0.2 - +0.2**: Systems roughly equivalent
- **-0.5 - -0.2**: System B moderately better
- **< -0.5**: System B significantly better

### Statistical Significance (p-value)

**Range**: 0.0 - 1.0  
**Lower indicates stronger evidence of difference**

#### Interpretation:
- **< 0.001**: Highly significant (p < 0.001)
- **0.001 - 0.01**: Very significant (p < 0.01)
- **0.01 - 0.05**: Significant (p < 0.05)
- **0.05 - 0.10**: Marginally significant
- **> 0.10**: Not significant

### Confidence Score

**Range**: 0.0 - 1.0  
**Higher indicates more reliable assessment**

#### Interpretation:
- **0.90 - 1.0**: Very high confidence, reliable assessment
- **0.80 - 0.89**: High confidence, mostly reliable
- **0.70 - 0.79**: Moderate confidence, some uncertainty
- **0.60 - 0.69**: Low confidence, significant uncertainty
- **< 0.60**: Very low confidence, unreliable assessment

## Contextual Considerations

### Application-Specific Thresholds

Different applications may require different quality thresholds:

#### **Conversational AI / Voice Assistants**
- Minimum STOI: 0.85
- Minimum MOS: 3.5
- Focus on intelligibility and naturalness

#### **Audiobooks / Long-form Content**
- Minimum MOS: 4.0
- Minimum Naturalness: 0.80
- Focus on naturalness and listening comfort

#### **Accessibility Applications**
- Minimum STOI: 0.90
- Minimum Intelligibility: 0.85
- Focus on clarity and comprehension

#### **Professional Broadcasting**
- Minimum MOS: 4.2
- Minimum PESQ: 3.8
- High standards for all metrics

### Language and Speaker Considerations

- **Non-native speakers**: May show lower pronunciation scores even with good intelligibility
- **Accented speech**: Consider language-specific norms and expectations
- **Child speech**: Different normative ranges for pronunciation and fluency
- **Elderly speakers**: May have different prosodic patterns

### Environmental Factors

- **Noise conditions**: Can significantly impact STOI and intelligibility
- **Transmission channels**: May affect PESQ scores
- **Playback systems**: Can influence overall quality perception

## Common Pitfalls and Misinterpretations

### 1. Metric Correlation
- High MOS doesn't guarantee high STOI
- Good MCD doesn't ensure natural prosody
- Consider multiple metrics together

### 2. Reference Dependency
- PESQ and MCD require appropriate reference audio
- No-reference metrics may miss certain quality aspects
- Reference quality affects comparative metrics

### 3. Statistical Significance
- Statistical significance ≠ practical significance
- Small differences may be statistically significant but perceptually irrelevant
- Consider effect sizes alongside p-values

### 4. Score Ranges
- Don't expect perfect scores (1.0 or 5.0) in practice
- Small improvements at high quality levels are more difficult
- Consider the practical range for your application

## Reporting Guidelines

### Scientific Publications
- Report mean ± standard deviation
- Include confidence intervals
- Specify reference conditions
- Provide sample sizes

### Commercial Evaluation
- Focus on user-relevant metrics
- Compare against established benchmarks
- Consider cost-benefit trade-offs
- Include user study validation

### Development Tracking
- Track trends over time
- Monitor metric correlations
- Set achievable targets
- Regular validation with human evaluation
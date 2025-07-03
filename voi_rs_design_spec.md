# VoiRS — A Pure‑Rust Neural Speech Synthesis Framework

**Version:** 0.1.0 (Technical Preview)\
**Status:** Draft for internal review\
**Authors:** Tetsuya Kitahata & Core Contributors\
**License:** MIT OR Apache‑2.0 (dual)

---

## Table of Contents

1. Vision & Mission
2. Goals & Non‑goals
3. Target Use‑Cases
4. Architectural Overview
5. Component Breakdown\
   5.1 Pure‑Rust TTS/Vocoder Pipeline *(new)*
6. Integration with Existing Crates
7. Workspace Layout & Build System
8. Coding Standards & Quality Gates
9. CI/CD & Release Strategy
10. Security, Ethics & Compliance
11. Roadmap & Milestones
12. Contribution & Governance Model
13. Appendices

---

## 1 | Vision & Mission

> **Vision** — Democratise state‑of‑the‑art speech synthesis by delivering a *fully open*, *memory‑safe* and *hardware‑portable* stack built 100 % in Rust.
>
> **Mission** — Unify the author’s high‑performance crate ecosystem (SciRS2, NumRS2, PandRS, TrustformeRS, QuantRS2, etc.) into a cohesive TTS/Vocoder solution that rivals or surpasses proprietary offerings in fidelity, latency, scalability and developer experience.

---

## 2 | Goals & Non‑Goals

| Category        | Goals                                                                      | Explicit Non‑Goals                            |
| --------------- | -------------------------------------------------------------------------- | --------------------------------------------- |
| **Quality**     | Naturalness ≥ MOS 4.4 @ 22 kHz; zero‑shot speaker similarity ≥ 0.85 Si‑SDR | Studio‑grade singing synthesis (Phase 2)      |
| **Performance** | ≤ 0.3× RTF on consumer CPUs, ≤ 0.05× RTF on RTX class GPUs                 | Ultra‑low‑power MCU support                   |
| **Portability** | `x86_64`, `aarch64`, `wasm32`, `nvptx64`, `metal`                          | Legacy 32‑bit targets                         |
| **Usability**   | Single‑line CLI & crate‑level API; first‑class SSML                        | Full GUI editor (provided by downstream apps) |
| **Security**    | Supply‑chain integrity, safe Rust, reproducible builds                     | Proprietary codec shipping                    |

---

## 3 | Target Use‑Cases

- **Edge AI:** Real‑time voice output on drones, kiosks, robots.
- **Assistive Tech:** Low‑latency screen‑reader voices, AAC devices.
- **Media Production:** Batch narration/render farms for podcasts & audiobooks.
- **Conversational Agents:** Integration with TrustformeRS for end‑to‑end LLM‑driven dialogue.

---

## 4 | Architectural Overview

```
┌──────────────────────┐
│   User Application   │  (CLI, gRPC, WASM)
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│     VoiRS Runner     │  (voirs‑cli / FFI)
└──────────┬───────────┘
           ▼
┌──────────────────────────────────────────┐
│           VoiRS Pipeline Core            │
│  ① Text → Phones  G2P  (voirs‑g2p)       │
│  ② Phones → Mel   Acoustic (voirs‑ac)    │
│  ③ Mel → Wave     Vocoder  (voirs‑voc)   │
└──────────┬──────────┬───────────┘
           ▼          ▼
  SciRS2 DSP   NumRS2 TensorCore  …
```

- **Data Exchange:** `ndarray`‑compatible zero‑copy tensors with `Cow<T>` for CPU/GPU seamlessness.
- **Back‑Ends:** `candle` (wgpu/CUDA/Metal) primary; `ort` fallback for ONNX models; experimental `burn` for training.

---

## 5 | Component Breakdown

| Crate            | Artifact     | Key Responsibilities                                                    | External Deps                       |
| ---------------- | ------------ | ----------------------------------------------------------------------- | ----------------------------------- |
| `voirs-g2p`      | `lib`, `bin` | Grapheme‑to‑Phoneme conversion; pluggable language adapters; IPA output | `phonetisaurus-g2p`, `openjtalk-rs` |
| `voirs-acoustic` | `lib`        | VITS‑derived acoustic model inference (Candle)                          | `candle-core`, `safetensors`        |
| `voirs-vocoder`  | `lib`        | HiFi‑GAN & DiffWave inference; model zoo loader                         | `ort`, `candle-core`                |
| `voirs-dataset`  | `lib`        | LJSpeech, JVS etc. download, normalise, shard                           | `pandrs`, `reqwest`                 |
| `voirs-cli`      | `bin`        | Command‑line synthesis, benchmarking, packaging                         | `clap`, `hound`, `opusenc`          |
| `voirs-ffi`      | `cdylib`     | C/Python bindings for Unreal, Unity, PyTorch                            | `ffi-support`                       |
| `voirs-sdk`      | meta         | Re‑export of public API; semantic version core                          | —                                   |

### 5.1 | Pure‑Rust TTS/Vocoder Pipeline

The following reference implementation illustrates how every stage—from text input to final waveform—can be executed **exclusively in safe Rust**.

#### 5.1.1 End‑to‑End Pipeline Diagram

```
┌──────────────┐   ┌──────────────────┐   ┌──────────────┐
│ Text / SSML  │→ │ Front‑End (G2P)   │→ │  Acoustic     │
│  input       │   │ phonemes → mels  │   │  Model (VITS)│
└──────────────┘   └──────────────────┘   └────┬─────────┘
                                               ▼
                                       ┌──────────────┐
                                       │ DiffWave /   │
                                       │ HiFi‑GAN     │
                                       │  Vocoder     │
                                       └────┬─────────┘
                                               ▼
                                       ┌──────────────┐
                                       │ WAV / OPUS   │
                                       └──────────────┘
```

#### 5.1.2 Front‑End (Grapheme‑to‑Phoneme)

| Purpose                | Crate                     | Notes                      |
| ---------------------- | ------------------------- | -------------------------- |
| English & Multilingual | `phonetisaurus-g2p`       | FST‑based, lightning fast  |
| Lightweight NN G2P     | `grapheme_to_phoneme`     | Bundled LSTM model         |
| Japanese               | `openjtalk-rs` (VOICEVOX) | FFI wrapper over OpenJTalk |

> **Tip:** Provide a `trait G2p { fn to_phonemes(&self, txt: &str) -> Vec<Phone>; }` so implementations can be swapped at compile‑time. SciRS2/NumRS2 may supply SIMD speed‑ups for DSP‑heavy phoneme post‑processing.

#### 5.1.3 Acoustic Model (Text → Mel)

- **Quick‑start**:
  - `piper-rs`—loads pre‑converted VITS ONNX, real‑time on Raspberry Pi.
  - `voicevox-rs`—high‑quality Japanese with emotion parameters.
- **Custom / Research**:
  - `candle` backend for `.safetensors` VITS graphs (CUDA/Metal/wgpu).
  - `burn` when you want *training* entirely in Rust.

#### 5.1.4 Vocoder (Mel → Wave)

| Method   | Pure‑Rust Option                   | Highlights                                |
| -------- | ---------------------------------- | ----------------------------------------- |
| HiFi‑GAN | ONNX + `ort` (CPU/GPU)             | Blazing fast, crisp high‑freqs            |
| DiffWave | ONNX + `ort`, or `candle-diffwave` | WaveNet‑level quality with simple L2 loss |

#### 5.1.5 Unified Runtime — Code Skeleton

```toml
# Cargo.toml
[dependencies]
phonetisaurus-g2p = "0.3"
candle             = { version = "0.4", features=["cuda"] }
ort                = { version = "1.22", features=["cuda"] }
hound              = "3"   # WAV output
anyhow             = "1"
```

```rust
use candle::{Tensor, Device};
use ort::{Environment, SessionBuilder};
use phonetisaurus_g2p::G2P;

fn main() -> anyhow::Result<()> {
    // 1. G2P
    let g2p = G2P::from_model("cmudict.fst")?;
    let phones = g2p.phoneticize("Rust makes systems fearless!", 1)?;

    // 2. Acoustic (VITS)
    let env = Environment::builder().with_name("voirs").build()?;
    let vits = SessionBuilder::new(&env)?.with_model_from_file("vits.onnx")?;
    let mel: Tensor = run_vits(&vits, &phones)?;

    // 3. Vocoder (DiffWave)
    let diff = SessionBuilder::new(&env)?.with_model_from_file("diffwave.onnx")?;
    let wav = run_vocoder(&diff, mel)?;

    // 4. Save
    save_wav("output.wav", &wav, 22_050)?;
    Ok(())
}
```

#### 5.1.6 Workspace‑On‑Crates Layout

```
voirs/
 ├─ crates/
 │   ├─ g2p-core/
 │   ├─ vits-infer/
 │   ├─ diffwave-infer/
 │   └─ pipeline/
 ├─ models/
 │   ├─ vits_en_us.onnx
 │   └─ diffwave_22k.onnx
 └─ examples/
     └─ synth.rs
```

#### 5.1.7 Build & Distribution Matrix

| Target          | Command / Notes                                                               |
| --------------- | ----------------------------------------------------------------------------- |
| **CPU Only**    | `cargo build --release --features="ort/cpu candle/blas"`                      |
| **NVIDIA GPU**  | `export ORT_USE_CUDA=1` then `cargo build --release --features="candle/cuda"` |
| **WebAssembly** | `wasm32-unknown-unknown` + wgpu backend; 12 kHz real‑time after quantisation  |

#### 5.1.8 Future Extensions

1. **Multilingual Expansion** — Generic `G2p` trait adapters for arbitrary language packs.
2. **On‑Device Optimisation** — TensorRT via ORT, Metal backend, WASM SIMD128.
3. **Rust‑Native Training** — Swap Python out with `burn` or `candle-train`.
4. **LLM Synergy** — Feed TrustformeRS tokens directly into VoiRS for conversational agents.

---

## 6 | Integration with Existing Crates

| Existing Crate   | Integration Strategy                                                |
| ---------------- | ------------------------------------------------------------------- |
| **SciRS2**       | FFT, mel filter‑bank, melspec loss; compile‑time SIMD feature flags |
| **NumRS2**       | BLAS/LAPACK kernels for LSTM & attention; autograd bridge to Candle |
| **PandRS**       | Dataset ETL, CSV/Parquet manifest handling                          |
| **TrustformeRS** | Prompt generation, voice style conditioning via LLM token stream    |
| **QuantRS2**     | Quantised model export and dynamic range calibration                |

> **Design Decision #001:** Treat SciRS2 & NumRS2 as *optional accelerators* behind `feature = "scirs" | "numrs"` to keep minimal builds tiny.

---

## 7 | Workspace Layout & Build System

```
voirs/
 ├─ Cargo.toml          # virtual workspace
 ├─ crates/
 │   ├─ g2p/
 │   ├─ acoustic/
 │   ├─ vocoder/
 │   ├─ dataset/
 │   ├─ cli/
 │   └─ ffi/
 ├─ models/
 │   ├─ vits_en_us.safetensors
 │   └─ diffwave_22k.onnx
 └─ .cargo/
     └─ config.toml     # target‑specific linker flags
```

- **Build Profiles:** `dev`, `release`, `release-gpu`, `dist` (LTO + PGO).
- **Cross‑Compilation:** `cross.toml` presets for `aarch64-apple-darwin`, `wasm32-wasi`.

---

## 8 | Coding Standards & Quality Gates

- **Rust Edition:** 2024
- **Lint:** `cargo clippy --all-targets --all-features --deny=warnings`
- **Fmt:** `cargo fmt --check` (rustfmt 2 profile)
- **Unsafe Code:** forbidden by default; justify via `#[allow(unsafe_code)]` per module.
- **Testing:** `cargo nextest`, golden audio regression (< ‑20 dB diff), property tests with `proptest`.
- **Docs:** `cargo doc --workspace --no-deps -Adead_code` + mdBook dev guide.

---

## 9 | CI/CD & Release Strategy

| Stage          | Tooling                                                             | Outcome                               |
| -------------- | ------------------------------------------------------------------- | ------------------------------------- |
| **CI**         | GitHub Actions matrix (`ubuntu-latest`, `macos-14`, `windows-2022`) | Unit + audio tests,  lint, doc build  |
| **Security**   | `cargo audit`, `cargo deny`                                         | SBOM & vuln scan                      |
| **CD**         | `cargo-dist`                                                        | Signed binaries (.tar.gz, .msi, .pkg) |
| **Container**  | `docker buildx` multi‑arch                                          | `ghcr.io/cool-japan/voirs:<tag>`      |
| **Versioning** | Semantic Versioning 2.0                                             |  `vX.Y.Z` Git tags                    |

---

## 10 | Security, Ethics & Compliance

- **Voice Clone Abuse Mitigation:** watermarking, opt‑in speaker IDs.
- **Model Licences:** bundle only data/models with permissive terms.
- **Supply‑Chain:** Sigstore + SLSA Level 2 for releases.
- **GDPR/PDPA:** No personal data stored within repo; datasets hosted externally.

---

## 11 | Roadmap & Milestones

| Q           | Milestone    | Key Deliverables                                              |
| ----------- | ------------ | ------------------------------------------------------------- |
| **Q3 2025** | **MVP 0.1**  | English VITS + DiffWave, CLI & WASM demo                      |
| **Q4 2025** | **v0.5**     | Multilingual G2P, GPU kernels, FFI bindings                   |
| **Q1 2026** | **v1.0 LTS** | Stable API, model zoo, TrustformeRS bridge                    |
| **Q3 2026** | **v2.0**     | End‑to‑end Rust training loop (Burn), live speaker adaptation |

---

## 12 | Contribution & Governance Model

- **BDFL:** Tetsuya Kitahata until v1.0 GA.
- **RFC Process:** PR + `000X-rfc-*.md` tracked in `docs/rfc/`.
- **CLA:** DCO sign‑off (`Signed‑off‑by`) suffices.
- **Meetings:** Monthly community call (recordings → GitHub Discussions).

---

## 13 | Appendices

- **Glossary** (MOS, RTF, IPA, SSML, etc.)
- **External References** (Piper, VITS, DiffWave papers)
- **Decision Log** in `docs/adr/`

---

*End of Document*


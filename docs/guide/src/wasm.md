# WebAssembly (WASM) Build Guide

This guide explains how to build relevant VoiRS crates for WebAssembly targets such as `wasm32-unknown-unknown`.

## Primary Crates

- `voirs-conversion` (real‑time voice conversion / Web Audio API integration)
- Any crate exposing a `wasm` feature

## `getrandom` 0.3 WebAssembly Requirements

With `getrandom` 0.3 the Web backend is not automatically selected for `wasm32-unknown-unknown`. You MUST do two things:

1. Enable the `wasm_js` feature in `Cargo.toml` (e.g. `getrandom = { version = "0.3", features=["wasm_js"] }`).
2. Provide a backend selection at build time via `RUSTFLAGS`:

```bash
RUSTFLAGS='--cfg getrandom_backend="wasm_js"' \
  cargo build --target wasm32-unknown-unknown -p voirs-conversion --features wasm
```

This activates the `Crypto.getRandomValues` path via `wasm-bindgen` / `js-sys` / `web-sys`.

## Recommended Targets

| Use Case | Target | Notes |
|----------|--------|-------|
| Browser execution | `wasm32-unknown-unknown` | Use `wasm-bindgen` + bundler (Vite, Webpack, etc.) |
| WASI experiments | `wasm32-wasip1` | Limited host I/O; adapt file access |

## Example Feature Build

Minimal web build for `voirs-conversion`:

```bash
RUSTFLAGS='--cfg getrandom_backend="wasm_js"' \
  cargo build --target wasm32-unknown-unknown \
  -p voirs-conversion \
  --no-default-features \
  --features wasm
```

## Size Optimization Tips

```bash
# Enable LTO, abort-on-panic, and then post-optimize the wasm
RUSTFLAGS='-C opt-level=z -C linker-plugin-lto -C panic=abort --cfg getrandom_backend="wasm_js"' \
  cargo build --release --target wasm32-unknown-unknown -p voirs-conversion --features wasm

wasm-opt -Oz -o output.opt.wasm target/wasm32-unknown-unknown/release/voirs_conversion.wasm
```

## Runtime Considerations

- Audio scheduling obeys `Web Audio API` sample rate/thread constraints.
- Full `tokio` features rely on `wasm-bindgen-futures`; timers and tasks are cooperatively scheduled.
- Direct filesystem APIs are unavailable in browsers; replace with fetch/IndexedDB abstractions.

## Common Errors

| Symptom | Cause | Fix |
|---------|-------|-----|
| `failed to select a version for getrandom` | Old `features=["js"]` name | Use `wasm_js` + proper `RUSTFLAGS` |
| `random device not available` | Missing backend cfg | Add `RUSTFLAGS='--cfg getrandom_backend="wasm_js"'` |
| `linking with wasm-ld failed` | Native-only dependency pulled in | Build with `--no-default-features` or guard via `cfg` |

## Testing in WASM

Example (headless Chrome):

```bash
cargo install wasm-pack
RUSTFLAGS='--cfg getrandom_backend="wasm_js"' wasm-pack test --headless --chrome crates/voirs-conversion
```

---
_Last updated: 2025-10-01_

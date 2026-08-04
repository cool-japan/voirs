# VoiRS Training Guide

Guide for training VoiRS models with the `voirs train` CLI command. Every flag,
default, and behavior described below was verified against the current source
in `crates/voirs-cli/src/commands/train/` (`mod.rs`, `vocoder/mod.rs`,
`acoustic.rs`, `g2p.rs`, `progress.rs`, `data_loader.rs`).

## Status at a glance

| Command | Status | What actually happens |
|---|---|---|
| `voirs train vocoder --model-type diffwave` | **Implemented** | Real forward + backward pass (Candle autograd, `AdamW`), real per-step and per-epoch losses, real SafeTensors checkpoints |
| `voirs train vocoder --model-type hifigan` | **Implemented (generator only)** | Real forward + backward pass on the HiFi-GAN V2 generator with an L1+L2 reconstruction loss; no discriminators / adversarial training |
| `voirs train acoustic --model-type vits` | **Fails closed** | Input is validated, then the command returns a `Not implemented:` diagnostic explaining that `VitsTrainer` never performs a backward pass |
| `voirs train acoustic --model-type fastspeech2` | **Fails closed** | Same guard; `FastSpeech2Trainer` never performs a backward pass, and its encoder collapses every phoneme to the same constant id |
| `voirs train g2p` | **Fails closed** | The pronunciation dictionary is loaded and parsed for real, but `LstmTrainer` never performs a backward pass; the command refuses before writing a fabricated model file |

This is a deliberate design decision, documented directly in the source
(`crates/voirs-cli/src/commands/train/acoustic.rs`,
`crates/voirs-cli/src/commands/train/g2p.rs`): rather than silently hand back
an untrained, randomly-initialized model dressed up with a "training
completed successfully" banner, these commands refuse to run and explain
exactly what is missing. See the root `README.md` (Core Components table) for
the same summary, and `TODO.md` for the broader CLI/training roadmap.

---

## 1. Vocoder training (implemented)

### Data requirements — LJSpeech format only

`voirs train vocoder` loads data through `voirs-dataset`'s `LjSpeechLoader`,
which is currently the **only** supported dataset format for this command. A
directory is recognized as valid when it contains:

```
<data-dir>/
├── metadata.csv        # pipe-delimited, >=3 fields per row: id|normalized_text|original_text
└── wavs/
    ├── <id>.wav
    ├── <id>.wav
    └── ...
```

- The text columns exist only for compatibility with the LJSpeech file format
  — vocoder training does not use transcriptions or phonemes. It reads
  `sample.audio` and derives a real mel spectrogram from it via FFT
  (`extract_mel_spectrogram`); no forced alignment or phoneme file is needed.
- A row whose `wavs/<id>.wav` is missing is skipped with a warning, not a hard
  failure.
- If `metadata.csv` or `wavs/` is missing entirely, the command fails
  immediately with "Unsupported dataset format ... Currently only LJSpeech is
  supported."
- 80 mel channels are used unconditionally (`MelSpectrogramConfig::default()`).

### Command

```bash
voirs train vocoder \
    --model-type diffwave \
    --data data/LJSpeech-1.1 \
    --output checkpoints/diffwave \
    --epochs 1000 \
    --batch-size 16 \
    --lr 0.0002 \
    --gpu
```

`--model-type` accepts `diffwave` (default) or `hifigan`.

### Flag reference (from the clap definition in `commands/train/mod.rs`)

| Flag | Default | Verified behavior |
|---|---|---|
| `--model-type <TYPE>` | `diffwave` | `diffwave` or `hifigan`; anything else is a config error |
| `--data <PATH>` | required | LJSpeech-format dataset directory (see above) |
| `-o, --output <PATH>` | `checkpoints/vocoder` | Output directory for checkpoints |
| `-c, --config <PATH>` | none | **Real.** TOML or JSON file (selected by extension), loaded and parsed into scheduler/early-stopping/checkpoint-cadence settings — see "Config file" below. An unreadable file, a parse error, or an unsupported extension is a hard error (`voirs train vocoder` refuses to run rather than silently ignoring the flag). |
| `--epochs <N>` | `1000` | Number of training epochs. CLI-only — not read from `--config`. |
| `--batch-size <N>` | `16` | Batch size. CLI-only — not read from `--config`. |
| `--lr <F>` | `0.0002` | Base learning rate passed to `AdamW::new_lr` at optimizer construction. CLI-only — not read from `--config`. Actual per-step/per-epoch rate can move away from this via warmup/scheduler, see below. |
| `--lr-scheduler <NAME>` | `none` | `none`, `step`, `cosine`, `exponential`, `onecycle`, `plateau`. **Real** — applied to the optimizer via `Optimizer::set_learning_rate`, see below. An unrecognized name is rejected up front (fail closed), not silently treated as `none`. Config-file fallback if `--config` is given and this flag is omitted. |
| `--lr-step-size <N>` | `100` | For `step`: decay interval in epochs. For `plateau`: number of epochs without validation improvement before one decay is applied. Real (see below). Config-file fallback. |
| `--lr-gamma <F>` | `0.1` | Decay factor for `step`/`exponential`/`plateau`. Real (see below). Config-file fallback. |
| `--early-stopping` | off | Enables early stopping (presence flag, no value). Can also be enabled by `--config`'s `early_stopping = true`; the flag can only turn it *on*, never override a config file's `true` back to `false`. |
| `--patience <N>` | `50` | Real epochs without validation improvement before stopping — see "Early stopping is epoch-based" below. Config-file fallback. |
| `--min-delta <F>` | `0.0001` | Minimum val-loss improvement to reset patience / save a new best checkpoint. Config-file fallback. |
| `--val-frequency <N>` | `5` | Run validation every N epochs (epoch 0 always validates, since `0 % N == 0`). Config-file fallback. |
| `--warmup-steps <N>` | `0` | Linear LR warmup over the first N optimizer steps. **Real** — applied to the optimizer, see below. Config-file fallback. |
| `--grad-clip <F>` | `1.0` | Real global-norm gradient clipping (the same algorithm as `torch.nn.utils.clip_grad_norm_`), identical for DiffWave and HiFi-GAN — see below. Config-file fallback. |
| `--save-frequency <N>` | `10` | Save an `epoch_<N>` checkpoint every N epochs (epoch 0 always saves). Config-file fallback. |
| `--resume <PATH>` | none | **Real.** Loads real weights from the checkpoint into the model's `VarMap` before training starts — see "Resume" below. A missing or incompatible checkpoint is a hard error, not a silent fresh start. |
| `--gpu` | off | Tries Metal (if built with the `metal` feature) or CUDA (if built with `cuda`, and not `metal`), falling back to CPU with a printed warning on failure or if neither feature was compiled in. Also enabled by the global `--gpu` flag (`voirs --gpu train vocoder ...`). CLI-only — not read from `--config`. |

### Config file

`--config <PATH>` loads a TOML or JSON file (selected by the `.toml`/`.json`
extension; anything else is rejected) into the scheduler/early-stopping/
checkpoint-cadence settings above (the same fields as `TrainingConfig`).
Fields use the struct's snake_case names (`lr_scheduler`, `lr_step_size`,
`lr_gamma`, `early_stopping`, `patience`, `min_delta`, `val_frequency`,
`warmup_steps`, `grad_clip`, `save_frequency`); the CLI flag's kebab-case
spelling also works as an alias (e.g. either `lr_scheduler` or `lr-scheduler`
as the key). A field left out of the file simply isn't overridden.

Precedence, highest first: **(1) an explicit CLI flag, (2) the config file,
(3) the built-in default.** `--early-stopping` is the one exception, since a
bare presence flag has no "explicitly false" state to compare against: it is
effectively OR'd with the file's `early_stopping` value (either can turn it
on; the CLI flag can never turn off a file's `true`).

Deliberately **not** read from the config file: `--epochs`, `--batch-size`,
`--lr`, `--data`, `--output`, `--resume`, `--gpu`. Those define the invocation
itself, not a tunable hyperparameter, and stay CLI-only.

A missing file, a parse error, or an unsupported extension is a hard,
typed error — `--config` is never silently ignored.

### Learning rate: warmup and scheduler are real

Both the warmup calculation and `apply_lr_scheduler(...)` compute a target
learning rate that is applied to the optimizer via candle-nn's
`Optimizer::set_learning_rate` (`AdamW` implements this), immediately after
the value is computed — warmup after each optimizer step while
`total_steps <= warmup_steps`, the scheduler once per epoch afterward. The
live "LR:" figure in the metrics bar and the periodic "📊 Learning rate: ..."
log line are both read back from `optimizer.learning_rate()`, the same value
just applied, so the display can never drift from what gradient updates
actually use.

`--lr-scheduler`'s value is validated up front (before the dataset/model are
even loaded): an unrecognized name is rejected with a listing of the
supported ones, rather than silently behaving like `none` deep inside the
training loop. `plateau` decays the learning rate by `--lr-gamma` for every
full `--lr-step-size` epochs validation loss has gone without improving
(purely a function of how long the run has been stagnant, so it resets to
flat the moment validation improves again) — it is a real,
`ReduceLROnPlateau`-style schedule now, not a `none`-equivalent placeholder.

**`onecycle` now genuinely changes training, including exceeding `--lr`:**
it ramps from 1×`--lr` up to 2×`--lr` at the run's midpoint, then back down
to 1×`--lr` by the end — it does not anneal toward a near-zero final rate the
way a textbook One-Cycle schedule does. This was harmless while the value
only drove a display number; now that it reaches the optimizer,
`--lr 0.0002 --lr-scheduler onecycle` genuinely trains at up to `0.0004` for
part of the run. Factor that in when choosing `--lr` for this scheduler.

### Gradient clipping and the "Grad:" figure are both real, and identical for both models

`--grad-clip` now performs true global-norm gradient clipping — the same
algorithm as `torch.nn.utils.clip_grad_norm_` — for **both** DiffWave and
HiFi-GAN, via one shared function (`backward_step_with_grad_norm` /
`clip_gradients`). After the backward pass, the real L2 norm of the gradients
is computed across every trainable parameter; if it exceeds `--grad-clip`
(and `--grad-clip > 0`), every gradient is rescaled so the resulting norm
equals `--grad-clip` exactly, before the optimizer step. The metrics bar's
"Grad:" figure is that same real, pre-clipping norm — not a hardcoded
constant. This replaces an earlier loss-scaling approximation that only ever
applied to DiffWave; HiFi-GAN previously ignored `--grad-clip` entirely.

### Resume

`--resume <PATH>` loads the checkpoint's real tensor values into the newly
constructed model's `VarMap` via candle's `VarMap::load`, after the model is
built (so every parameter name is registered) and before the optimizer is
created. A missing checkpoint file, or one whose tensor names/shapes don't
match the model being trained, is a typed error — training refuses to
silently fall back to a fresh, randomly-initialized model.

Only **weights** are restored. `save_checkpoint` never persisted optimizer
state (`AdamW`'s per-parameter first/second moment buffers), so those always
restart fresh on resume — this is a real, honestly-documented limitation of
the checkpoint format, not something resume pretends to solve.

The epoch to resume from is best-effort recovered from the checkpoint's
`<name>.json` sidecar (written alongside every `.safetensors` checkpoint by
`save_checkpoint`; see "Checkpoints" below): training continues at the epoch
after the one the sidecar recorded. If the sidecar is missing or unreadable
(e.g. only the `.safetensors` file was copied), weights still load correctly,
but the epoch counter honestly restarts at 0 — this is logged, not silently
assumed.

### Training loop behavior

- **Per-step loss**: DiffWave uses MSE between predicted and actual diffusion
  noise; HiFi-GAN uses a 0.45×L1 + 0.55×L2 reconstruction loss between
  generated and target audio. Both are real values from real forward passes.
- **Per-epoch loss**: `epoch_loss` is divided by the number of *successfully
  completed* batches, not by `batches_per_epoch` — a failed batch cannot dilute
  the reported average toward a falsely lower number.
- **Failure handling**: a per-batch training-step failure is logged, and the
  batch's displayed loss falls back to the last real measurement (or `NaN` if
  none has succeeded yet). Failures accumulate in a counter that is **never
  reset between epochs**. Once accumulated failures exceed half of one
  epoch's batch count (`error_count > batches_per_epoch / 2`), the run aborts
  with a typed error naming the failure count and the last underlying error,
  instead of continuing indefinitely on synthetic data.
- **Validation is not a held-out split**: `run_validation` draws its samples
  from the same data loader used for training, starting at whatever index
  training last left off, and restores that index afterward so training can
  continue. It uses `max(len/10, 32)` samples and runs forward-only (no
  optimizer step). There is no dedicated train/val partition.
- **Early stopping is epoch-based, matching its `--help` text.** The run
  tracks `last_improvement_epoch` (the real epoch validation loss last
  improved at) and compares `epoch - last_improvement_epoch` against
  `--patience` directly — real elapsed epochs, not a counter incremented once
  per validation *event*. With the defaults (`--val-frequency 5`,
  `--patience 50`), stopping now genuinely triggers after 50 epochs without
  improvement, not ~250. (Design choice: patience was converted to mean real
  epochs, matching its documented "(epochs)" semantics, rather than changing
  the help text to describe the old validation-event counting.) A `--resume`d
  run initializes `last_improvement_epoch` to the resumed start epoch, so it
  doesn't inherit a stale improvement point from before the process
  restarted.

### Checkpoints

Written into `--output`, using a hand-built (but format-compliant) SafeTensors
encoder that serializes the real, current parameter values from the model's
`VarMap`:

| File | When written |
|---|---|
| `epoch_<N>.safetensors` + `epoch_<N>.json` | Every `save_frequency` epochs (epoch 0 included) |
| `best_model.safetensors` + `best_model.json` | Whenever validation loss improves by more than `--min-delta` |
| `final_model.safetensors` + `final_model.json` | Once at the end of the run, **only if at least one epoch completed**; otherwise nothing is written and a warning is printed instead |

Each `.json` sidecar contains real values: `epoch`, `train_loss`, `val_loss`
(`null`, not a fabricated number, if validation didn't run that epoch),
`timestamp`, `model_type` (`"DiffWave"` or `"HiFiGan"`), and the list of
tensor names/shapes actually written.

### Progress display

- Epoch and batch progress bars (`indicatif`) with live loss and samples/sec,
  both computed from real batch timings.
- The metrics bar shows `Loss / LR / Grad / Time`, and all four are now real:
  loss and elapsed time as before; LR is read back from
  `optimizer.learning_rate()` after warmup/scheduler apply it (see "Learning
  rate" above); `Grad` is the real pre-clipping gradient global L2 norm (see
  "Gradient clipping" above) — no hardcoded placeholder remains.
- A resources line shows `CPU: X% | RAM: Y GB | GPU: N/A`. CPU% is a real
  instantaneous rate computed from `getrusage` deltas on Unix, normalized by
  core count (honestly reported as `0.0` on non-Unix, since no sampling API is
  wired up there — never a fabricated constant). RAM is real (`vm_statistics64`
  on macOS, `/proc/meminfo` on Linux, `0.0` fallback elsewhere). GPU% is always
  `None`/`N/A` — no GPU usage monitoring is implemented.
- Final summary (`Total time`, `Epochs completed`, `Total steps`, `Final
  train/val loss`, `Best val loss`, `Avg samples/sec`) is built entirely from
  values accumulated during the real run.

---

## 2. Acoustic and G2P training (not yet implemented)

`voirs train acoustic vits|fastspeech2` and `voirs train g2p` accept the same
kind of arguments as the vocoder command (data/dictionary path, output path,
epochs, batch size, learning rate, etc. — see `--help` for the exact list per
subcommand), validate their inputs for real, and then return a typed
`CliError::NotImplemented` (`"Not implemented: ..."`) rather than run a
training loop that cannot learn.

Unlike `train vocoder`, these two subcommands' `--config <PATH>` flag is
still parsed but not read (`AcousticModelTrainingArgs.config` /
`run_train_g2p`'s `config` parameter are unused). This is a deliberate,
narrower scope than the vocoder fix: both subcommands fail closed
*unconditionally*, before any hyperparameter a config file could supply would
ever matter, so the user already gets a hard, explanatory diagnostic on every
invocation regardless of `--config` — there is no silently-different behavior
for a config file to enable or disable. Real config-file loading for these
two subcommands should be revisited once `voirs-acoustic`/`voirs-g2p` gain
real gradient-based training and the guard above is removed.

That error now propagates all the way to the process exit code.
`commands::train::execute_train_command` (and every function it calls —
`run_train_vocoder`, `run_train_acoustic`, `run_train_g2p`) returns
`crate::error::Result<T>` (i.e. `CliError`, not a generic
`voirs_sdk::VoirsError`), so the original variant survives. `CliApp`'s
`Commands::Train` arm (`crates/voirs-cli/src/lib.rs`) matches on that `Err`,
prints it, flushes stdout, and calls `std::process::exit(cli_err.exit_code())`
directly, instead of re-wrapping it into a generic `VoirsError::config_error`
and letting `main()`'s default `#[tokio::main]` failure path exit with `1`
for every kind of training failure. `CliError::exit_code()`
(`crates/voirs-cli/src/error/mod.rs`) is therefore live code for training
commands: a fail-closed `NotImplemented` (as both `train acoustic` and
`train g2p` currently always are) exits with code `19`; other `CliError`
variants — e.g. a `Config` error for a bad `--model-type` or a missing
`--data` directory — exit with their own documented code (`Config` is `1`,
same as the old default, so that particular case isn't visibly different).

Verify with `voirs train acoustic --model-type vits --data <a-real-empty-dir> 2>/dev/null; echo $?` → `19`.
(A nonexistent `--data` path fails validation *before* reaching the
`NotImplemented` fail-closed path, exiting `1` for `CliError::Config`
instead — pass a real, existing, empty directory to reach `19`.)

The model **architectures** ship and work for inference; the **trainers** do
not yet perform real gradient-based training. Evidence, from the source doc
comments:

- **VITS** (`crates/voirs-acoustic/src/vits/trainer.rs`): `VitsTrainer::
  train_step` never calls a backward pass or optimizer step, so weights never
  change; `PeriodDiscriminator`/`ScaleDiscriminator::simulate_conv` return
  `Tensor::zeros(..)` regardless of input, making every
  discriminator/adversarial/feature-matching loss an input-independent
  constant; KL-divergence and duration losses are `fastrand`-generated
  numbers unrelated to the model; `validate_step` is entirely
  `fastrand`-based; `save_checkpoint` writes JSON metadata, not real tensor
  weights, to the `.safetensors` path.
- **FastSpeech2** (`crates/voirs-acoustic/src/fastspeech2_trainer.rs`): same
  no-backward-pass issue; additionally `FastSpeech2Encoder::forward` maps
  every phoneme to the same constant id (`.map(|_| 1u32)`), discarding all
  phoneme identity before it reaches the model, and the variance adaptor
  predicts duration/pitch/energy from an all-zero placeholder feature vector.
- **G2P** (`crates/voirs-g2p/src/backends/neural/training.rs`):
  `LstmTrainer::train_epoch` genuinely runs a real forward pass over real
  dictionary data (real index tensors, a real encoder/decoder pass, a real
  loss computation) — the closest to real of the three — but the code
  comments directly above the loss accumulation say "Simulated backward pass
  and parameter update," so weights never change; `calculate_sequence_loss`
  substitutes a constant `0.5` on tensor-shape mismatches; and
  `save_model_safetensors` ignores the trained encoder/decoder entirely,
  writing a constant dummy tensor and a descriptive string instead of real
  weights.

Running any of these loops today would silently hand back an untrained (or,
for FastSpeech2, content-blind) model presented as a completed training run —
so the CLI refuses instead. This guard should be removed once the
corresponding trainer implements a real backward pass / optimizer step. See
`TODO.md` for the broader training/CLI implementation roadmap.

---

## 3. Pre-trained models

If you don't need a custom-trained vocoder, you can list and fetch existing
models instead of training your own:

```bash
voirs list-models [--backend <name>] [--detailed]
voirs download-model <model-id> [--force]
```

See `voirs list-models --help` / `voirs download-model --help` for the
current option list; this guide does not reproduce the model catalog to avoid
drift.

---

## Support

- Issues: https://github.com/cool-japan/voirs/issues
- License: Apache-2.0 (see `LICENSE`)

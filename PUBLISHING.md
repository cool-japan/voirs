# Publishing VoiRS to crates.io

This is a runbook for publishing the VoiRS workspace, not a description of a
process that has been executed yet. Every claim below was verified against
the actual repository state (`cargo metadata`, `cargo package`, `cargo tree`)
on the `0.1.0` branch; re-verify before an actual publish if time has passed.

## 1. Pre-publish gate

Run these from the repo root and require all to pass before publishing anything:

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo nextest run --all-features --no-fail-fast
cargo deny check bans          # must exit 0 (see deny.toml)
```

`cargo deny check licenses` currently still fails on one pre-existing,
intentionally-unresolved item: `mp3lame-encoder`/`mp3lame-sys` (pulled in
transitively through `voirs-sdk`'s optional MP3 support, only reachable with
`--all-features`) are LGPL-3.0 **and** `mp3lame-sys` wraps the C `libmp3lame`
library — a likely Pure Rust policy violation in addition to the license
concern. This was not in this batch's ownership (`voirs-sdk/Cargo.toml`); it
needs its own review (drop the optional MP3 feature's default inclusion in
`--all-features`, or replace it with a pure-Rust MP3 encoder) before an
`--all-features` publish/build claim is fully policy-compliant.

## 2. Required publish order

Package versions are pinned via `.workspace = true` (root `Cargo.toml`
`version = "0.1.0"`). Cargo resolves a crate's *published* dependencies from
crates.io, not from the local workspace, so **every internal dependency of a
crate you're about to publish must already exist on crates.io at that exact
version** — otherwise `cargo publish`/`cargo package` fails immediately with
a "candidate versions found which didn't match" error (verified below).

The order is derived from `cargo metadata --no-deps`'s actual internal
dependency edges among the 15 publishable crates (`voirs-examples` and
`voirs-integration-tests` are `publish = false` / workspace-only and never
published), not assumed:

| Wave | Crates | Depends on (internal) |
|------|--------|------------------------|
| 1 | `voirs-g2p`, `voirs-vocoder`, `voirs-dataset`, `voirs-singing`, `voirs-spatial` | none |
| 2 | `voirs-acoustic` | g2p, vocoder |
| 3 | `voirs-cloning`, `voirs-emotion` | acoustic (+ g2p for cloning) |
| 4 | `voirs-conversion` | acoustic, cloning, emotion, spatial |
| 5 | `voirs-sdk` | acoustic, cloning, conversion, dataset, emotion, g2p, spatial, vocoder |
| 6 | `voirs-recognizer` | sdk |
| 7 | `voirs-evaluation`, `voirs-ffi` | g2p, recognizer, sdk (+ vocoder for ffi) |
| 8 | `voirs-feedback`, `voirs-cli` | evaluation, recognizer, sdk (+ everything else, for cli) |

Within a wave, crates can be published in any order (no edges between them).
**Publish strictly wave-by-wave**, waiting for each crate to actually appear
on crates.io (index propagation can lag a minute or two) before starting the
next wave.

Note this corrects an earlier draft ordering that grouped `voirs-cloning`/
`voirs-emotion`/`voirs-conversion` *after* `voirs-sdk`: `voirs-sdk` directly
depends on all three, so they must publish strictly before it, not after.
Likewise `voirs-recognizer` → `voirs-evaluation` → `voirs-feedback` is a
strict chain (each depends on the previous), not a single parallel batch.

### Verifying the order before a real publish

```bash
cargo metadata --no-deps --format-version 1 | \
  python3 -c "import json,sys; d=json.load(sys.stdin); \
  [print(p['name'], sorted(x['name'] for x in p['dependencies'] if x['name'].startswith('voirs-'))) \
   for p in d['packages'] if p['name'].startswith('voirs-') and p.get('publish') != []]"
```

Re-run this if `Cargo.toml` dependency edges change between now and the
actual publish; the table above is a snapshot, not a guarantee.

### Per-crate publish loop

```bash
for crate in voirs-g2p voirs-vocoder voirs-dataset voirs-singing voirs-spatial \
             voirs-acoustic \
             voirs-cloning voirs-emotion \
             voirs-conversion \
             voirs-sdk \
             voirs-recognizer \
             voirs-evaluation voirs-ffi \
             voirs-feedback voirs-cli; do
  cargo package -p "$crate" --no-verify --allow-dirty || break
  # once the previous wave's crates are confirmed live on crates.io:
  # cargo publish -p "$crate"
done
```

`--no-verify` skips cargo's own build-during-package step, which is useful
for a fast pre-flight (`cargo package` will still fail immediately if the
internal-dependency version requirement can't be satisfied against
crates.io, independent of `--no-verify`). Do a real `cargo publish` (without
`--no-verify`) for the actual release.

**Verified current blocker (reproducible today):**

```
$ cargo package -p voirs-g2p --no-verify --allow-dirty
   Packaging voirs-g2p v0.1.0 ...
   Packaged 118 files, 2.2MiB (435.4KiB compressed)      # OK: leaf crate

$ cargo package -p voirs-acoustic --no-verify --allow-dirty
error: failed to prepare local package for uploading
Caused by:
  failed to select a version for the requirement `voirs-g2p = "^0.1.0"`
  candidate versions found which didn't match: 0.1.0-rc.1, 0.1.0-beta.1, 0.1.0-alpha.3, ...
```

This is expected and not a code defect: `voirs-g2p` (and the other wave-1/2
crates) have not yet been published as final `0.1.0` to crates.io, only
prerelease versions exist there today. It resolves itself automatically once
each wave is actually published in order — this is a process/sequencing
constraint, not something fixable in the workspace `Cargo.toml`.

## 3. GPU / CUDA feature caveat

`candle-kernels` and `cudarc` (transitive deps of `candle-core`'s `cuda`
feature, reachable via `voirs-acoustic --features cuda` and similar) are
patched locally to non-panicking stubs:

```toml
# root Cargo.toml
[patch.crates-io]
candle-kernels = { path = "patches/candle-kernels" }
cudarc = { path = "patches/cudarc" }
```

**`[patch.crates-io]` does not propagate through `cargo publish`** — it only
applies within this workspace checkout. Anyone who builds a *published*
VoiRS crate with `--features cuda`/`gpu` or `--all-features` on a machine
without the CUDA toolchain installed will get the real, unpatched
`candle-kernels`/`cudarc`, whose `build.rs` scripts panic at build time
(not merely at runtime) when `nvidia-smi`/`nvcc` are absent.

Until this is fixed upstream (candle-kernels/cudarc making their build
scripts CUDA-optional), documented `--all-features` builds of the
**published** crates are only reliably buildable on a machine with a CUDA
toolchain, contrary to the impression an unqualified "`--all-features`"
recommendation gives. Local workspace builds (this checkout) are unaffected
by this caveat — the patches apply automatically via `[patch.crates-io]`.

## 4. After publishing

- Verify each crate on `docs.rs` builds (docs.rs does **not** see
  `[patch.crates-io]` either, so any doc-only code path gated behind `cuda`
  needs `#[cfg(docsrs)]` handling if it would otherwise try to compile
  candle-kernels/cudarc for docs.rs's build).
- Update `CHANGELOG.md` and tag the release per the branch policy in
  `CLAUDE.md` ("Availability of X.Y.Z" commit on `master`/`main`).

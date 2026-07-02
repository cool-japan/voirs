//! Real asymmetric-cryptography signature verification for the VoiRS CLI
//! self-updater.
//!
//! This module used to contain (in `update.rs`) a `simulate_signature_verification`
//! helper that computed `SHA256(sha256(binary) || public_key)` and compared a
//! 16-byte prefix against the "signature" — a check built entirely from
//! PUBLIC, attacker-derivable inputs (the hash of the downloaded file and a
//! public key). It was not a signature check at all: anyone could construct a
//! payload that "passed" it without ever holding a private key. Everything in
//! this module replaces that with genuine asymmetric verification using the
//! pure-Rust RustCrypto crates (`ed25519-dalek`, `rsa`, `p256`), matching the
//! project's documented preference for pure-Rust RustCrypto over `ring`/
//! OpenSSL-backed alternatives (see e.g. `voirs-cloning::consent_crypto`).
//!
//! ## Message convention
//!
//! All three algorithms below verify the signature directly over the raw
//! update-package bytes (`data`), *not* over a separately computed SHA-256
//! digest. This matches the native behavior of each underlying crate:
//! Ed25519 hashes the message internally with SHA-512 as part of the EdDSA
//! algorithm, `rsa::pkcs1v15::VerifyingKey<Sha256>` hashes the message with
//! SHA-256 internally before checking the PKCS#1v1.5 padding (equivalent to
//! `openssl dgst -sha256 -sign`), and `p256::ecdsa::VerifyingKey` hashes the
//! message with SHA-256 internally as required by NIST P-256 ECDSA. Passing
//! the raw bytes avoids any risk of accidentally double-hashing or of the
//! caller and the signer disagreeing about what was actually hashed.
//!
//! ## CRITICAL OPERATIONAL GAP — read before enabling self-update in production
//!
//! [`embedded_public_key`] intentionally returns `None` for every algorithm.
//! **No real VoiRS project signing keypair has been generated or provisioned
//! yet.** The previous code filled this slot with obviously-fake filler bytes
//! (`0x11, 0x22, 0x33, ...`); doing that again — even with different-looking
//! bytes — would silently reintroduce the exact vulnerability this module
//! fixes, because "verification" against a key nobody's private half
//! corresponds to a real signer is just as fake as no verification at all.
//!
//! Instead, when no key is configured from any source (embedded constant,
//! `VOIRS_PUBLIC_KEY` env var, or `UpdateConfig::public_key_path`),
//! [`crate::packaging::update::UpdateManager`] fails **closed**: signature
//! verification returns `Err`, which propagates out of
//! `UpdateManager::perform_update` and aborts the update *before* the running
//! binary is ever replaced (see `get_verification_public_key` in
//! `update.rs`).
//!
//! Before self-update can be safely enabled in a real deployment, a
//! maintainer must:
//!
//! 1. Generate a real keypair **offline**, on a machine that is not the build
//!    server, e.g.:
//!    - Ed25519: `ed25519_dalek::SigningKey::generate` (or `openssl genpkey
//!      -algorithm ed25519`), or
//!    - ECDSA P-256: `openssl ecparam -genkey -name prime256v1`, or
//!    - RSA-2048+: `openssl genrsa 3072`.
//! 2. Keep the private key offline (ideally in an HSM or an air-gapped
//!    signer used only to sign release artifacts) — it must never be
//!    committed to this repository or embedded in the binary.
//! 3. Embed the PUBLIC half of the keypair by either:
//!    - editing [`embedded_public_key`] below to return
//!      `Some(include_bytes!("../../keys/update-signing-key.pub"))` (raw
//!      bytes: 32 for Ed25519, SEC1 for ECDSA, DER for RSA), or
//!    - shipping the key out-of-band via the `VOIRS_PUBLIC_KEY` environment
//!      variable or `UpdateConfig::public_key_path` at deploy time (no code
//!      change required).
//! 4. Wire up real signature delivery: `UpdateManager::fetch_latest_version`
//!    currently never populates `VersionInfo::signature` (release-asset
//!    fetching is out of scope for this change — see the `TODO` there). A
//!    real `.sig` file (or a `signature` field in the release manifest) must
//!    be fetched and threaded through before self-update is end-to-end
//!    functional; until then, `perform_update` will (correctly) refuse to
//!    apply updates whenever `UpdateConfig::verify_signatures` is `true`
//!    (the default), because there is nothing genuine to verify against.

use anyhow::{anyhow, Result};

/// Ed25519 signature length in bytes (fixed-size `r || s`-style encoding
/// specific to Ed25519, i.e. a 32-byte `R` point followed by a 32-byte `s`
/// scalar).
pub const ED25519_SIGNATURE_LEN: usize = 64;

/// Ed25519 public key length in bytes.
pub const ED25519_PUBLIC_KEY_LEN: usize = 32;

/// Verify a real Ed25519 signature over `data` using `public_key`.
///
/// * `data` — the exact bytes that were signed (the full update package).
/// * `signature` — the raw 64-byte Ed25519 signature.
/// * `public_key` — the raw 32-byte Ed25519 public key.
///
/// Returns `Ok(true)`/`Ok(false)` for a well-formed signature/key pair that
/// cryptographically does/doesn't match; returns `Err` only when the
/// signature or key bytes are structurally malformed (wrong length or not a
/// valid curve point), since that indicates corrupted input rather than a
/// legitimately-failed verification.
pub fn verify_ed25519(data: &[u8], signature: &[u8], public_key: &[u8]) -> Result<bool> {
    use ed25519_dalek::{Signature, Verifier, VerifyingKey};

    let key_bytes: [u8; ED25519_PUBLIC_KEY_LEN] = public_key.try_into().map_err(|_| {
        anyhow!(
            "Ed25519 public key must be exactly {} bytes, got {}",
            ED25519_PUBLIC_KEY_LEN,
            public_key.len()
        )
    })?;
    let verifying_key = VerifyingKey::from_bytes(&key_bytes)
        .map_err(|e| anyhow!("invalid Ed25519 public key: {e}"))?;

    let sig_bytes: [u8; ED25519_SIGNATURE_LEN] = signature.try_into().map_err(|_| {
        anyhow!(
            "Ed25519 signature must be exactly {} bytes, got {}",
            ED25519_SIGNATURE_LEN,
            signature.len()
        )
    })?;
    let sig = Signature::from_bytes(&sig_bytes);

    Ok(verifying_key.verify(data, &sig).is_ok())
}

/// Verify a real RSASSA-PKCS1-v1_5 (SHA-256) signature over `data`.
///
/// * `data` — the exact bytes that were signed.
/// * `signature` — the raw PKCS#1v1.5 signature bytes (big-endian integer,
///   same byte length as the RSA modulus).
/// * `public_key_der` — the RSA public key, DER-encoded either as SPKI
///   (`SubjectPublicKeyInfo`, i.e. what `openssl rsa -pubout` produces) or as
///   a bare PKCS#1 `RSAPublicKey`. Both encodings are tried.
///
/// Returns `Ok(true)`/`Ok(false)` for structurally valid inputs depending on
/// whether the signature cryptographically checks out; returns `Err` when
/// the key or signature cannot be parsed at all.
pub fn verify_rsa_pkcs1v15_sha256(
    data: &[u8],
    signature: &[u8],
    public_key_der: &[u8],
) -> Result<bool> {
    use rsa::pkcs1::DecodeRsaPublicKey;
    use rsa::pkcs1v15::{Signature as RsaSignature, VerifyingKey};
    use rsa::pkcs8::DecodePublicKey;
    // Use the `Sha256` re-exported by the `rsa` crate itself (rather than
    // this workspace's top-level `sha2` dependency) so the `Digest` impl
    // used for the generic `VerifyingKey<D>` parameter is guaranteed to be
    // the exact version `rsa` 0.9.x's trait bounds were compiled against —
    // the RustCrypto ecosystem currently has sha2 0.10.x and 0.11.x
    // coexisting in this workspace's dependency graph, which are different,
    // incompatible major `digest` versions.
    use rsa::sha2::Sha256 as RsaSha256;
    use rsa::signature::Verifier as RsaVerifier;
    use rsa::RsaPublicKey;

    let public_key = RsaPublicKey::from_public_key_der(public_key_der)
        .or_else(|_| RsaPublicKey::from_pkcs1_der(public_key_der))
        .map_err(|e| anyhow!("invalid RSA public key (tried SPKI and PKCS#1 DER): {e}"))?;

    let verifying_key = VerifyingKey::<RsaSha256>::new(public_key);

    let sig = RsaSignature::try_from(signature)
        .map_err(|e| anyhow!("invalid RSA PKCS#1v1.5 signature encoding: {e}"))?;

    Ok(verifying_key.verify(data, &sig).is_ok())
}

/// Verify a real ECDSA P-256 (secp256r1, SHA-256) signature over `data`.
///
/// * `data` — the exact bytes that were signed.
/// * `signature` — either the raw fixed-size 64-byte `r || s` encoding, or a
///   DER-encoded `ECDSA-Sig-Value` sequence. Both are tried.
/// * `public_key` — the SEC1-encoded public key (33-byte compressed or
///   65-byte uncompressed point).
///
/// Returns `Ok(true)`/`Ok(false)` for structurally valid inputs depending on
/// whether the signature cryptographically checks out; returns `Err` when
/// the key or signature cannot be parsed at all.
pub fn verify_ecdsa_p256_sha256(data: &[u8], signature: &[u8], public_key: &[u8]) -> Result<bool> {
    use p256::ecdsa::signature::Verifier as EcdsaVerifier;
    use p256::ecdsa::{Signature, VerifyingKey};

    let verifying_key = VerifyingKey::from_sec1_bytes(public_key)
        .map_err(|e| anyhow!("invalid ECDSA P-256 public key (expected SEC1 bytes): {e}"))?;

    let sig = Signature::try_from(signature)
        .or_else(|_| Signature::from_der(signature))
        .map_err(|e| {
            anyhow!("invalid ECDSA P-256 signature encoding (tried raw r||s and DER): {e}")
        })?;

    Ok(verifying_key.verify(data, &sig).is_ok())
}

/// Embedded public key material for update signature verification, keyed by
/// the configured algorithm name (`"ed25519"`, `"rsa"`, or `"ecdsa"`).
///
/// This intentionally returns `None` for every algorithm right now — see the
/// module-level "CRITICAL OPERATIONAL GAP" doc comment above for exactly why
/// and what a maintainer needs to do before this can hold a real key.
/// `UpdateManager::get_verification_public_key` treats `None` here (combined
/// with no env var / config-file key) as "no key configured" and fails
/// closed rather than falling back to any compiled-in default.
///
/// # DO NOT
///
/// Do not "fix" a failing test or a blocked update by putting placeholder
/// bytes here. A key nobody holds the private half of provides zero
/// security and reintroduces the fabricated-verification vulnerability this
/// module exists to close.
pub fn embedded_public_key(algorithm: &str) -> Option<&'static [u8]> {
    match algorithm {
        "ed25519" | "rsa" | "ecdsa" => None,
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Fills `len` bytes using an OS-backed CSPRNG (`rand_core::OsRng`).
    ///
    /// Used only to generate ephemeral test keypairs / seeds; nothing here
    /// runs in production code.
    fn os_random_bytes(len: usize) -> Vec<u8> {
        use rand_core::{OsRng, RngCore};
        let mut buf = vec![0_u8; len];
        OsRng.fill_bytes(&mut buf);
        buf
    }

    // ---------------------------------------------------------------
    // Ed25519
    // ---------------------------------------------------------------

    #[test]
    fn ed25519_accepts_genuine_signature() {
        use ed25519_dalek::{Signer, SigningKey};

        let seed: [u8; 32] = os_random_bytes(32)
            .try_into()
            .expect("os_random_bytes(32) always returns 32 bytes");
        let signing_key = SigningKey::from_bytes(&seed);
        let verifying_key = signing_key.verifying_key();

        let message = b"voirs-cli-update-package-v0.4.0-real-bytes";
        let signature = signing_key.sign(message);

        let is_valid = verify_ed25519(message, &signature.to_bytes(), verifying_key.as_bytes())
            .expect("well-formed genuine signature must verify without error");

        assert!(is_valid, "genuine Ed25519 signature must be accepted");
    }

    #[test]
    fn ed25519_rejects_tampered_message() {
        use ed25519_dalek::{Signer, SigningKey};

        let seed: [u8; 32] = os_random_bytes(32)
            .try_into()
            .expect("os_random_bytes(32) always returns 32 bytes");
        let signing_key = SigningKey::from_bytes(&seed);
        let verifying_key = signing_key.verifying_key();

        let original_message = b"voirs-cli-update-package-v0.4.0".to_vec();
        let signature = signing_key.sign(&original_message);

        let mut tampered_message = original_message.clone();
        let last = tampered_message.len() - 1;
        tampered_message[last] ^= 0xFF;

        let is_valid = verify_ed25519(
            &tampered_message,
            &signature.to_bytes(),
            verifying_key.as_bytes(),
        )
        .expect("well-formed signature/key must produce a verdict, not an error");

        assert!(
            !is_valid,
            "signature for the original message must NOT verify against tampered bytes"
        );
    }

    #[test]
    fn ed25519_rejects_forged_signature() {
        use ed25519_dalek::{Signer, SigningKey};

        let seed: [u8; 32] = os_random_bytes(32)
            .try_into()
            .expect("os_random_bytes(32) always returns 32 bytes");
        let signing_key = SigningKey::from_bytes(&seed);
        let verifying_key = signing_key.verifying_key();

        let message = b"voirs-cli-update-package-v0.4.0";
        let mut forged_signature = signing_key.sign(message).to_bytes();
        forged_signature[0] ^= 0x01;

        let is_valid = verify_ed25519(message, &forged_signature, verifying_key.as_bytes())
            .expect("well-formed signature/key must produce a verdict, not an error");

        assert!(!is_valid, "bit-flipped signature must NOT verify");
    }

    #[test]
    fn ed25519_rejects_signature_from_a_different_key() {
        use ed25519_dalek::{Signer, SigningKey};

        let seed_a: [u8; 32] = os_random_bytes(32)
            .try_into()
            .expect("os_random_bytes(32) always returns 32 bytes");
        let seed_b: [u8; 32] = os_random_bytes(32)
            .try_into()
            .expect("os_random_bytes(32) always returns 32 bytes");
        let signing_key_a = SigningKey::from_bytes(&seed_a);
        let signing_key_b = SigningKey::from_bytes(&seed_b);

        let message = b"voirs-cli-update-package-v0.4.0";
        let signature_from_a = signing_key_a.sign(message);

        // Verify signature produced by key A against key B's public key.
        let is_valid = verify_ed25519(
            message,
            &signature_from_a.to_bytes(),
            signing_key_b.verifying_key().as_bytes(),
        )
        .expect("well-formed signature/key must produce a verdict, not an error");

        assert!(
            !is_valid,
            "signature made by one keypair must not verify under a different keypair's public key"
        );
    }

    #[test]
    fn ed25519_rejects_malformed_public_key_length() {
        let result = verify_ed25519(b"data", &[0_u8; ED25519_SIGNATURE_LEN], &[0_u8; 4]);
        assert!(
            result.is_err(),
            "a public key of the wrong length must be a hard error, not a false verdict"
        );
    }

    #[test]
    fn ed25519_rejects_malformed_signature_length() {
        let key_bytes = os_random_bytes(ED25519_PUBLIC_KEY_LEN);
        let result = verify_ed25519(b"data", &[0_u8; 4], &key_bytes);
        assert!(
            result.is_err(),
            "a signature of the wrong length must be a hard error, not a false verdict"
        );
    }

    // ---------------------------------------------------------------
    // RSA (PKCS#1 v1.5, SHA-256)
    // ---------------------------------------------------------------

    /// RSA key generation needs a real `CryptoRngCore`; `OsRng` (from
    /// `rand_core`, a dev-dependency here) satisfies that directly.
    fn generate_test_rsa_keypair() -> (rsa::RsaPrivateKey, rsa::RsaPublicKey) {
        use rand_core::OsRng;
        use rsa::{RsaPrivateKey, RsaPublicKey};

        // 2048 bits is the minimum realistic size; kept as small as
        // reasonable to keep test runtime low while still exercising real
        // RSA math end-to-end.
        let private_key =
            RsaPrivateKey::new(&mut OsRng, 2048).expect("RSA-2048 keygen must succeed with OsRng");
        let public_key = RsaPublicKey::from(&private_key);
        (private_key, public_key)
    }

    #[test]
    fn rsa_accepts_genuine_signature() {
        use rsa::pkcs1v15::SigningKey;
        use rsa::pkcs8::EncodePublicKey;
        use rsa::sha2::Sha256 as RsaSha256;
        use rsa::signature::{SignatureEncoding, Signer as RsaSigner};

        let (private_key, public_key) = generate_test_rsa_keypair();
        let signing_key = SigningKey::<RsaSha256>::new(private_key);

        let message = b"voirs-cli-update-package-v0.4.0-rsa";
        let signature = signing_key.sign(message);

        let public_key_der = public_key
            .to_public_key_der()
            .expect("encoding a freshly generated RSA public key as SPKI DER must succeed");

        let is_valid =
            verify_rsa_pkcs1v15_sha256(message, &signature.to_bytes(), public_key_der.as_bytes())
                .expect("well-formed genuine RSA signature must verify without error");

        assert!(
            is_valid,
            "genuine RSA PKCS#1v1.5 signature must be accepted"
        );
    }

    #[test]
    fn rsa_rejects_tampered_message() {
        use rsa::pkcs1v15::SigningKey;
        use rsa::pkcs8::EncodePublicKey;
        use rsa::sha2::Sha256 as RsaSha256;
        use rsa::signature::{SignatureEncoding, Signer as RsaSigner};

        let (private_key, public_key) = generate_test_rsa_keypair();
        let signing_key = SigningKey::<RsaSha256>::new(private_key);

        let message = b"voirs-cli-update-package-v0.4.0-rsa".to_vec();
        let signature = signing_key.sign(&message);
        let public_key_der = public_key
            .to_public_key_der()
            .expect("encoding a freshly generated RSA public key as SPKI DER must succeed");

        let mut tampered = message.clone();
        let last = tampered.len() - 1;
        tampered[last] ^= 0xFF;

        let is_valid =
            verify_rsa_pkcs1v15_sha256(&tampered, &signature.to_bytes(), public_key_der.as_bytes())
                .expect("well-formed signature/key must produce a verdict, not an error");

        assert!(
            !is_valid,
            "RSA signature for original message must NOT verify against tampered bytes"
        );
    }

    #[test]
    fn rsa_rejects_forged_signature() {
        use rsa::pkcs1v15::SigningKey;
        use rsa::pkcs8::EncodePublicKey;
        use rsa::sha2::Sha256 as RsaSha256;
        use rsa::signature::{SignatureEncoding, Signer as RsaSigner};

        let (private_key, public_key) = generate_test_rsa_keypair();
        let signing_key = SigningKey::<RsaSha256>::new(private_key);

        let message = b"voirs-cli-update-package-v0.4.0-rsa";
        let mut forged_signature = signing_key.sign(message).to_vec();
        let last = forged_signature.len() - 1;
        forged_signature[last] ^= 0x01;

        let public_key_der = public_key
            .to_public_key_der()
            .expect("encoding a freshly generated RSA public key as SPKI DER must succeed");

        let is_valid =
            verify_rsa_pkcs1v15_sha256(message, &forged_signature, public_key_der.as_bytes())
                .expect("well-formed signature/key must produce a verdict, not an error");

        assert!(!is_valid, "bit-flipped RSA signature must NOT verify");
    }

    // ---------------------------------------------------------------
    // ECDSA P-256 (secp256r1, SHA-256)
    // ---------------------------------------------------------------

    /// P-256 keys can be constructed directly from a random 32-byte scalar
    /// seed with `SigningKey::from_slice`, no `CryptoRngCore` needed — the
    /// same "raw random bytes -> deterministic key" pattern already used by
    /// `voirs-cloning::consent_crypto` for Ed25519.
    fn generate_test_p256_keypair() -> p256::ecdsa::SigningKey {
        let seed = os_random_bytes(32);
        p256::ecdsa::SigningKey::from_slice(&seed).expect(
            "a uniformly random 32-byte seed is a valid P-256 scalar with overwhelming probability",
        )
    }

    #[test]
    fn ecdsa_accepts_genuine_signature() {
        use p256::ecdsa::signature::Signer as EcdsaSigner;
        use p256::ecdsa::{Signature, VerifyingKey};

        let signing_key = generate_test_p256_keypair();
        let verifying_key = VerifyingKey::from(&signing_key);

        let message = b"voirs-cli-update-package-v0.4.0-ecdsa";
        let signature: Signature = signing_key.sign(message);

        let is_valid = verify_ecdsa_p256_sha256(
            message,
            &signature.to_bytes(),
            verifying_key.to_encoded_point(false).as_bytes(),
        )
        .expect("well-formed genuine ECDSA signature must verify without error");

        assert!(is_valid, "genuine ECDSA P-256 signature must be accepted");
    }

    #[test]
    fn ecdsa_rejects_tampered_message() {
        use p256::ecdsa::signature::Signer as EcdsaSigner;
        use p256::ecdsa::{Signature, VerifyingKey};

        let signing_key = generate_test_p256_keypair();
        let verifying_key = VerifyingKey::from(&signing_key);

        let message = b"voirs-cli-update-package-v0.4.0-ecdsa".to_vec();
        let signature: Signature = signing_key.sign(&message);

        let mut tampered = message.clone();
        let last = tampered.len() - 1;
        tampered[last] ^= 0xFF;

        let is_valid = verify_ecdsa_p256_sha256(
            &tampered,
            &signature.to_bytes(),
            verifying_key.to_encoded_point(false).as_bytes(),
        )
        .expect("well-formed signature/key must produce a verdict, not an error");

        assert!(
            !is_valid,
            "ECDSA signature for original message must NOT verify against tampered bytes"
        );
    }

    #[test]
    fn ecdsa_rejects_forged_signature() {
        use p256::ecdsa::signature::SignatureEncoding;
        use p256::ecdsa::signature::Signer as EcdsaSigner;
        use p256::ecdsa::{Signature, VerifyingKey};

        let signing_key = generate_test_p256_keypair();
        let verifying_key = VerifyingKey::from(&signing_key);

        let message = b"voirs-cli-update-package-v0.4.0-ecdsa";
        let signature: Signature = signing_key.sign(message);
        let mut forged_bytes = signature.to_vec();
        let last = forged_bytes.len() - 1;
        forged_bytes[last] ^= 0x01;

        let is_valid = verify_ecdsa_p256_sha256(
            message,
            &forged_bytes,
            verifying_key.to_encoded_point(false).as_bytes(),
        )
        .expect("well-formed signature/key must produce a verdict, not an error");

        assert!(!is_valid, "bit-flipped ECDSA signature must NOT verify");
    }

    // ---------------------------------------------------------------
    // Fail-closed: no real key provisioned
    // ---------------------------------------------------------------

    #[test]
    fn embedded_public_key_is_absent_for_every_known_algorithm() {
        // This is the load-bearing assertion for the "fail closed until a
        // maintainer provisions a real key" behavior: as long as this stays
        // `None`, `UpdateManager::get_verification_public_key` cannot silently
        // fall back to a fabricated "verified" result when no real key has
        // been configured via env var or config file.
        assert_eq!(embedded_public_key("ed25519"), None);
        assert_eq!(embedded_public_key("rsa"), None);
        assert_eq!(embedded_public_key("ecdsa"), None);
        assert_eq!(embedded_public_key("totally-unknown-algorithm"), None);
    }
}

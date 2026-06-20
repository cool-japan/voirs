//! Encrypted Inference for privacy-preserving predictions.
//!
//! This module provides homomorphic encryption capabilities for performing
//! inference on encrypted data without decrypting it. This enables secure
//! cloud-based inference where the server cannot access the input data.

use scirs2_core::random::{thread_rng, Dimension, Distribution, Rng};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Encrypted inference error types
#[derive(Debug, Error)]
pub enum EncryptionError {
    /// Encryption operation failed
    #[error("Encryption failed: {0}")]
    EncryptionFailed(String),

    /// Decryption operation failed
    #[error("Decryption failed: {0}")]
    DecryptionFailed(String),

    /// Invalid or corrupted encryption key
    #[error("Invalid key: {0}")]
    InvalidKey(String),

    /// Ciphertext incompatible with operation
    #[error("Incompatible ciphertext: {0}")]
    IncompatibleCiphertext(String),

    /// Configuration error
    #[error("Configuration error: {0}")]
    ConfigError(String),

    /// Unsupported encryption scheme
    #[error("Scheme not supported: {0}")]
    UnsupportedScheme(String),
}

/// Encryption scheme types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EncryptionScheme {
    /// Paillier homomorphic encryption (additive)
    Paillier,
    /// `ElGamal` encryption (multiplicative)
    ElGamal,
    /// CKKS (approximate homomorphic encryption for real numbers)
    CKKS,
    /// BGV (integer arithmetic)
    BGV,
    /// Mock encryption for testing (no actual encryption)
    Mock,
}

/// Secure inference configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecureInferenceConfig {
    /// Encryption scheme to use
    pub scheme: EncryptionScheme,

    /// Key size in bits
    pub key_size: usize,

    /// Precision for approximate schemes (CKKS)
    pub precision_bits: Option<usize>,

    /// Enable batch encoding
    pub enable_batching: bool,

    /// Maximum circuit depth (for leveled HE)
    pub max_depth: Option<usize>,
}

impl Default for SecureInferenceConfig {
    fn default() -> Self {
        Self {
            scheme: EncryptionScheme::Mock,
            key_size: 2048,
            precision_bits: Some(20),
            enable_batching: true,
            max_depth: Some(10),
        }
    }
}

/// Public key for encryption
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PublicKey {
    /// Modulus n (for Paillier, RSA-like)
    pub n: Vec<u8>,

    /// Generator g
    pub g: Vec<u8>,

    /// Key size
    pub size: usize,

    /// Encryption scheme
    pub scheme: EncryptionScheme,
}

/// Private key for decryption
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivateKey {
    /// Lambda (Carmichael's totient)
    pub lambda: Vec<u8>,

    /// Mu (modular inverse)
    pub mu: Vec<u8>,

    /// Key size
    pub size: usize,

    /// Encryption scheme
    pub scheme: EncryptionScheme,
}

/// Encrypted value (ciphertext)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ciphertext {
    /// Encrypted data
    pub data: Vec<u8>,

    /// Encryption scheme used
    pub scheme: EncryptionScheme,

    /// Metadata for batched encoding
    pub batch_size: Option<usize>,
}

impl Ciphertext {
    /// Create new ciphertext
    #[must_use]
    pub fn new(data: Vec<u8>, scheme: EncryptionScheme) -> Self {
        Self {
            data,
            scheme,
            batch_size: None,
        }
    }

    /// Get ciphertext size in bytes
    #[must_use]
    pub fn size(&self) -> usize {
        self.data.len()
    }
}

/// Homomorphic encryption interface
pub trait HomomorphicEncryption: Send + Sync {
    /// Encrypt a plaintext value
    fn encrypt(
        &self,
        plaintext: f32,
        public_key: &PublicKey,
    ) -> Result<Ciphertext, EncryptionError>;

    /// Decrypt a ciphertext
    fn decrypt(
        &self,
        ciphertext: &Ciphertext,
        private_key: &PrivateKey,
    ) -> Result<f32, EncryptionError>;

    /// Homomorphic addition
    fn add(&self, c1: &Ciphertext, c2: &Ciphertext) -> Result<Ciphertext, EncryptionError>;

    /// Homomorphic multiplication
    fn multiply(&self, c1: &Ciphertext, c2: &Ciphertext) -> Result<Ciphertext, EncryptionError>;

    /// Scalar multiplication (multiply ciphertext by plaintext)
    fn scalar_multiply(
        &self,
        ciphertext: &Ciphertext,
        scalar: f32,
    ) -> Result<Ciphertext, EncryptionError>;

    /// Get encryption scheme
    fn scheme(&self) -> EncryptionScheme;
}

/// Mock encryption for testing (no real encryption)
#[derive(Debug, Clone)]
pub struct MockEncryption {
    scheme: EncryptionScheme,
}

impl MockEncryption {
    /// Create a new mock encryption instance for testing
    #[must_use]
    pub fn new() -> Self {
        Self {
            scheme: EncryptionScheme::Mock,
        }
    }

    fn f32_to_bytes(&self, value: f32) -> Vec<u8> {
        value.to_le_bytes().to_vec()
    }

    fn bytes_to_f32(&self, bytes: &[u8]) -> Result<f32, EncryptionError> {
        if bytes.len() != 4 {
            return Err(EncryptionError::DecryptionFailed(
                "Invalid byte length for f32".to_string(),
            ));
        }

        let mut array = [0u8; 4];
        array.copy_from_slice(bytes);
        Ok(f32::from_le_bytes(array))
    }
}

impl Default for MockEncryption {
    fn default() -> Self {
        Self::new()
    }
}

impl HomomorphicEncryption for MockEncryption {
    fn encrypt(
        &self,
        plaintext: f32,
        _public_key: &PublicKey,
    ) -> Result<Ciphertext, EncryptionError> {
        Ok(Ciphertext::new(self.f32_to_bytes(plaintext), self.scheme))
    }

    fn decrypt(
        &self,
        ciphertext: &Ciphertext,
        _private_key: &PrivateKey,
    ) -> Result<f32, EncryptionError> {
        self.bytes_to_f32(&ciphertext.data)
    }

    fn add(&self, c1: &Ciphertext, c2: &Ciphertext) -> Result<Ciphertext, EncryptionError> {
        let v1 = self.bytes_to_f32(&c1.data)?;
        let v2 = self.bytes_to_f32(&c2.data)?;
        Ok(Ciphertext::new(self.f32_to_bytes(v1 + v2), self.scheme))
    }

    fn multiply(&self, c1: &Ciphertext, c2: &Ciphertext) -> Result<Ciphertext, EncryptionError> {
        let v1 = self.bytes_to_f32(&c1.data)?;
        let v2 = self.bytes_to_f32(&c2.data)?;
        Ok(Ciphertext::new(self.f32_to_bytes(v1 * v2), self.scheme))
    }

    fn scalar_multiply(
        &self,
        ciphertext: &Ciphertext,
        scalar: f32,
    ) -> Result<Ciphertext, EncryptionError> {
        let v = self.bytes_to_f32(&ciphertext.data)?;
        Ok(Ciphertext::new(self.f32_to_bytes(v * scalar), self.scheme))
    }

    fn scheme(&self) -> EncryptionScheme {
        self.scheme
    }
}

/// Encrypted inference engine
pub struct EncryptedInference {
    /// Configuration
    config: SecureInferenceConfig,

    /// Homomorphic encryption implementation
    he: Box<dyn HomomorphicEncryption>,

    /// Public key
    public_key: PublicKey,

    /// Private key (optional, only for key owner)
    private_key: Option<PrivateKey>,
}

impl EncryptedInference {
    /// Create new encrypted inference engine
    pub fn new(config: SecureInferenceConfig) -> Result<Self, EncryptionError> {
        let he: Box<dyn HomomorphicEncryption> = match config.scheme {
            EncryptionScheme::Mock => Box::new(MockEncryption::new()),
            EncryptionScheme::Paillier => paillier_encryption_or_err(&config)?,
            EncryptionScheme::ElGamal => elgamal_encryption_or_err(&config)?,
            EncryptionScheme::CKKS => {
                return Err(EncryptionError::UnsupportedScheme(
                    "CKKS not yet implemented, use Mock for now".to_string(),
                ));
            }
            EncryptionScheme::BGV => {
                return Err(EncryptionError::UnsupportedScheme(
                    "BGV not yet implemented, use Mock for now".to_string(),
                ));
            }
        };

        // Generate keys (mock implementation)
        let (public_key, private_key) = Self::generate_keypair(config.scheme, config.key_size)?;

        Ok(Self {
            config,
            he,
            public_key,
            private_key: Some(private_key),
        })
    }

    /// Generate encryption key pair
    fn generate_keypair(
        scheme: EncryptionScheme,
        key_size: usize,
    ) -> Result<(PublicKey, PrivateKey), EncryptionError> {
        // Dispatch to real keygen for schemes with homomorphic feature enabled
        #[cfg(feature = "homomorphic")]
        match scheme {
            EncryptionScheme::Paillier => return Ok(paillier_keygen_real(key_size)),
            EncryptionScheme::ElGamal => return Ok(elgamal_keygen_real(key_size)),
            _ => {}
        }

        // Mock implementation - generates random bytes for remaining schemes
        use scirs2_core::random::Rng;
        let mut rng = thread_rng();

        let n: Vec<u8> = (0..key_size / 8).map(|_| rng.random()).collect();
        let g: Vec<u8> = (0..key_size / 8).map(|_| rng.random()).collect();
        let lambda: Vec<u8> = (0..key_size / 8).map(|_| rng.random()).collect();
        let mu: Vec<u8> = (0..key_size / 8).map(|_| rng.random()).collect();

        let public_key = PublicKey {
            n,
            g,
            size: key_size,
            scheme,
        };

        let private_key = PrivateKey {
            lambda,
            mu,
            size: key_size,
            scheme,
        };

        Ok((public_key, private_key))
    }

    /// Encrypt plaintext value
    pub fn encrypt(&self, plaintext: f32) -> Result<Ciphertext, EncryptionError> {
        self.he.encrypt(plaintext, &self.public_key)
    }

    /// Encrypt vector of values
    pub fn encrypt_vec(&self, plaintexts: &[f32]) -> Result<Vec<Ciphertext>, EncryptionError> {
        plaintexts.iter().map(|&p| self.encrypt(p)).collect()
    }

    /// Decrypt ciphertext
    pub fn decrypt(&self, ciphertext: &Ciphertext) -> Result<f32, EncryptionError> {
        match &self.private_key {
            Some(sk) => self.he.decrypt(ciphertext, sk),
            None => Err(EncryptionError::InvalidKey(
                "No private key available".to_string(),
            )),
        }
    }

    /// Decrypt vector of ciphertexts
    pub fn decrypt_vec(&self, ciphertexts: &[Ciphertext]) -> Result<Vec<f32>, EncryptionError> {
        ciphertexts.iter().map(|c| self.decrypt(c)).collect()
    }

    /// Homomorphic matrix-vector multiplication
    pub fn encrypted_matvec(
        &self,
        matrix: &[Vec<f32>],
        encrypted_vec: &[Ciphertext],
    ) -> Result<Vec<Ciphertext>, EncryptionError> {
        let mut result = Vec::new();

        for row in matrix {
            let mut row_sum: Option<Ciphertext> = None;

            for (i, &weight) in row.iter().enumerate() {
                let weighted = self.he.scalar_multiply(&encrypted_vec[i], weight)?;

                row_sum = match row_sum {
                    None => Some(weighted),
                    Some(sum) => Some(self.he.add(&sum, &weighted)?),
                };
            }

            result.push(
                row_sum
                    .ok_or_else(|| EncryptionError::ConfigError("Empty matrix row".to_string()))?,
            );
        }

        Ok(result)
    }

    /// Encrypted neural network layer (linear transformation)
    pub fn encrypted_linear_layer(
        &self,
        weights: &[Vec<f32>],
        encrypted_input: &[Ciphertext],
        bias: Option<&[f32]>,
    ) -> Result<Vec<Ciphertext>, EncryptionError> {
        let mut output = self.encrypted_matvec(weights, encrypted_input)?;

        // Add bias if provided
        if let Some(bias_vec) = bias {
            for (i, encrypted_val) in output.iter_mut().enumerate() {
                if i < bias_vec.len() {
                    let encrypted_bias = self.encrypt(bias_vec[i])?;
                    *encrypted_val = self.he.add(encrypted_val, &encrypted_bias)?;
                }
            }
        }

        Ok(output)
    }

    /// Approximate activation function on encrypted data (polynomial approximation)
    pub fn encrypted_activation(
        &self,
        encrypted_input: &Ciphertext,
        activation_type: ActivationType,
    ) -> Result<Ciphertext, EncryptionError> {
        match activation_type {
            ActivationType::Linear => Ok(encrypted_input.clone()),
            ActivationType::Quadratic => {
                // x^2 approximation
                self.he.multiply(encrypted_input, encrypted_input)
            }
            ActivationType::PolynomialApproxReLU => {
                // ReLU approximation: 0.5x + 0.25x^2 (low-degree polynomial)
                let x = encrypted_input;
                let x2 = self.he.multiply(x, x)?;

                let term1 = self.he.scalar_multiply(x, 0.5)?;
                let term2 = self.he.scalar_multiply(&x2, 0.25)?;

                self.he.add(&term1, &term2)
            }
        }
    }

    /// Get public key for distribution to clients
    #[must_use]
    pub fn public_key(&self) -> &PublicKey {
        &self.public_key
    }

    /// Get encryption scheme
    #[must_use]
    pub fn scheme(&self) -> EncryptionScheme {
        self.config.scheme
    }
}

/// Activation function types for encrypted computation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivationType {
    /// Linear (identity)
    Linear,
    /// Quadratic (x^2)
    Quadratic,
    /// Polynomial approximation of `ReLU`
    PolynomialApproxReLU,
}

// ============================================================
// Feature-gated helper functions for scheme dispatch
// ============================================================

#[cfg(feature = "homomorphic")]
fn paillier_encryption_or_err(
    config: &SecureInferenceConfig,
) -> Result<Box<dyn HomomorphicEncryption>, EncryptionError> {
    Ok(Box::new(paillier::PaillierEncryption::new(
        config.precision_bits.unwrap_or(20) as u32,
    )))
}

#[cfg(not(feature = "homomorphic"))]
#[allow(dead_code)]
fn paillier_encryption_or_err(
    _config: &SecureInferenceConfig,
) -> Result<Box<dyn HomomorphicEncryption>, EncryptionError> {
    Err(EncryptionError::UnsupportedScheme(
        "Paillier requires the 'homomorphic' feature flag".to_string(),
    ))
}

#[cfg(feature = "homomorphic")]
fn elgamal_encryption_or_err(
    config: &SecureInferenceConfig,
) -> Result<Box<dyn HomomorphicEncryption>, EncryptionError> {
    Ok(Box::new(elgamal::ElGamalEncryption::new(
        config.precision_bits.unwrap_or(20) as u32,
    )))
}

#[cfg(not(feature = "homomorphic"))]
#[allow(dead_code)]
fn elgamal_encryption_or_err(
    _config: &SecureInferenceConfig,
) -> Result<Box<dyn HomomorphicEncryption>, EncryptionError> {
    Err(EncryptionError::UnsupportedScheme(
        "ElGamal requires the 'homomorphic' feature flag".to_string(),
    ))
}

#[cfg(feature = "homomorphic")]
fn paillier_keygen_real(key_size: usize) -> (PublicKey, PrivateKey) {
    let bits = key_size.max(256);
    let (pub_key, priv_key_raw) = paillier::generate_keypair(bits);
    // Encode n into priv_key.mu as: 4-byte LE n_len || n_bytes || mu_bytes
    // This allows decrypt to recover n from the private key
    let n_bytes = pub_key.n.clone();
    let mu_bytes = priv_key_raw.mu.clone();
    let mut encoded_mu = Vec::with_capacity(4 + n_bytes.len() + mu_bytes.len());
    encoded_mu.extend_from_slice(&(n_bytes.len() as u32).to_le_bytes());
    encoded_mu.extend_from_slice(&n_bytes);
    encoded_mu.extend_from_slice(&mu_bytes);
    let priv_key = PrivateKey {
        lambda: priv_key_raw.lambda,
        mu: encoded_mu,
        size: priv_key_raw.size,
        scheme: priv_key_raw.scheme,
    };
    (pub_key, priv_key)
}

#[cfg(feature = "homomorphic")]
fn elgamal_keygen_real(key_size: usize) -> (PublicKey, PrivateKey) {
    let (pub_key, priv_key_raw) = elgamal::generate_keypair(key_size);
    // Encode p into priv_key.lambda as: 4-byte LE p_len || p_bytes || x_bytes
    let p_bytes = pub_key.n.clone();
    let x_bytes = priv_key_raw.lambda.clone();
    let mut encoded_lambda = Vec::with_capacity(4 + p_bytes.len() + x_bytes.len());
    encoded_lambda.extend_from_slice(&(p_bytes.len() as u32).to_le_bytes());
    encoded_lambda.extend_from_slice(&p_bytes);
    encoded_lambda.extend_from_slice(&x_bytes);
    let priv_key = PrivateKey {
        lambda: encoded_lambda,
        mu: priv_key_raw.mu,
        size: priv_key_raw.size,
        scheme: priv_key_raw.scheme,
    };
    (pub_key, priv_key)
}

// ============================================================
// Paillier homomorphic encryption (additive)
// ============================================================

#[cfg(feature = "homomorphic")]
mod paillier {
    use super::{
        Ciphertext, EncryptionError, EncryptionScheme, HomomorphicEncryption, PrivateKey, PublicKey,
    };
    use num_bigint_dig::{BigInt, BigUint, RandPrime, ToBigUint};
    use num_bigint_dig::traits::ModInverse;

    pub fn gcd(a: &BigUint, b: &BigUint) -> BigUint {
        let mut a = a.clone();
        let mut b = b.clone();
        while b != BigUint::from(0u32) {
            let t = b.clone();
            b = a % &b;
            a = t;
        }
        a
    }

    pub fn l_func(x: &BigUint, n: &BigUint) -> BigUint {
        (x - BigUint::from(1u32)) / n
    }

    /// Returns (PublicKey, PrivateKey) where priv_key.mu = raw mu bytes (no n-prefix).
    /// The caller (paillier_keygen_real) adds the n-length prefix encoding.
    pub fn generate_keypair(bits: usize) -> (PublicKey, PrivateKey) {
        let mut rng = rand::thread_rng();
        let half_bits = bits / 2;

        let p: BigUint = rng.gen_prime(half_bits);
        let q: BigUint = loop {
            let candidate: BigUint = rng.gen_prime(half_bits);
            if candidate != p {
                break candidate;
            }
        };

        let n = &p * &q;
        let n_sq = &n * &n;
        let g = &n + BigUint::from(1u32);

        let p1 = &p - BigUint::from(1u32);
        let q1 = &q - BigUint::from(1u32);
        let phi = &p1 * &q1;
        let gcd_val = gcd(&p1, &q1);
        let lambda = &phi / &gcd_val;

        let g_lambda = g.modpow(&lambda, &n_sq);
        let l_val = l_func(&g_lambda, &n);
        // mod_inverse returns Option<BigInt> (signed). Normalise to positive residue mod n.
        use num_bigint_dig::Sign;
        let mu_big: BigInt = l_val
            .mod_inverse(&n)
            .unwrap_or_else(|| BigInt::from(1u32));
        let mu = if mu_big.sign() == Sign::Minus {
            // negative residue: add n to get canonical positive representative
            let n_big = BigInt::from(n.clone());
            (mu_big + n_big).to_biguint().unwrap_or_else(|| BigUint::from(1u32))
        } else {
            mu_big.to_biguint().unwrap_or_else(|| BigUint::from(1u32))
        };

        let pub_key = PublicKey {
            n: n.to_bytes_be(),
            g: g.to_bytes_be(),
            size: bits,
            scheme: EncryptionScheme::Paillier,
        };
        let priv_key = PrivateKey {
            lambda: lambda.to_bytes_be(),
            mu: mu.to_bytes_be(),
            size: bits,
            scheme: EncryptionScheme::Paillier,
        };
        (pub_key, priv_key)
    }

    fn f32_to_int(x: f32, precision_bits: u32) -> i64 {
        let scale = 1i64 << precision_bits;
        (x * scale as f32).round() as i64
    }

    fn biguint_to_u64_low(v: &BigUint) -> u64 {
        // Extract the lowest 8 bytes (little-endian) from big-endian representation.
        let bytes = v.to_bytes_le();
        let mut buf = [0u8; 8];
        let take = bytes.len().min(8);
        buf[..take].copy_from_slice(&bytes[..take]);
        u64::from_le_bytes(buf)
    }

    fn int_to_f32(x: &BigUint, n: &BigUint, precision_bits: u32) -> f32 {
        let scale = 1i64 << precision_bits;
        let half_n = n >> 1usize;
        let signed: i128 = if *x > half_n {
            let neg = n - x;
            -(biguint_to_u64_low(&neg) as i128)
        } else {
            biguint_to_u64_low(x) as i128
        };
        signed as f32 / scale as f32
    }

    pub struct PaillierEncryption {
        pub precision_bits: u32,
    }

    impl PaillierEncryption {
        pub fn new(precision_bits: u32) -> Self {
            Self { precision_bits }
        }
    }

    impl HomomorphicEncryption for PaillierEncryption {
        fn encrypt(
            &self,
            plaintext: f32,
            public_key: &PublicKey,
        ) -> Result<Ciphertext, EncryptionError> {
            let n = BigUint::from_bytes_be(&public_key.n);
            let g = BigUint::from_bytes_be(&public_key.g);
            let n_sq = &n * &n;

            let m_int = f32_to_int(plaintext, self.precision_bits);
            let m: BigUint = if m_int < 0 {
                let abs_val = BigUint::from(m_int.unsigned_abs());
                if abs_val >= n {
                    return Err(EncryptionError::EncryptionFailed(
                        "Value out of range for Paillier encryption".to_string(),
                    ));
                }
                &n - &abs_val
            } else {
                BigUint::from(m_int as u64)
            };

            let n_bytes = n.to_bytes_be();
            let r = loop {
                let rand_bytes: Vec<u8> = (0..n_bytes.len())
                    .map(|_| fastrand::u8(..))
                    .collect();
                let candidate = BigUint::from_bytes_be(&rand_bytes) % &n;
                if candidate > BigUint::from(0u32) {
                    break candidate;
                }
            };

            let gm = g.modpow(&m, &n_sq);
            let rn = r.modpow(&n, &n_sq);
            let c = (gm * rn) % &n_sq;

            Ok(Ciphertext {
                data: c.to_bytes_be(),
                scheme: EncryptionScheme::Paillier,
                batch_size: None,
            })
        }

        fn decrypt(
            &self,
            ciphertext: &Ciphertext,
            private_key: &PrivateKey,
        ) -> Result<f32, EncryptionError> {
            if ciphertext.scheme != EncryptionScheme::Paillier {
                return Err(EncryptionError::IncompatibleCiphertext(
                    "Expected Paillier ciphertext".to_string(),
                ));
            }
            // priv_key.mu encodes: 4-byte LE n_len || n_bytes || mu_bytes
            let mu_field = &private_key.mu;
            if mu_field.len() < 4 {
                return Err(EncryptionError::InvalidKey(
                    "Invalid Paillier private key encoding".to_string(),
                ));
            }
            let n_len =
                u32::from_le_bytes([mu_field[0], mu_field[1], mu_field[2], mu_field[3]]) as usize;
            if mu_field.len() < 4 + n_len {
                return Err(EncryptionError::InvalidKey(
                    "Truncated Paillier private key".to_string(),
                ));
            }
            let n = BigUint::from_bytes_be(&mu_field[4..4 + n_len]);
            let mu = BigUint::from_bytes_be(&mu_field[4 + n_len..]);
            let lambda = BigUint::from_bytes_be(&private_key.lambda);
            let n_sq = &n * &n;

            let c = BigUint::from_bytes_be(&ciphertext.data);
            let c_lambda = c.modpow(&lambda, &n_sq);
            let l_val = l_func(&c_lambda, &n);
            let m = (&l_val * &mu) % &n;

            Ok(int_to_f32(&m, &n, self.precision_bits))
        }

        fn add(&self, c1: &Ciphertext, c2: &Ciphertext) -> Result<Ciphertext, EncryptionError> {
            if c1.scheme != EncryptionScheme::Paillier || c2.scheme != EncryptionScheme::Paillier {
                return Err(EncryptionError::IncompatibleCiphertext(
                    "Both ciphertexts must use Paillier scheme".to_string(),
                ));
            }
            Err(EncryptionError::IncompatibleCiphertext(
                "Paillier homomorphic add requires public key context (n\u{00b2})".to_string(),
            ))
        }

        fn multiply(
            &self,
            _c1: &Ciphertext,
            _c2: &Ciphertext,
        ) -> Result<Ciphertext, EncryptionError> {
            Err(EncryptionError::IncompatibleCiphertext(
                "Paillier is additively homomorphic only".to_string(),
            ))
        }

        fn scalar_multiply(
            &self,
            _ciphertext: &Ciphertext,
            _scalar: f32,
        ) -> Result<Ciphertext, EncryptionError> {
            Err(EncryptionError::IncompatibleCiphertext(
                "Paillier scalar_multiply requires public key context".to_string(),
            ))
        }

        fn scheme(&self) -> EncryptionScheme {
            EncryptionScheme::Paillier
        }
    }
}

// ============================================================
// ElGamal homomorphic encryption (multiplicative)
// ============================================================

#[cfg(feature = "homomorphic")]
mod elgamal {
    use super::{
        Ciphertext, EncryptionError, EncryptionScheme, HomomorphicEncryption, PrivateKey, PublicKey,
    };
    use num_bigint_dig::traits::ModInverse;
    use num_bigint_dig::{BigUint, ToBigUint};

    // RFC 3526 2048-bit MODP group prime (hex, continuous)
    const RFC3526_P_HEX: &str = "FFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD129024E088A67CC74020BBEA63B139B22514A08798E3404DDEF9519B3CD3A431B302B0A6DF25F14374FE1356D6D51C245E485B576625E7EC6F44C42E9A637ED6B0BFF5CB6F406B7EDEE386BFB5A899FA5AE9F24117C4B1FE649286651ECE45B3DC2007CB8A163BF0598DA48361C55D39A69163FA8FD24CF5F83655D23DCA3AD961C62F356208552BB9ED529077096966D670C354E4ABC9804F1746C08CA18217C32905E462E36CE3BE39E772C180E86039B2783A2EC07A28FB5C55DF06F4C52C9DE2BCBF6955817183995497CEA956AE515D2261898FA051015728E5A8AACAA68FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF";

    fn parse_prime() -> BigUint {
        let hex_clean: String = RFC3526_P_HEX
            .chars()
            .filter(|c| c.is_ascii_hexdigit())
            .collect();
        // The RFC 3526 hex constant is always valid; the fallback is a large odd number.
        BigUint::parse_bytes(hex_clean.as_bytes(), 16)
            .unwrap_or_else(|| {
                // Fallback: construct 2^511 - 1 via byte manipulation.
                let mut bytes = vec![0xFFu8; 64];
                bytes[0] &= 0x7F; // clear top bit → 2^511 - 1
                BigUint::from_bytes_be(&bytes)
            })
    }

    /// Returns (PublicKey, PrivateKey) where priv_key.lambda = raw x bytes.
    /// The caller (elgamal_keygen_real) adds the p-length prefix encoding.
    pub fn generate_keypair(key_size: usize) -> (PublicKey, PrivateKey) {
        let p = parse_prime();
        let g = BigUint::from(2u32);

        let p_bytes = p.to_bytes_be();
        let x = loop {
            let rand_bytes: Vec<u8> = (0..p_bytes.len())
                .map(|_| fastrand::u8(..))
                .collect();
            let candidate = BigUint::from_bytes_be(&rand_bytes) % (&p - 2u32) + 2u32;
            if candidate > BigUint::from(1u32) && candidate < p {
                break candidate;
            }
        };

        let h = g.modpow(&x, &p);

        let pub_key = PublicKey {
            n: p.to_bytes_be(),
            g: h.to_bytes_be(),
            size: key_size,
            scheme: EncryptionScheme::ElGamal,
        };
        let priv_key = PrivateKey {
            lambda: x.to_bytes_be(),
            mu: g.to_bytes_be(),
            size: key_size,
            scheme: EncryptionScheme::ElGamal,
        };
        (pub_key, priv_key)
    }

    fn biguint_to_u64_low(v: &BigUint) -> u64 {
        // Extract the lowest 8 bytes (little-endian) from little-endian byte representation.
        let bytes = v.to_bytes_le();
        let mut buf = [0u8; 8];
        let take = bytes.len().min(8);
        buf[..take].copy_from_slice(&bytes[..take]);
        u64::from_le_bytes(buf)
    }

    fn f32_to_int(x: f32, precision_bits: u32) -> u64 {
        let scale = 1u64 << precision_bits;
        ((x.abs() * scale as f32).round() as u64).max(1)
    }

    fn int_to_f32(x: u64, precision_bits: u32) -> f32 {
        let scale = 1u64 << precision_bits;
        x as f32 / scale as f32
    }

    fn serialize_pair(c1: &BigUint, c2: &BigUint) -> Vec<u8> {
        let c1_bytes = c1.to_bytes_be();
        let c2_bytes = c2.to_bytes_be();
        let c1_len = c1_bytes.len() as u32;
        let mut out = Vec::with_capacity(4 + c1_bytes.len() + c2_bytes.len());
        out.extend_from_slice(&c1_len.to_le_bytes());
        out.extend_from_slice(&c1_bytes);
        out.extend_from_slice(&c2_bytes);
        out
    }

    fn deserialize_pair(data: &[u8]) -> Result<(BigUint, BigUint), EncryptionError> {
        if data.len() < 4 {
            return Err(EncryptionError::DecryptionFailed(
                "ElGamal ciphertext too short".to_string(),
            ));
        }
        let c1_len =
            u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
        if data.len() < 4 + c1_len {
            return Err(EncryptionError::DecryptionFailed(
                "ElGamal ciphertext truncated".to_string(),
            ));
        }
        let c1 = BigUint::from_bytes_be(&data[4..4 + c1_len]);
        let c2 = BigUint::from_bytes_be(&data[4 + c1_len..]);
        Ok((c1, c2))
    }

    pub struct ElGamalEncryption {
        pub precision_bits: u32,
    }

    impl ElGamalEncryption {
        pub fn new(precision_bits: u32) -> Self {
            Self { precision_bits }
        }
    }

    impl HomomorphicEncryption for ElGamalEncryption {
        fn encrypt(
            &self,
            plaintext: f32,
            public_key: &PublicKey,
        ) -> Result<Ciphertext, EncryptionError> {
            let p = BigUint::from_bytes_be(&public_key.n);
            let h = BigUint::from_bytes_be(&public_key.g);
            let g = BigUint::from(2u32);

            let m_int = f32_to_int(plaintext, self.precision_bits);
            let m = BigUint::from(m_int);

            let p_bytes = p.to_bytes_be();
            let k = loop {
                let rand_bytes: Vec<u8> = (0..p_bytes.len())
                    .map(|_| fastrand::u8(..))
                    .collect();
                let candidate =
                    BigUint::from_bytes_be(&rand_bytes) % (&p - 2u32) + 2u32;
                if candidate > BigUint::from(1u32) {
                    break candidate;
                }
            };

            let c1 = g.modpow(&k, &p);
            let hk = h.modpow(&k, &p);
            let c2 = (&m * &hk) % &p;

            Ok(Ciphertext {
                data: serialize_pair(&c1, &c2),
                scheme: EncryptionScheme::ElGamal,
                batch_size: None,
            })
        }

        fn decrypt(
            &self,
            ciphertext: &Ciphertext,
            private_key: &PrivateKey,
        ) -> Result<f32, EncryptionError> {
            if ciphertext.scheme != EncryptionScheme::ElGamal {
                return Err(EncryptionError::IncompatibleCiphertext(
                    "Expected ElGamal ciphertext".to_string(),
                ));
            }
            // priv_key.lambda encodes: 4-byte LE p_len || p_bytes || x_bytes
            let lambda_field = &private_key.lambda;
            if lambda_field.len() < 4 {
                return Err(EncryptionError::InvalidKey(
                    "Invalid ElGamal private key encoding".to_string(),
                ));
            }
            let p_len = u32::from_le_bytes([
                lambda_field[0],
                lambda_field[1],
                lambda_field[2],
                lambda_field[3],
            ]) as usize;
            if lambda_field.len() < 4 + p_len {
                return Err(EncryptionError::InvalidKey(
                    "Truncated ElGamal private key".to_string(),
                ));
            }
            let p = BigUint::from_bytes_be(&lambda_field[4..4 + p_len]);
            let x = BigUint::from_bytes_be(&lambda_field[4 + p_len..]);

            let (c1, c2) = deserialize_pair(&ciphertext.data)?;

            let s = c1.modpow(&x, &p);
            // mod_inverse returns Option<BigInt>; normalise negative to positive residue mod p.
            use num_bigint_dig::{BigInt, Sign, ToBigUint as _};
            let s_inv = s
                .mod_inverse(&p)
                .and_then(|v: BigInt| {
                    if v.sign() == Sign::Minus {
                        let p_big = BigInt::from(p.clone());
                        (v + p_big).to_biguint()
                    } else {
                        v.to_biguint()
                    }
                })
                .ok_or_else(|| {
                    EncryptionError::DecryptionFailed(
                        "ElGamal: s has no inverse mod p".to_string(),
                    )
                })?;
            let m = (&c2 * &s_inv) % &p;

            let m_u64 = biguint_to_u64_low(&m);
            Ok(int_to_f32(m_u64, self.precision_bits))
        }

        fn add(
            &self,
            _c1: &Ciphertext,
            _c2: &Ciphertext,
        ) -> Result<Ciphertext, EncryptionError> {
            Err(EncryptionError::IncompatibleCiphertext(
                "ElGamal is multiplicatively homomorphic only; addition is not supported"
                    .to_string(),
            ))
        }

        fn multiply(
            &self,
            c1: &Ciphertext,
            c2: &Ciphertext,
        ) -> Result<Ciphertext, EncryptionError> {
            if c1.scheme != EncryptionScheme::ElGamal
                || c2.scheme != EncryptionScheme::ElGamal
            {
                return Err(EncryptionError::IncompatibleCiphertext(
                    "Both ciphertexts must use ElGamal scheme".to_string(),
                ));
            }
            Err(EncryptionError::IncompatibleCiphertext(
                "ElGamal homomorphic multiply requires prime p context".to_string(),
            ))
        }

        fn scalar_multiply(
            &self,
            _ciphertext: &Ciphertext,
            _scalar: f32,
        ) -> Result<Ciphertext, EncryptionError> {
            Err(EncryptionError::IncompatibleCiphertext(
                "ElGamal scalar_multiply requires prime p context".to_string(),
            ))
        }

        fn scheme(&self) -> EncryptionScheme {
            EncryptionScheme::ElGamal
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_encryption_basic() {
        let he = MockEncryption::new();
        let (public_key, private_key) =
            EncryptedInference::generate_keypair(EncryptionScheme::Mock, 2048).unwrap();

        let plaintext = 42.5;
        let ciphertext = he.encrypt(plaintext, &public_key).unwrap();
        let decrypted = he.decrypt(&ciphertext, &private_key).unwrap();

        assert!((decrypted - plaintext).abs() < 1e-5);
    }

    #[test]
    fn test_homomorphic_addition() {
        let he = MockEncryption::new();
        let (public_key, _) =
            EncryptedInference::generate_keypair(EncryptionScheme::Mock, 2048).unwrap();

        let c1 = he.encrypt(10.0, &public_key).unwrap();
        let c2 = he.encrypt(20.0, &public_key).unwrap();

        let c_sum = he.add(&c1, &c2).unwrap();

        let (_, private_key) =
            EncryptedInference::generate_keypair(EncryptionScheme::Mock, 2048).unwrap();

        let result = he.decrypt(&c_sum, &private_key).unwrap();
        assert!((result - 30.0).abs() < 1e-5);
    }

    #[test]
    fn test_homomorphic_multiplication() {
        let he = MockEncryption::new();
        let (public_key, private_key) =
            EncryptedInference::generate_keypair(EncryptionScheme::Mock, 2048).unwrap();

        let c1 = he.encrypt(10.0, &public_key).unwrap();
        let c2 = he.encrypt(5.0, &public_key).unwrap();

        let c_product = he.multiply(&c1, &c2).unwrap();
        let result = he.decrypt(&c_product, &private_key).unwrap();

        assert!((result - 50.0).abs() < 1e-5);
    }

    #[test]
    fn test_encrypted_inference_creation() {
        let config = SecureInferenceConfig::default();
        let ei = EncryptedInference::new(config);

        assert!(ei.is_ok());
    }

    #[test]
    fn test_encrypted_vector_operations() {
        let config = SecureInferenceConfig::default();
        let ei = EncryptedInference::new(config).unwrap();

        let plaintexts = vec![1.0, 2.0, 3.0, 4.0];
        let encrypted = ei.encrypt_vec(&plaintexts).unwrap();

        assert_eq!(encrypted.len(), plaintexts.len());

        let decrypted = ei.decrypt_vec(&encrypted).unwrap();

        for (orig, dec) in plaintexts.iter().zip(decrypted.iter()) {
            assert!((orig - dec).abs() < 1e-5);
        }
    }

    #[test]
    fn test_encrypted_matrix_vector_multiplication() {
        let config = SecureInferenceConfig::default();
        let ei = EncryptedInference::new(config).unwrap();

        // 2x3 matrix
        let matrix = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];

        // 3D vector
        let vector = vec![1.0, 2.0, 3.0];
        let encrypted_vec = ei.encrypt_vec(&vector).unwrap();

        // Perform encrypted computation
        let encrypted_result = ei.encrypted_matvec(&matrix, &encrypted_vec).unwrap();

        // Decrypt result
        let result = ei.decrypt_vec(&encrypted_result).unwrap();

        // Expected: [1*1 + 2*2 + 3*3, 4*1 + 5*2 + 6*3] = [14, 32]
        assert!((result[0] - 14.0).abs() < 1e-4);
        assert!((result[1] - 32.0).abs() < 1e-4);
    }

    #[test]
    fn test_encrypted_linear_layer() {
        let config = SecureInferenceConfig::default();
        let ei = EncryptedInference::new(config).unwrap();

        let weights = vec![vec![1.0, 2.0], vec![3.0, 4.0]];

        let bias = vec![0.5, 1.0];
        let input = vec![2.0, 3.0];

        let encrypted_input = ei.encrypt_vec(&input).unwrap();
        let encrypted_output = ei
            .encrypted_linear_layer(&weights, &encrypted_input, Some(&bias))
            .unwrap();

        let output = ei.decrypt_vec(&encrypted_output).unwrap();

        // Expected: [1*2 + 2*3 + 0.5, 3*2 + 4*3 + 1.0] = [8.5, 19.0]
        assert!((output[0] - 8.5).abs() < 1e-4);
        assert!((output[1] - 19.0).abs() < 1e-4);
    }

    #[test]
    fn test_encrypted_activation() {
        let config = SecureInferenceConfig::default();
        let ei = EncryptedInference::new(config).unwrap();

        let input = 3.0;
        let encrypted_input = ei.encrypt(input).unwrap();

        // Test quadratic activation (x^2)
        let encrypted_output = ei
            .encrypted_activation(&encrypted_input, ActivationType::Quadratic)
            .unwrap();
        let output = ei.decrypt(&encrypted_output).unwrap();

        assert!((output - 9.0).abs() < 1e-4);
    }

    #[cfg(feature = "homomorphic")]
    mod homomorphic_tests {
        use super::*;

        #[test]
        fn test_paillier_keygen_generates_valid_keys() {
            let (pub_key, priv_key) = paillier_keygen_real(512);
            assert!(!pub_key.n.is_empty());
            assert!(!pub_key.g.is_empty());
            assert!(!priv_key.lambda.is_empty());
            assert!(!priv_key.mu.is_empty());
            assert_eq!(pub_key.scheme, EncryptionScheme::Paillier);
            assert_eq!(priv_key.scheme, EncryptionScheme::Paillier);
        }

        #[test]
        fn test_paillier_encrypt_decrypt_roundtrip() {
            let (pub_key, priv_key) = paillier_keygen_real(512);
            let enc = paillier::PaillierEncryption::new(16);
            let plaintext = 3.5_f32;
            let ct = enc.encrypt(plaintext, &pub_key).expect("encrypt must succeed");
            let decrypted = enc.decrypt(&ct, &priv_key).expect("decrypt must succeed");
            let err = (decrypted - plaintext).abs();
            assert!(
                err < 0.02,
                "Decrypted {} differs from plaintext {} by {}",
                decrypted,
                plaintext,
                err
            );
        }

        #[test]
        fn test_paillier_multiply_returns_err() {
            let (pub_key, _) = paillier_keygen_real(512);
            let enc = paillier::PaillierEncryption::new(16);
            let ct = enc.encrypt(1.0, &pub_key).expect("encrypt must succeed");
            let result = enc.multiply(&ct, &ct);
            assert!(result.is_err());
        }

        #[test]
        fn test_paillier_add_returns_err() {
            let (pub_key, _) = paillier_keygen_real(512);
            let enc = paillier::PaillierEncryption::new(16);
            let ct = enc.encrypt(1.0, &pub_key).expect("encrypt must succeed");
            let result = enc.add(&ct, &ct);
            assert!(result.is_err());
        }

        #[test]
        fn test_elgamal_keygen_generates_valid_keys() {
            let (pub_key, priv_key) = elgamal_keygen_real(2048);
            assert!(!pub_key.n.is_empty());
            assert!(!pub_key.g.is_empty());
            assert!(!priv_key.lambda.is_empty());
            assert_eq!(pub_key.scheme, EncryptionScheme::ElGamal);
        }

        #[test]
        fn test_elgamal_encrypt_decrypt_roundtrip() {
            let (pub_key, priv_key) = elgamal_keygen_real(2048);
            let enc = elgamal::ElGamalEncryption::new(16);
            let plaintext = 2.5_f32;
            let ct = enc.encrypt(plaintext, &pub_key).expect("encrypt must succeed");
            let decrypted = enc.decrypt(&ct, &priv_key).expect("decrypt must succeed");
            let err = (decrypted - plaintext).abs();
            assert!(
                err < 0.02,
                "Decrypted {} differs from plaintext {} by {}",
                decrypted,
                plaintext,
                err
            );
        }

        #[test]
        fn test_elgamal_add_returns_err() {
            let (pub_key, _) = elgamal_keygen_real(2048);
            let enc = elgamal::ElGamalEncryption::new(16);
            let ct = enc.encrypt(1.0, &pub_key).expect("encrypt must succeed");
            let result = enc.add(&ct, &ct);
            assert!(result.is_err());
        }

        #[test]
        fn test_encrypted_inference_paillier_creation() {
            let config = SecureInferenceConfig {
                scheme: EncryptionScheme::Paillier,
                key_size: 512,
                precision_bits: Some(16),
                enable_batching: false,
                max_depth: None,
            };
            let ei = EncryptedInference::new(config);
            assert!(ei.is_ok());
        }

        #[test]
        fn test_encrypted_inference_elgamal_creation() {
            let config = SecureInferenceConfig {
                scheme: EncryptionScheme::ElGamal,
                key_size: 2048,
                precision_bits: Some(16),
                enable_batching: false,
                max_depth: None,
            };
            let ei = EncryptedInference::new(config);
            assert!(ei.is_ok());
        }
    }
}

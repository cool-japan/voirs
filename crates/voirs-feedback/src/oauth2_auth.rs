//! OAuth 2.0 Authentication Implementation for VoiRS Feedback System
//!
//! This module provides comprehensive OAuth 2.0 authentication with support for
//! multiple providers, JWT tokens, PKCE (Proof Key for Code Exchange), and
//! refresh token management.

use async_trait::async_trait;
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use chrono::{DateTime, Duration, Utc};
use jsonwebtoken::{decode, encode, Algorithm, DecodingKey, EncodingKey, Header, Validation};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::RwLock;
use uuid::Uuid;

/// OAuth 2.0 authentication errors
#[derive(Error, Debug)]
pub enum OAuth2Error {
    /// Invalid authorization code
    #[error("Invalid authorization code")]
    InvalidAuthorizationCode,

    /// Invalid access token
    #[error("Invalid access token: {message}")]
    InvalidAccessToken { message: String },

    /// Token expired
    #[error("Token expired")]
    TokenExpired,

    /// Invalid refresh token
    #[error("Invalid refresh token")]
    InvalidRefreshToken,

    /// Provider error
    #[error("OAuth provider error: {provider} - {message}")]
    ProviderError { provider: String, message: String },

    /// Configuration error
    #[error("OAuth configuration error: {message}")]
    ConfigurationError { message: String },

    /// Network error
    #[error("Network error: {message}")]
    NetworkError { message: String },

    /// JWT error
    #[error("JWT error: {message}")]
    JwtError { message: String },

    /// PKCE error
    #[error("PKCE verification error")]
    PkceError,

    /// Scope error
    #[error("Insufficient scope: required {required}, got {actual}")]
    ScopeError { required: String, actual: String },
}

/// Result type for OAuth 2.0 operations
pub type OAuth2Result<T> = Result<T, OAuth2Error>;

/// Supported OAuth 2.0 providers
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum OAuth2Provider {
    /// Google OAuth
    Google,
    /// Microsoft Azure AD
    Microsoft,
    /// GitHub OAuth
    GitHub,
    /// Auth0
    Auth0,
    /// Custom provider
    Custom { name: String, base_url: String },
}

/// OAuth 2.0 grant types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum GrantType {
    /// Authorization Code Grant
    AuthorizationCode,
    /// Implicit Grant (deprecated)
    Implicit,
    /// Resource Owner Password Credentials Grant
    ResourceOwnerPassword,
    /// Client Credentials Grant
    ClientCredentials,
    /// Refresh Token Grant
    RefreshToken,
    /// Device Authorization Grant
    DeviceAuthorization,
}

/// OAuth 2.0 configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OAuth2Config {
    /// OAuth provider
    pub provider: OAuth2Provider,
    /// Client ID
    pub client_id: String,
    /// Client secret
    pub client_secret: String,
    /// Redirect URI
    pub redirect_uri: String,
    /// Authorization endpoint
    pub authorization_endpoint: String,
    /// Token endpoint
    pub token_endpoint: String,
    /// User info endpoint
    pub user_info_endpoint: Option<String>,
    /// Supported scopes
    pub scopes: Vec<String>,
    /// Default scopes
    pub default_scopes: Vec<String>,
    /// Enable PKCE
    pub enable_pkce: bool,
    /// JWT signing key
    pub jwt_secret: String,
    /// Token expiration time
    pub access_token_expiry: Duration,
    /// Refresh token expiration time
    pub refresh_token_expiry: Duration,
}

/// OAuth 2.0 authorization request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthorizationRequest {
    /// Request ID
    pub request_id: Uuid,
    /// Client ID
    pub client_id: String,
    /// Redirect URI
    pub redirect_uri: String,
    /// Requested scopes
    pub scopes: Vec<String>,
    /// State parameter for CSRF protection
    pub state: String,
    /// PKCE code challenge
    pub code_challenge: Option<String>,
    /// PKCE code challenge method
    pub code_challenge_method: Option<String>,
    /// Created timestamp
    pub created_at: DateTime<Utc>,
}

/// OAuth 2.0 token response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenResponse {
    /// Access token
    pub access_token: String,
    /// Token type (usually "Bearer")
    pub token_type: String,
    /// Token expiration in seconds
    pub expires_in: u64,
    /// Refresh token
    pub refresh_token: Option<String>,
    /// Granted scopes
    pub scope: Option<String>,
    /// ID token (OpenID Connect)
    pub id_token: Option<String>,
}

/// User information from OAuth provider
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserInfo {
    /// User ID
    pub id: String,
    /// Email address
    pub email: String,
    /// Email verification status
    pub email_verified: bool,
    /// Full name
    pub name: Option<String>,
    /// First name
    pub given_name: Option<String>,
    /// Last name
    pub family_name: Option<String>,
    /// Profile picture URL
    pub picture: Option<String>,
    /// Locale
    pub locale: Option<String>,
}

/// JWT claims structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JwtClaims {
    /// Subject (user ID)
    pub sub: String,
    /// Issued at
    pub iat: u64,
    /// Expiration time
    pub exp: u64,
    /// Issuer
    pub iss: String,
    /// Audience
    pub aud: String,
    /// User email
    pub email: String,
    /// Granted scopes
    pub scopes: Vec<String>,
    /// Provider
    pub provider: String,
}

/// PKCE (Proof Key for Code Exchange) helper
#[derive(Debug, Clone)]
pub struct PkceChallenge {
    /// Code verifier
    pub code_verifier: String,
    /// Code challenge
    pub code_challenge: String,
    /// Challenge method
    pub method: String,
}

impl PkceChallenge {
    /// Generate new PKCE challenge
    pub fn new() -> Self {
        use rand::{distributions::Alphanumeric, Rng};

        // Generate random code verifier (43-128 characters)
        let code_verifier: String = rand::thread_rng()
            .sample_iter(&Alphanumeric)
            .take(128)
            .map(char::from)
            .collect();

        // Generate code challenge using SHA256
        let digest = Sha256::digest(code_verifier.as_bytes());
        let code_challenge = URL_SAFE_NO_PAD.encode(digest);

        Self {
            code_verifier,
            code_challenge,
            method: String::from("S256"),
        }
    }

    /// Verify PKCE challenge
    pub fn verify(&self, verifier: &str) -> bool {
        let digest = Sha256::digest(verifier.as_bytes());
        let challenge = URL_SAFE_NO_PAD.encode(digest);
        challenge == self.code_challenge
    }
}

/// OAuth 2.0 authentication manager
pub struct OAuth2Manager {
    /// Configuration
    config: OAuth2Config,
    /// HTTP client
    client: Client,
    /// Active authorization requests
    auth_requests: Arc<RwLock<HashMap<String, AuthorizationRequest>>>,
    /// PKCE challenges
    pkce_challenges: Arc<RwLock<HashMap<String, PkceChallenge>>>,
    /// JWT encoding key
    jwt_encoding_key: EncodingKey,
    /// JWT decoding key
    jwt_decoding_key: DecodingKey,
}

impl OAuth2Manager {
    /// Create new OAuth 2.0 manager
    pub fn new(config: OAuth2Config) -> OAuth2Result<Self> {
        let jwt_secret = config.jwt_secret.as_bytes();
        let jwt_encoding_key = EncodingKey::from_secret(jwt_secret);
        let jwt_decoding_key = DecodingKey::from_secret(jwt_secret);

        Ok(Self {
            config,
            client: Client::new(),
            auth_requests: Arc::new(RwLock::new(HashMap::new())),
            pkce_challenges: Arc::new(RwLock::new(HashMap::new())),
            jwt_encoding_key,
            jwt_decoding_key,
        })
    }

    /// Generate authorization URL
    pub async fn get_authorization_url(
        &self,
        scopes: Option<Vec<String>>,
        state: Option<String>,
    ) -> OAuth2Result<(String, String)> {
        let request_id = Uuid::new_v4();
        let state = state.unwrap_or_else(|| Uuid::new_v4().to_string());
        let scopes = scopes.unwrap_or_else(|| self.config.default_scopes.clone());

        let mut url_params = vec![
            ("response_type", String::from("code")),
            ("client_id", self.config.client_id.clone()),
            ("redirect_uri", self.config.redirect_uri.clone()),
            ("scope", scopes.join(" ")),
            ("state", state.clone()),
        ];

        // Add PKCE challenge if enabled
        let mut pkce_challenge = None;
        if self.config.enable_pkce {
            let challenge = PkceChallenge::new();
            url_params.push(("code_challenge", challenge.code_challenge.clone()));
            url_params.push(("code_challenge_method", challenge.method.clone()));

            self.pkce_challenges
                .write()
                .await
                .insert(state.clone(), challenge.clone());
            pkce_challenge = Some(challenge);
        }

        // Store authorization request
        let auth_request = AuthorizationRequest {
            request_id,
            client_id: self.config.client_id.clone(),
            redirect_uri: self.config.redirect_uri.clone(),
            scopes,
            state: state.clone(),
            code_challenge: pkce_challenge.as_ref().map(|c| c.code_challenge.clone()),
            code_challenge_method: pkce_challenge.as_ref().map(|_| String::from("S256")),
            created_at: Utc::now(),
        };

        self.auth_requests
            .write()
            .await
            .insert(state.clone(), auth_request);

        // Build authorization URL
        let query_string = url_params
            .into_iter()
            .map(|(k, v)| format!("{}={}", k, urlencoding::encode(&v)))
            .collect::<Vec<_>>()
            .join("&");

        let auth_url = format!("{}?{}", self.config.authorization_endpoint, query_string);

        Ok((auth_url, state))
    }

    /// Exchange authorization code for tokens
    pub async fn exchange_code(
        &self,
        code: &str,
        state: &str,
    ) -> OAuth2Result<(TokenResponse, UserInfo)> {
        // Validate state and get authorization request
        let auth_request = {
            let mut requests = self.auth_requests.write().await;
            requests
                .remove(state)
                .ok_or(OAuth2Error::InvalidAuthorizationCode)?
        };

        // Build token request parameters
        let mut params = vec![
            ("grant_type", String::from("authorization_code")),
            ("code", code.to_string()),
            ("redirect_uri", auth_request.redirect_uri),
            ("client_id", self.config.client_id.clone()),
            ("client_secret", self.config.client_secret.clone()),
        ];

        // Add PKCE verifier if enabled
        if self.config.enable_pkce {
            if let Some(challenge) = self.pkce_challenges.write().await.remove(state) {
                params.push(("code_verifier", challenge.code_verifier));
            } else {
                return Err(OAuth2Error::PkceError);
            }
        }

        // Exchange code for tokens
        let response = self
            .client
            .post(&self.config.token_endpoint)
            .form(&params)
            .send()
            .await
            .map_err(|e| OAuth2Error::NetworkError {
                message: e.to_string(),
            })?;

        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(OAuth2Error::ProviderError {
                provider: format!("{:?}", self.config.provider),
                message: error_text,
            });
        }

        let token_response: TokenResponse =
            response
                .json()
                .await
                .map_err(|e| OAuth2Error::NetworkError {
                    message: e.to_string(),
                })?;

        // Get user information
        let user_info = self.get_user_info(&token_response.access_token).await?;

        Ok((token_response, user_info))
    }

    /// Get user information using access token
    pub async fn get_user_info(&self, access_token: &str) -> OAuth2Result<UserInfo> {
        let user_info_endpoint = self.config.user_info_endpoint.as_ref().ok_or_else(|| {
            OAuth2Error::ConfigurationError {
                message: String::from("User info endpoint not configured"),
            }
        })?;

        let response = self
            .client
            .get(user_info_endpoint)
            .bearer_auth(access_token)
            .send()
            .await
            .map_err(|e| OAuth2Error::NetworkError {
                message: e.to_string(),
            })?;

        if !response.status().is_success() {
            return Err(OAuth2Error::InvalidAccessToken {
                message: String::from("Failed to get user info"),
            });
        }

        let user_info: UserInfo = response
            .json()
            .await
            .map_err(|e| OAuth2Error::NetworkError {
                message: e.to_string(),
            })?;

        Ok(user_info)
    }

    /// Generate JWT token
    pub fn generate_jwt(&self, user_info: &UserInfo, scopes: Vec<String>) -> OAuth2Result<String> {
        let now = Utc::now();
        let claims = JwtClaims {
            sub: user_info.id.clone(),
            iat: now.timestamp() as u64,
            exp: (now + self.config.access_token_expiry).timestamp() as u64,
            iss: String::from("voirs-feedback"),
            aud: self.config.client_id.clone(),
            email: user_info.email.clone(),
            scopes,
            provider: format!("{:?}", self.config.provider),
        };

        encode(&Header::default(), &claims, &self.jwt_encoding_key).map_err(|e| {
            OAuth2Error::JwtError {
                message: e.to_string(),
            }
        })
    }

    /// Validate JWT token
    pub fn validate_jwt(&self, token: &str) -> OAuth2Result<JwtClaims> {
        let mut validation = Validation::new(Algorithm::HS256);
        validation.set_audience(&[&self.config.client_id]);
        validation.set_issuer(&["voirs-feedback"]);

        decode::<JwtClaims>(token, &self.jwt_decoding_key, &validation)
            .map(|data| data.claims)
            .map_err(|e| OAuth2Error::JwtError {
                message: e.to_string(),
            })
    }

    /// Refresh access token
    pub async fn refresh_token(&self, refresh_token: &str) -> OAuth2Result<TokenResponse> {
        let params = vec![
            ("grant_type", String::from("refresh_token")),
            ("refresh_token", refresh_token.to_string()),
            ("client_id", self.config.client_id.clone()),
            ("client_secret", self.config.client_secret.clone()),
        ];

        let response = self
            .client
            .post(&self.config.token_endpoint)
            .form(&params)
            .send()
            .await
            .map_err(|e| OAuth2Error::NetworkError {
                message: e.to_string(),
            })?;

        if !response.status().is_success() {
            return Err(OAuth2Error::InvalidRefreshToken);
        }

        let token_response: TokenResponse =
            response
                .json()
                .await
                .map_err(|e| OAuth2Error::NetworkError {
                    message: e.to_string(),
                })?;

        Ok(token_response)
    }

    /// Validate required scopes
    pub fn validate_scopes(&self, required: &[String], actual: &[String]) -> OAuth2Result<()> {
        for scope in required {
            if !actual.contains(scope) {
                return Err(OAuth2Error::ScopeError {
                    required: required.join(", "),
                    actual: actual.join(", "),
                });
            }
        }
        Ok(())
    }

    /// Cleanup expired authorization requests
    pub async fn cleanup_expired_requests(&self) {
        let mut requests = self.auth_requests.write().await;
        let mut challenges = self.pkce_challenges.write().await;
        let cutoff = Utc::now() - Duration::minutes(10); // 10 minute expiry

        requests.retain(|state, request| {
            let keep = request.created_at > cutoff;
            if !keep {
                challenges.remove(state);
            }
            keep
        });
    }

    /// Get OAuth provider configurations
    pub fn get_provider_config(provider: OAuth2Provider) -> OAuth2Config {
        match provider {
            OAuth2Provider::Google => OAuth2Config {
                provider,
                client_id: std::env::var("GOOGLE_CLIENT_ID").unwrap_or_default(),
                client_secret: std::env::var("GOOGLE_CLIENT_SECRET").unwrap_or_default(),
                redirect_uri: String::from("http://localhost:3000/auth/callback/google"),
                authorization_endpoint: String::from(
                    "https://accounts.google.com/o/oauth2/v2/auth",
                ),
                token_endpoint: String::from("https://oauth2.googleapis.com/token"),
                user_info_endpoint: Some(String::from(
                    "https://openidconnector.googleapis.com/v1/userinfo",
                )),
                scopes: vec![
                    String::from("openid"),
                    String::from("email"),
                    String::from("profile"),
                ],
                default_scopes: vec![String::from("openid"), String::from("email")],
                enable_pkce: true,
                jwt_secret: std::env::var("JWT_SECRET")
                    .unwrap_or_else(|_| String::from("default-secret")),
                access_token_expiry: Duration::hours(1),
                refresh_token_expiry: Duration::days(30),
            },
            OAuth2Provider::Microsoft => OAuth2Config {
                provider,
                client_id: std::env::var("MICROSOFT_CLIENT_ID").unwrap_or_default(),
                client_secret: std::env::var("MICROSOFT_CLIENT_SECRET").unwrap_or_default(),
                redirect_uri: String::from("http://localhost:3000/auth/callback/microsoft"),
                authorization_endpoint: String::from(
                    "https://login.microsoftonline.com/common/oauth2/v2.0/authorize",
                ),
                token_endpoint: "https://login.microsoftonline.com/common/oauth2/v2.0/token"
                    .to_string(),
                user_info_endpoint: Some(String::from("https://graph.microsoft.com/v1.0/me")),
                scopes: vec![
                    String::from("openid"),
                    String::from("email"),
                    String::from("profile"),
                ],
                default_scopes: vec![String::from("openid"), String::from("email")],
                enable_pkce: true,
                jwt_secret: std::env::var("JWT_SECRET")
                    .unwrap_or_else(|_| String::from("default-secret")),
                access_token_expiry: Duration::hours(1),
                refresh_token_expiry: Duration::days(30),
            },
            OAuth2Provider::GitHub => OAuth2Config {
                provider,
                client_id: std::env::var("GITHUB_CLIENT_ID").unwrap_or_default(),
                client_secret: std::env::var("GITHUB_CLIENT_SECRET").unwrap_or_default(),
                redirect_uri: String::from("http://localhost:3000/auth/callback/github"),
                authorization_endpoint: String::from("https://github.com/login/oauth/authorize"),
                token_endpoint: String::from("https://github.com/login/oauth/access_token"),
                user_info_endpoint: Some(String::from("https://api.github.com/user")),
                scopes: vec![String::from("user:email")],
                default_scopes: vec![String::from("user:email")],
                enable_pkce: false, // GitHub doesn't support PKCE yet
                jwt_secret: std::env::var("JWT_SECRET")
                    .unwrap_or_else(|_| String::from("default-secret")),
                access_token_expiry: Duration::hours(1),
                refresh_token_expiry: Duration::days(30),
            },
            _ => panic!("Provider configuration not implemented"),
        }
    }
}

/// OAuth 2.0 middleware for request authentication
pub struct OAuth2Middleware {
    manager: Arc<OAuth2Manager>,
    required_scopes: Vec<String>,
}

impl OAuth2Middleware {
    /// Create new OAuth 2.0 middleware
    pub fn new(manager: Arc<OAuth2Manager>, required_scopes: Vec<String>) -> Self {
        Self {
            manager,
            required_scopes,
        }
    }

    /// Authenticate request
    pub async fn authenticate(&self, auth_header: &str) -> OAuth2Result<JwtClaims> {
        // Extract Bearer token
        let token = auth_header
            .strip_prefix("Bearer ")
            .ok_or(OAuth2Error::InvalidAccessToken {
                message: String::from("Invalid authorization header format"),
            })?;

        // Validate JWT
        let claims = self.manager.validate_jwt(token)?;

        // Check token expiration
        let now = Utc::now().timestamp() as u64;
        if claims.exp < now {
            return Err(OAuth2Error::TokenExpired);
        }

        // Validate required scopes
        self.manager
            .validate_scopes(&self.required_scopes, &claims.scopes)?;

        Ok(claims)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pkce_challenge_generation() {
        let challenge = PkceChallenge::new();
        assert_eq!(challenge.method, "S256");
        assert!(!challenge.code_verifier.is_empty());
        assert!(!challenge.code_challenge.is_empty());
    }

    #[test]
    fn test_pkce_verification() {
        let challenge = PkceChallenge::new();
        assert!(challenge.verify(&challenge.code_verifier));
        assert!(!challenge.verify("wrong_verifier"));
    }

    #[tokio::test]
    async fn test_oauth2_manager_creation() {
        let config = OAuth2Manager::get_provider_config(OAuth2Provider::Google);
        let manager = OAuth2Manager::new(config);
        assert!(manager.is_ok());
    }

    #[tokio::test]
    async fn test_authorization_url_generation() {
        let config = OAuth2Manager::get_provider_config(OAuth2Provider::Google);
        let manager = OAuth2Manager::new(config).unwrap();

        let (url, state) = manager.get_authorization_url(None, None).await.unwrap();
        assert!(url.contains("accounts.google.com"));
        assert!(url.contains(&state));
        assert!(url.contains("code_challenge")); // PKCE enabled for Google
    }

    #[test]
    fn test_jwt_generation_and_validation() {
        let config = OAuth2Manager::get_provider_config(OAuth2Provider::Google);
        let manager = OAuth2Manager::new(config).unwrap();

        let user_info = UserInfo {
            id: String::from("test_user"),
            email: String::from("test@example.com"),
            email_verified: true,
            name: Some(String::from("Test User")),
            given_name: Some(String::from("Test")),
            family_name: Some(String::from("User")),
            picture: None,
            locale: Some(String::from("en-US")),
        };

        let scopes = vec![String::from("openid"), String::from("email")];
        let token = manager.generate_jwt(&user_info, scopes.clone()).unwrap();
        let claims = manager.validate_jwt(&token).unwrap();

        assert_eq!(claims.sub, user_info.id);
        assert_eq!(claims.email, user_info.email);
        assert_eq!(claims.scopes, scopes);
    }

    #[test]
    fn test_scope_validation() {
        let config = OAuth2Manager::get_provider_config(OAuth2Provider::Google);
        let manager = OAuth2Manager::new(config).unwrap();

        let required = vec![String::from("openid"), String::from("email")];
        let actual = vec![
            String::from("openid"),
            String::from("email"),
            String::from("profile"),
        ];

        assert!(manager.validate_scopes(&required, &actual).is_ok());

        let insufficient = vec![String::from("openid")];
        assert!(manager.validate_scopes(&required, &insufficient).is_err());
    }
}

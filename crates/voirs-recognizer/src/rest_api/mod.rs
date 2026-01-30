//! REST API and microservice support for speech recognition.
//!
//! This module provides a production-ready REST API server built on Axum, offering
//! HTTP endpoints for speech recognition, model management, and system monitoring.
//! It includes WebSocket support for real-time streaming recognition.
//!
//! # Features
//!
//! - **HTTP/REST API**: RESTful endpoints for batch recognition
//! - **WebSocket Streaming**: Real-time bidirectional audio streaming
//! - **OpenAPI/Swagger**: Auto-generated API documentation
//! - **Middleware Support**: CORS, rate limiting, authentication, compression
//! - **Load Balancing**: Ready for multi-instance deployment
//! - **Health Checks**: Kubernetes-ready liveness and readiness probes
//! - **Metrics Export**: Prometheus-compatible metrics endpoints
//!
//! # API Endpoints
//!
//! - `POST /api/v1/recognize` - Recognize speech from audio file
//! - `POST /api/v1/recognize/stream` - Real-time streaming recognition
//! - `GET /api/v1/models` - List available models
//! - `GET /api/v1/health` - Health check endpoint
//! - `GET /api/v1/metrics` - Prometheus metrics
//! - `WS /ws/recognize` - WebSocket streaming endpoint
//!
//! # Examples
//!
//! ```rust,no_run
//! use voirs_recognizer::rest_api::ApiServer;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let server = ApiServer::new("0.0.0.0:8080").await?;
//!     server.serve().await?;
//!     Ok(())
//! }
//! ```

// Allow unused async for REST API handler consistency and future compatibility
#![allow(clippy::unused_async)]

#[cfg(feature = "rest-api")]
mod server;

#[cfg(feature = "rest-api")]
mod handlers;

#[cfg(feature = "rest-api")]
mod types;

#[cfg(feature = "rest-api")]
mod middleware;

#[cfg(feature = "rest-api")]
mod openapi;

#[cfg(feature = "rest-api")]
mod websocket;

#[cfg(feature = "rest-api")]
pub use server::*;

#[cfg(feature = "rest-api")]
pub use handlers::*;

#[cfg(feature = "rest-api")]
pub use types::*;

#[cfg(feature = "rest-api")]
pub use middleware::*;

#[cfg(feature = "rest-api")]
pub use openapi::*;

#[cfg(feature = "rest-api")]
pub use websocket::*;

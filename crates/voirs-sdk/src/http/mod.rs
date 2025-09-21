pub mod api;
pub mod handlers;
pub mod middleware;
pub mod websocket;

use crate::{VoirsError, VoirsPipeline};
use axum::{Extension, Router};
use std::sync::Arc;
use tokio::sync::RwLock;
use tower::ServiceBuilder;
use tower_http::{
    compression::CompressionLayer,
    cors::{Any, CorsLayer},
    limit::RequestBodyLimitLayer,
    trace::TraceLayer,
};

pub type SharedPipeline = Arc<RwLock<VoirsPipeline>>;

#[allow(dead_code)]
pub struct HttpServer {
    pipeline: SharedPipeline,
    router: Router,
}

impl HttpServer {
    pub fn new(pipeline: VoirsPipeline) -> Self {
        let shared_pipeline = Arc::new(RwLock::new(pipeline));
        let router = create_router(shared_pipeline.clone());

        Self {
            pipeline: shared_pipeline,
            router,
        }
    }

    pub fn router(&self) -> Router {
        self.router.clone()
    }

    pub async fn serve(self, addr: std::net::SocketAddr) -> Result<(), VoirsError> {
        let listener = tokio::net::TcpListener::bind(addr).await.map_err(|e| {
            VoirsError::io_error(
                addr.to_string(),
                crate::error::types::IoOperation::Create,
                e,
            )
        })?;

        tracing::info!("HTTP server listening on {}", addr);

        axum::serve(listener, self.router)
            .await
            .map_err(|e| VoirsError::internal("http_server", format!("Server error: {e}")))
    }
}

fn create_router(pipeline: SharedPipeline) -> Router {
    let middleware = ServiceBuilder::new()
        .layer(TraceLayer::new_for_http())
        .layer(CompressionLayer::new())
        .layer(RequestBodyLimitLayer::new(16 * 1024 * 1024)) // 16MB limit
        .layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods(Any)
                .allow_headers(Any),
        );

    Router::new()
        .merge(api::create_api_routes())
        .merge(websocket::create_websocket_routes())
        .layer(middleware)
        .layer(Extension(pipeline))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::VoirsPipelineBuilder;

    #[tokio::test]
    async fn test_server_creation() {
        let pipeline = VoirsPipelineBuilder::new()
            .build()
            .await
            .expect("Failed to build pipeline");

        let server = HttpServer::new(pipeline);
        assert!(!server.router().fallback_service().is_none());
    }
}

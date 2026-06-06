//! VoiRS CLI main executable.

use voirs_cli::CliApp;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Install the pure-Rust rustls CryptoProvider before any TLS handshake.
    // reqwest is built with `rustls-no-provider`, so a default provider must be set.
    voirs_sdk::ensure_crypto_provider();

    CliApp::run().await?;
    Ok(())
}

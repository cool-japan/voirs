//! VoiRS CLI main executable.

use voirs_cli::CliApp;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    CliApp::run().await?;
    Ok(())
}

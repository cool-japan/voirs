//! G2P model training command implementation
//!
//! Provides CLI interface for training G2P (Grapheme-to-Phoneme) models.
//!
//! # Current status: fails closed
//!
//! `voirs-g2p`'s `LstmTrainer` does run a real forward pass over real dictionary
//! data (real character/phoneme index tensors, a real encoder/decoder forward
//! pass, a real loss computation), but it never performs a backward pass or
//! optimizer step (see `G2P_TRAINING_BLOCKED_REASON` below for exact evidence), so
//! model weights never change and the saved "model" file is fabricated data, not
//! serialized weights. Rather than run a loop that cannot learn and then present
//! its output as a trained model, this command validates and loads the
//! pronunciation dictionary (genuinely useful, independent of training) and then
//! refuses to train, with a diagnostic explaining exactly what is missing. This
//! guard should be removed once `voirs-g2p` implements real gradient-based
//! training.

use crate::error::CliError;
use crate::GlobalOptions;
use std::path::PathBuf;
use voirs_g2p::LanguageCode;
use voirs_sdk::Result;

/// Run G2P model training
pub async fn run_train_g2p(
    language: String,
    dictionary: PathBuf,
    output: PathBuf,
    config: Option<PathBuf>,
    epochs: usize,
    lr: f64,
    global: &GlobalOptions,
) -> Result<()> {
    if !global.quiet {
        println!("╔═══════════════════════════════════════════════════════════╗");
        println!("║          📖 VoiRS G2P Model Training                      ║");
        println!("╠═══════════════════════════════════════════════════════════╣");
        println!("║ Language:      {:<40} ║", language);
        println!("║ Dictionary:    {:<40} ║", truncate_path(&dictionary, 40));
        println!("║ Output path:   {:<40} ║", truncate_path(&output, 40));
        println!("║ Epochs:        {:<40} ║", epochs);
        println!("║ Learning rate: {:<40} ║", lr);
        if let Some(ref config_path) = config {
            println!("║ Config:        {:<40} ║", truncate_path(config_path, 40));
        }
        println!("╚═══════════════════════════════════════════════════════════╝");
        println!();
    }

    // Validate dictionary file
    if !dictionary.exists() {
        return Err(voirs_sdk::VoirsError::config_error(format!(
            "Dictionary file not found: {}",
            dictionary.display()
        )));
    }

    // Create output directory
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent)?;
    }

    train_g2p_model(language, dictionary, output, epochs, lr, global).await
}

async fn train_g2p_model(
    language: String,
    dictionary: PathBuf,
    output: PathBuf,
    epochs: usize,
    lr: f64,
    global: &GlobalOptions,
) -> Result<()> {
    if !global.quiet {
        println!("🔧 Initializing G2P training for language: {}\n", language);
        println!(
            "📚 Loading pronunciation dictionary from {}...",
            dictionary.display()
        );
    }

    // Load and validate dictionary. This is real, useful validation independent
    // of whether training itself can run: it parses the actual file and reports
    // real entry counts / malformed lines.
    let dict_entries = load_pronunciation_dictionary(&dictionary, &language).await?;
    let resolved_language = parse_language_code(&language);

    if !global.quiet {
        println!("   ✓ Loaded dictionary: {} entries", dict_entries.len());
        println!(
            "   ✓ Language: {} (resolved: {:?})",
            language, resolved_language
        );
        println!();
    }

    let entry_count = dict_entries.len();
    let output_display = output.display();
    Err(CliError::NotImplemented(format!(
        "G2P training requested (language={language}, dictionary entries={entry_count}, \
         output={output_display}, epochs={epochs}, lr={lr}) but refused: \
         {G2P_TRAINING_BLOCKED_REASON}"
    ))
    .into())
}

/// Why G2P training refuses to run.
///
/// Evidence, all in `crates/voirs-g2p/src/backends/neural/training.rs`:
/// - `LstmTrainer::train_epoch` (~L207-239) genuinely runs a real forward pass
///   (real character/phoneme index tensors built from the dictionary, a real
///   `SimpleEncoder`/`SimpleDecoder` forward pass, a real loss computation), but
///   the comment directly above the loss accumulation (~L225) states: "Simulated
///   backward pass and parameter update / In a full implementation, this would
///   involve actual gradients and optimization" -- no backward pass or optimizer
///   step ever runs, so weights never change.
/// - `train_model` (~L155) calls `prepare_batches` once before the epoch loop, so
///   with unchanging weights every epoch would report a bit-identical loss: there
///   is no learning curve at all, real or otherwise.
/// - `calculate_sequence_loss` (~L361-373) returns a constant `Ok(0.5)` whenever
///   predicted/target tensor shapes mismatch, silently substituting a stub loss.
/// - `save_model_safetensors` (~L423-456) ignores the encoder/decoder it is given
///   (parameters are literally named `_encoder`/`_decoder`) and writes a constant
///   dummy tensor (`vec![0.1f32; 100]`) plus a human-readable descriptive string
///   as the file contents -- not a real SafeTensors payload and not the model's
///   actual (never-updated) weights either way.
const G2P_TRAINING_BLOCKED_REASON: &str =
    "LstmTrainer::train_epoch runs a real forward pass over real dictionary data but never \
     performs a backward pass or optimizer step (the code says so directly: \"Simulated \
     backward pass and parameter update\"), so model weights never change and every epoch \
     would report the same loss; calculate_sequence_loss substitutes a constant Ok(0.5) on \
     any tensor shape mismatch instead of a real computed loss; and save_model_safetensors \
     ignores the trained encoder/decoder entirely and writes a constant dummy tensor and a \
     descriptive string rather than real weights or a real SafeTensors payload. Training \
     would silently produce a fabricated \"model\" file with no relationship to the \
     dictionary or the requested epochs, so this command refuses to run until voirs-g2p \
     implements real gradient-based training";

// Helper functions

/// Load pronunciation dictionary from file
async fn load_pronunciation_dictionary(
    path: &PathBuf,
    language: &str,
) -> Result<Vec<DictionaryEntry>> {
    // Check if file exists
    if !path.exists() {
        return Err(voirs_sdk::VoirsError::config_error(format!(
            "Dictionary file not found: {}",
            path.display()
        )));
    }

    // Read file contents
    let contents =
        tokio::fs::read_to_string(path)
            .await
            .map_err(|e| voirs_sdk::VoirsError::IoError {
                path: path.clone(),
                operation: voirs_sdk::error::IoOperation::Read,
                source: e,
            })?;

    // Parse dictionary entries
    let mut entries = Vec::new();
    for (line_num, line) in contents.lines().enumerate() {
        let line = line.trim();

        // Skip empty lines and comments
        if line.is_empty() || line.starts_with('#') || line.starts_with(';') {
            continue;
        }

        // Parse line: "word  phoneme1 phoneme2 phoneme3"
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 2 {
            eprintln!(
                "Warning: Skipping invalid entry at line {}: '{}'",
                line_num + 1,
                line
            );
            continue;
        }

        let grapheme = parts[0].to_lowercase();
        let phonemes = parts[1..].iter().map(|s| s.to_string()).collect();

        entries.push(DictionaryEntry {
            grapheme,
            phonemes,
            language: language.to_string(),
        });
    }

    if entries.is_empty() {
        return Err(voirs_sdk::VoirsError::config_error(
            "No valid dictionary entries found",
        ));
    }

    Ok(entries)
}

/// Dictionary entry structure
#[derive(Debug, Clone)]
struct DictionaryEntry {
    grapheme: String,
    phonemes: Vec<String>,
    language: String,
}

/// Parse language string to LanguageCode
fn parse_language_code(lang: &str) -> LanguageCode {
    match lang.to_lowercase().as_str() {
        "en" | "en-us" | "english" => LanguageCode::EnUs,
        "en-gb" | "english-uk" => LanguageCode::EnGb,
        "ja" | "ja-jp" | "japanese" => LanguageCode::Ja,
        "zh" | "zh-cn" | "chinese" | "mandarin" => LanguageCode::ZhCn,
        "ko" | "ko-kr" | "korean" => LanguageCode::Ko,
        "es" | "es-es" | "spanish" => LanguageCode::Es,
        "fr" | "fr-fr" | "french" => LanguageCode::Fr,
        "de" | "de-de" | "german" => LanguageCode::De,
        "it" | "it-it" | "italian" => LanguageCode::It,
        "pt" | "pt-br" | "pt-pt" | "portuguese" => LanguageCode::Pt,
        _ => {
            eprintln!("Warning: Unknown language '{}', defaulting to en-US", lang);
            LanguageCode::EnUs
        }
    }
}

fn truncate_path(path: &std::path::Path, max_len: usize) -> String {
    let path_str = path.display().to_string();
    if path_str.len() <= max_len {
        path_str
    } else {
        format!("...{}", &path_str[path_str.len() - (max_len - 3)..])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_truncate_path() {
        let path = PathBuf::from("/very/long/path/to/some/directory/file.txt");
        let truncated = truncate_path(&path, 20);
        assert!(truncated.len() <= 20);
        assert!(truncated.starts_with("..."));
    }

    #[test]
    fn test_parse_language_code() {
        assert!(matches!(parse_language_code("en"), LanguageCode::EnUs));
        assert!(matches!(parse_language_code("ja-jp"), LanguageCode::Ja));
        assert!(matches!(parse_language_code("chinese"), LanguageCode::ZhCn));
        assert!(matches!(parse_language_code("korean"), LanguageCode::Ko));
        assert!(matches!(parse_language_code("german"), LanguageCode::De));
        assert!(matches!(parse_language_code("unknown"), LanguageCode::EnUs)); // Defaults to English
    }

    fn test_global_options(quiet: bool) -> GlobalOptions {
        GlobalOptions {
            config: None,
            verbose: 0,
            quiet,
            format: None,
            voice: None,
            gpu: false,
            threads: None,
        }
    }

    /// A dictionary that fails to parse must still be rejected with a real error
    /// (unaffected by the fail-closed training guard: this is input validation).
    #[tokio::test]
    async fn test_missing_dictionary_is_rejected() {
        let temp = tempfile::tempdir().expect("failed to create temp dir");
        let missing = temp.path().join("does-not-exist.dict");
        let output = temp.path().join("out").join("g2p.safetensors");

        let result = run_train_g2p(
            "en".to_string(),
            missing,
            output,
            None,
            5,
            0.001,
            &test_global_options(true),
        )
        .await;

        assert!(result.is_err(), "missing dictionary must be rejected");
    }

    /// G2P training must fail closed (never fabricate a "trained" model file) and
    /// must explain why, even when given a well-formed dictionary.
    #[tokio::test]
    async fn test_g2p_training_fails_closed_without_fabricating_a_model() {
        let temp = tempfile::tempdir().expect("failed to create temp dir");
        let dict_path = temp.path().join("test.dict");
        std::fs::write(&dict_path, "hello HH AH L OW\nworld W ER L D\n")
            .expect("failed to write dictionary");
        let output_dir = temp.path().join("out");
        let output = output_dir.join("g2p.safetensors");

        let result = run_train_g2p(
            "en".to_string(),
            dict_path,
            output,
            None,
            5,
            0.001,
            &test_global_options(true),
        )
        .await;

        assert!(
            result.is_err(),
            "G2P training must fail closed instead of fabricating a trained model"
        );
        let message = result.unwrap_err().to_string();
        assert!(
            message.contains("backward pass"),
            "error should explain the real blocker (no backward pass), got: {message}"
        );
        assert!(
            message.contains("2 entries") || message.contains("entries=2"),
            "error should reflect the real parsed dictionary size, got: {message}"
        );

        // No model file should ever be fabricated for an untrained model.
        assert!(
            !output_dir.exists() || std::fs::read_dir(&output_dir).unwrap().next().is_none(),
            "must not write any model file for a model that was never actually trained"
        );
    }
}

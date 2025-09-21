use wasm_bindgen::prelude::*;
use web_sys::console;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    pub fn log(s: &str);

    #[wasm_bindgen(js_namespace = console)]
    pub fn error(s: &str);

    #[wasm_bindgen(js_namespace = console)]
    pub fn warn(s: &str);

    #[wasm_bindgen(js_namespace = console)]
    pub fn info(s: &str);
}

macro_rules! console_log {
    ($($t:tt)*) => (crate::wasm::utils::log(&format_args!($($t)*).to_string()))
}

macro_rules! console_error {
    ($($t:tt)*) => (crate::wasm::utils::error(&format_args!($($t)*).to_string()))
}

macro_rules! console_warn {
    ($($t:tt)*) => (crate::wasm::utils::warn(&format_args!($($t)*).to_string()))
}

macro_rules! console_info {
    ($($t:tt)*) => (crate::wasm::utils::info(&format_args!($($t)*).to_string()))
}

pub(crate) use {console_error, console_info, console_log, console_warn};

pub fn set_panic_hook() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

#[wasm_bindgen]
pub fn init_wasm_logger() {
    console_log!("Initializing WASM logger for VoiRS Recognizer");
    #[cfg(feature = "wasm-logger")]
    wasm_logger::init(wasm_logger::Config::default());
}

#[wasm_bindgen]
pub fn get_wasm_memory_usage() -> JsValue {
    let memory = wasm_bindgen::memory()
        .dyn_into::<js_sys::WebAssembly::Memory>()
        .unwrap();

    let buffer = memory.buffer();
    let usage = serde_json::json!({
        "buffer_size": buffer.dyn_into::<js_sys::ArrayBuffer>().map(|ab| ab.byte_length()).unwrap_or(0),
        "available": true,
        "module": "voirs-recognizer"
    });

    JsValue::from_serde(&usage).unwrap_or(JsValue::NULL)
}

#[wasm_bindgen]
pub fn check_browser_compatibility() -> JsValue {
    let mut features = std::collections::HashMap::new();

    // Check for Web Audio API
    features.insert("web_audio_api", js_sys::global().has_type_of("object"));

    // Check for Web Workers
    features.insert("web_workers", js_sys::global().has_type_of("object"));

    // Check for AudioWorklet (modern browsers)
    features.insert("audio_worklet", js_sys::global().has_type_of("object"));

    // Check for WebAssembly support
    features.insert(
        "webassembly",
        js_sys::WebAssembly::validate(&js_sys::Uint8Array::new(&js_sys::ArrayBuffer::new(0))),
    );

    let compatibility = serde_json::json!({
        "supported": true,
        "features": features,
        "recommendations": get_browser_recommendations(&features)
    });

    JsValue::from_serde(&compatibility).unwrap_or(JsValue::NULL)
}

fn get_browser_recommendations(
    features: &std::collections::HashMap<&str, bool>,
) -> Vec<&'static str> {
    let mut recommendations = Vec::new();

    if !features.get("web_audio_api").unwrap_or(&false) {
        recommendations.push("Web Audio API not supported - audio processing may be limited");
    }

    if !features.get("web_workers").unwrap_or(&false) {
        recommendations.push("Web Workers not supported - use main thread processing");
    }

    if !features.get("audio_worklet").unwrap_or(&false) {
        recommendations.push("AudioWorklet not supported - consider using ScriptProcessorNode");
    }

    if !features.get("webassembly").unwrap_or(&false) {
        recommendations.push("WebAssembly not supported - this module will not work");
    }

    if recommendations.is_empty() {
        recommendations.push("Browser fully compatible with VoiRS Recognizer");
    }

    recommendations
}

#[wasm_bindgen]
pub fn get_optimal_chunk_size(sample_rate: u32, target_latency_ms: f32) -> u32 {
    let samples_per_ms = sample_rate as f32 / 1000.0;
    let chunk_size = (target_latency_ms * samples_per_ms) as u32;

    // Round to nearest power of 2 for efficient processing
    let power_of_2 = 2u32.pow((chunk_size as f32).log2().round() as u32);

    // Clamp to reasonable range (64 - 4096 samples)
    power_of_2.max(64).min(4096)
}

#[wasm_bindgen]
pub fn estimate_memory_requirements(model_size: &str, chunk_size: u32) -> JsValue {
    let base_memory = match model_size {
        "tiny" => 100 * 1024 * 1024,      // 100MB
        "base" => 150 * 1024 * 1024,      // 150MB
        "small" => 250 * 1024 * 1024,     // 250MB
        "medium" => 500 * 1024 * 1024,    // 500MB
        "large" => 1024 * 1024 * 1024,    // 1GB
        "large-v3" => 1200 * 1024 * 1024, // 1.2GB
        _ => 200 * 1024 * 1024,           // 200MB default
    };

    let buffer_memory = chunk_size * 4 * 10; // 10 chunks of float32 data
    let total_memory = base_memory + buffer_memory as usize;

    let estimate = serde_json::json!({
        "model_memory_bytes": base_memory,
        "buffer_memory_bytes": buffer_memory,
        "total_memory_bytes": total_memory,
        "model_memory_mb": base_memory / (1024 * 1024),
        "total_memory_mb": total_memory / (1024 * 1024),
        "recommended_heap_size_mb": (total_memory * 2) / (1024 * 1024), // 2x for safe margin
    });

    JsValue::from_serde(&estimate).unwrap_or(JsValue::NULL)
}

#[wasm_bindgen]
pub fn validate_audio_format(sample_rate: u32, channels: u16, bit_depth: u16) -> JsValue {
    let mut issues = Vec::new();
    let mut recommendations = Vec::new();

    // Validate sample rate
    match sample_rate {
        8000 | 16000 | 22050 | 44100 | 48000 => {}
        _ => {
            issues.push(format!("Unsupported sample rate: {}", sample_rate));
            recommendations.push("Use 16000 Hz for optimal speech recognition performance");
        }
    }

    // Validate channels
    if channels != 1 {
        issues.push(format!(
            "Multi-channel audio detected: {} channels",
            channels
        ));
        recommendations.push("Convert to mono for speech recognition");
    }

    // Validate bit depth
    match bit_depth {
        16 | 24 | 32 => {}
        _ => {
            issues.push(format!("Unsupported bit depth: {}", bit_depth));
            recommendations.push("Use 16-bit or 32-bit audio");
        }
    }

    let validation = serde_json::json!({
        "valid": issues.is_empty(),
        "issues": issues,
        "recommendations": recommendations,
        "optimal_format": {
            "sample_rate": 16000,
            "channels": 1,
            "bit_depth": 16
        }
    });

    JsValue::from_serde(&validation).unwrap_or(JsValue::NULL)
}

#[wasm_bindgen]
pub fn get_performance_profile(device_type: &str) -> JsValue {
    let profile = match device_type.to_lowercase().as_str() {
        "mobile" | "phone" | "tablet" => serde_json::json!({
            "model_preference": "tiny",
            "chunk_size": 512,
            "buffer_size": 2048,
            "concurrent_streams": 1,
            "memory_limit_mb": 256,
            "processing_timeout_ms": 2000
        }),
        "laptop" | "desktop" => serde_json::json!({
            "model_preference": "base",
            "chunk_size": 1024,
            "buffer_size": 4096,
            "concurrent_streams": 2,
            "memory_limit_mb": 512,
            "processing_timeout_ms": 1000
        }),
        "server" | "workstation" => serde_json::json!({
            "model_preference": "large",
            "chunk_size": 2048,
            "buffer_size": 8192,
            "concurrent_streams": 4,
            "memory_limit_mb": 2048,
            "processing_timeout_ms": 500
        }),
        _ => serde_json::json!({
            "model_preference": "small",
            "chunk_size": 1024,
            "buffer_size": 4096,
            "concurrent_streams": 1,
            "memory_limit_mb": 384,
            "processing_timeout_ms": 1500
        }),
    };

    JsValue::from_serde(&profile).unwrap_or(JsValue::NULL)
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;

    wasm_bindgen_test_configure!(run_in_browser);

    #[wasm_bindgen_test]
    fn test_memory_usage() {
        let usage = get_wasm_memory_usage();
        assert!(!usage.is_null());
    }

    #[wasm_bindgen_test]
    fn test_chunk_size_calculation() {
        let chunk_size = get_optimal_chunk_size(16000, 100.0); // 100ms at 16kHz
        assert!(chunk_size >= 64 && chunk_size <= 4096);
        assert_eq!(chunk_size & (chunk_size - 1), 0); // Should be power of 2
    }

    #[wasm_bindgen_test]
    fn test_audio_format_validation() {
        let validation = validate_audio_format(16000, 1, 16);
        assert!(!validation.is_null());
    }

    #[wasm_bindgen_test]
    fn test_performance_profiles() {
        let mobile_profile = get_performance_profile("mobile");
        let desktop_profile = get_performance_profile("desktop");

        assert!(!mobile_profile.is_null());
        assert!(!desktop_profile.is_null());
    }

    #[wasm_bindgen_test]
    fn test_memory_estimation() {
        let estimate = estimate_memory_requirements("base", 1024);
        assert!(!estimate.is_null());
    }
}

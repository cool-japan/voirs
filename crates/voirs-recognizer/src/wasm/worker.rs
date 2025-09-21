use crate::wasm::recognizer::WasmVoirsRecognizer;
use crate::wasm::utils::{console_error, console_log};
use js_sys::{Array, Object, Reflect};
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;
use web_sys::{DedicatedWorkerGlobalScope, MessageEvent, Worker};

#[derive(Serialize, Deserialize)]
pub struct WorkerMessage {
    pub id: String,
    pub command: String,
    pub data: serde_json::Value,
}

#[derive(Serialize, Deserialize)]
pub struct WorkerResponse {
    pub id: String,
    pub success: bool,
    pub data: Option<serde_json::Value>,
    pub error: Option<String>,
}

#[wasm_bindgen]
pub struct WasmRecognizerWorker {
    recognizer: WasmVoirsRecognizer,
    worker_scope: Option<DedicatedWorkerGlobalScope>,
}

#[wasm_bindgen]
impl WasmRecognizerWorker {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        crate::wasm::utils::set_panic_hook();

        let worker_scope = js_sys::global()
            .dyn_into::<DedicatedWorkerGlobalScope>()
            .ok();

        Self {
            recognizer: WasmVoirsRecognizer::new(),
            worker_scope,
        }
    }

    #[wasm_bindgen]
    pub fn start_worker(&self) {
        if let Some(scope) = &self.worker_scope {
            let closure = Closure::wrap(Box::new(move |event: MessageEvent| {
                // Handle messages from main thread
                if let Ok(data) = event.data().into_serde::<WorkerMessage>() {
                    // Process the worker message
                    // This would be implemented with async handling
                    console_log!("Worker received message: {}", data.command);
                }
            }) as Box<dyn FnMut(_)>);

            scope.set_onmessage(Some(closure.as_ref().unchecked_ref()));
            closure.forget();
        }
    }

    #[wasm_bindgen]
    pub async fn handle_message(&mut self, message: JsValue) -> Result<JsValue, JsValue> {
        let worker_msg: WorkerMessage = message
            .into_serde()
            .map_err(|e| JsValue::from_str(&format!("Message parsing error: {e}")))?;

        let response = match worker_msg.command.as_str() {
            "initialize" => {
                match self
                    .recognizer
                    .initialize(JsValue::from_serde(&worker_msg.data).unwrap())
                    .await
                {
                    Ok(_) => WorkerResponse {
                        id: worker_msg.id,
                        success: true,
                        data: Some(serde_json::json!({"initialized": true})),
                        error: None,
                    },
                    Err(e) => WorkerResponse {
                        id: worker_msg.id,
                        success: false,
                        data: None,
                        error: Some(format!("{:?}", e)),
                    },
                }
            }
            "recognize" => {
                // Extract audio data from the message
                if let Some(audio_data) = worker_msg.data.as_array() {
                    let audio_bytes: Vec<u8> = audio_data
                        .iter()
                        .filter_map(|v| v.as_u64().map(|n| n as u8))
                        .collect();

                    match self.recognizer.recognize_audio(&audio_bytes).await {
                        Ok(result) => WorkerResponse {
                            id: worker_msg.id,
                            success: true,
                            data: result.into_serde().ok(),
                            error: None,
                        },
                        Err(e) => WorkerResponse {
                            id: worker_msg.id,
                            success: false,
                            data: None,
                            error: Some(format!("{:?}", e)),
                        },
                    }
                } else {
                    WorkerResponse {
                        id: worker_msg.id,
                        success: false,
                        data: None,
                        error: Some("Invalid audio data format".to_string()),
                    }
                }
            }
            "stream_start" => {
                match self
                    .recognizer
                    .start_streaming(JsValue::from_serde(&worker_msg.data).unwrap())
                    .await
                {
                    Ok(_) => WorkerResponse {
                        id: worker_msg.id,
                        success: true,
                        data: Some(serde_json::json!({"streaming": true})),
                        error: None,
                    },
                    Err(e) => WorkerResponse {
                        id: worker_msg.id,
                        success: false,
                        data: None,
                        error: Some(format!("{:?}", e)),
                    },
                }
            }
            "stream_audio" => {
                if let Some(audio_data) = worker_msg.data.as_array() {
                    let audio_bytes: Vec<u8> = audio_data
                        .iter()
                        .filter_map(|v| v.as_u64().map(|n| n as u8))
                        .collect();

                    match self.recognizer.stream_audio(&audio_bytes).await {
                        Ok(result) => WorkerResponse {
                            id: worker_msg.id,
                            success: true,
                            data: result.into_serde().ok(),
                            error: None,
                        },
                        Err(e) => WorkerResponse {
                            id: worker_msg.id,
                            success: false,
                            data: None,
                            error: Some(format!("{:?}", e)),
                        },
                    }
                } else {
                    WorkerResponse {
                        id: worker_msg.id,
                        success: false,
                        data: None,
                        error: Some("Invalid audio data format".to_string()),
                    }
                }
            }
            "stream_stop" => {
                self.recognizer.stop_streaming();
                WorkerResponse {
                    id: worker_msg.id,
                    success: true,
                    data: Some(serde_json::json!({"streaming": false})),
                    error: None,
                }
            }
            "get_capabilities" => {
                let capabilities = self.recognizer.get_capabilities();
                WorkerResponse {
                    id: worker_msg.id,
                    success: true,
                    data: capabilities.into_serde().ok(),
                    error: None,
                }
            }
            "get_models" => match self.recognizer.get_supported_models().await {
                Ok(models) => WorkerResponse {
                    id: worker_msg.id,
                    success: true,
                    data: models.into_serde().ok(),
                    error: None,
                },
                Err(e) => WorkerResponse {
                    id: worker_msg.id,
                    success: false,
                    data: None,
                    error: Some(format!("{:?}", e)),
                },
            },
            "get_languages" => match self.recognizer.get_supported_languages().await {
                Ok(languages) => WorkerResponse {
                    id: worker_msg.id,
                    success: true,
                    data: languages.into_serde().ok(),
                    error: None,
                },
                Err(e) => WorkerResponse {
                    id: worker_msg.id,
                    success: false,
                    data: None,
                    error: Some(format!("{:?}", e)),
                },
            },
            "switch_model" => {
                if let Some(model_name) = worker_msg.data.as_str() {
                    match self.recognizer.switch_model(model_name).await {
                        Ok(_) => WorkerResponse {
                            id: worker_msg.id,
                            success: true,
                            data: Some(serde_json::json!({"model": model_name})),
                            error: None,
                        },
                        Err(e) => WorkerResponse {
                            id: worker_msg.id,
                            success: false,
                            data: None,
                            error: Some(format!("{:?}", e)),
                        },
                    }
                } else {
                    WorkerResponse {
                        id: worker_msg.id,
                        success: false,
                        data: None,
                        error: Some("Model name required".to_string()),
                    }
                }
            }
            _ => WorkerResponse {
                id: worker_msg.id,
                success: false,
                data: None,
                error: Some(format!("Unknown command: {}", worker_msg.command)),
            },
        };

        JsValue::from_serde(&response)
            .map_err(|e| JsValue::from_str(&format!("Response serialization error: {e}")))
    }

    #[wasm_bindgen]
    pub fn post_response(&self, response: &JsValue) {
        if let Some(scope) = &self.worker_scope {
            if let Err(e) = scope.post_message(response) {
                console_error!("Failed to post message: {:?}", e);
            }
        }
    }
}

impl Default for WasmRecognizerWorker {
    fn default() -> Self {
        Self::new()
    }
}

// Helper functions for web worker integration

#[wasm_bindgen]
pub fn create_recognizer_worker_script() -> String {
    r#"
// Web Worker script for VoiRS Recognizer
importScripts('./pkg/voirs_recognizer.js');

let wasmModule;
let recognizerWorker;

async function initializeWasm() {
    wasmModule = await import('./pkg/voirs_recognizer.js');
    await wasmModule.default(); // Initialize WASM
    
    recognizerWorker = new wasmModule.WasmRecognizerWorker();
    recognizerWorker.start_worker();
    
    console.log('VoiRS Recognizer Worker initialized');
}

self.onmessage = async function(event) {
    if (!recognizerWorker) {
        await initializeWasm();
    }
    
    try {
        const response = await recognizerWorker.handle_message(event.data);
        self.postMessage(response);
    } catch (error) {
        self.postMessage({
            id: event.data.id || 'unknown',
            success: false,
            data: null,
            error: error.toString()
        });
    }
};

// Handle worker errors
self.onerror = function(error) {
    console.error('Worker error:', error);
    self.postMessage({
        id: 'error',
        success: false,
        data: null,
        error: error.toString()
    });
};

console.log('VoiRS Recognizer Worker script loaded');
"#
    .to_string()
}

#[wasm_bindgen]
pub fn get_worker_capabilities() -> JsValue {
    let capabilities = serde_json::json!({
        "web_workers": true,
        "offscreen_processing": true,
        "concurrent_recognition": true,
        "background_processing": true,
        "memory_isolation": true,
        "commands": [
            "initialize",
            "recognize",
            "stream_start",
            "stream_audio",
            "stream_stop",
            "get_capabilities",
            "get_models",
            "get_languages",
            "switch_model"
        ]
    });

    JsValue::from_serde(&capabilities).unwrap_or(JsValue::NULL)
}

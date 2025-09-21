//! OpenAPI documentation generation for the VoiRS recognizer REST API.

use serde_json::{json, Value};
use std::collections::HashMap;

/// Generate OpenAPI 3.0 specification for the VoiRS recognizer API
pub fn generate_openapi_spec() -> Value {
    json!({
        "openapi": "3.0.3",
        "info": {
            "title": "VoiRS Speech Recognition API",
            "description": "A comprehensive speech recognition API powered by VoiRS (Voice Recognition System). Supports multiple models, real-time streaming, and advanced audio processing features.",
            "version": env!("CARGO_PKG_VERSION"),
            "contact": {
                "name": "VoiRS Team",
                "url": "https://github.com/cool-japan/voirs",
                "email": "support@voirs.dev"
            },
            "license": {
                "name": "MIT",
                "url": "https://opensource.org/licenses/MIT"
            }
        },
        "servers": [
            {
                "url": "http://localhost:8080",
                "description": "Local development server"
            },
            {
                "url": "https://api.voirs.dev",
                "description": "Production server"
            }
        ],
        "tags": [
            {
                "name": "health",
                "description": "Health check and system status endpoints"
            },
            {
                "name": "recognition",
                "description": "Speech recognition endpoints"
            },
            {
                "name": "models",
                "description": "Model management endpoints"
            },
            {
                "name": "streaming",
                "description": "Real-time streaming recognition"
            },
            {
                "name": "websocket",
                "description": "WebSocket streaming endpoints"
            }
        ],
        "paths": generate_paths(),
        "components": {
            "schemas": generate_schemas(),
            "securitySchemes": generate_security_schemes(),
            "responses": generate_common_responses()
        },
        "security": [
            {
                "ApiKeyAuth": []
            },
            {
                "BearerAuth": []
            }
        ]
    })
}

/// Generate API paths
fn generate_paths() -> Value {
    json!({
        "/health": {
            "get": {
                "tags": ["health"],
                "summary": "Basic health check",
                "description": "Returns basic system health information",
                "operationId": "getHealth",
                "responses": {
                    "200": {
                        "description": "System is healthy",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/HealthResponse"
                                }
                            }
                        }
                    }
                }
            }
        },
        "/health/detailed": {
            "get": {
                "tags": ["health"],
                "summary": "Detailed health check",
                "description": "Returns comprehensive system health and configuration information",
                "operationId": "getDetailedHealth",
                "responses": {
                    "200": {
                        "description": "Detailed system information",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "additionalProperties": true
                                }
                            }
                        }
                    }
                }
            }
        },
        "/health/ready": {
            "get": {
                "tags": ["health"],
                "summary": "Readiness check",
                "description": "Kubernetes readiness probe endpoint",
                "operationId": "getReadiness",
                "responses": {
                    "200": {
                        "description": "Service is ready",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "additionalProperties": {
                                        "type": "boolean"
                                    }
                                }
                            }
                        }
                    },
                    "503": {
                        "description": "Service is not ready"
                    }
                }
            }
        },
        "/health/live": {
            "get": {
                "tags": ["health"],
                "summary": "Liveness check",
                "description": "Kubernetes liveness probe endpoint",
                "operationId": "getLiveness",
                "responses": {
                    "200": {
                        "description": "Service is alive",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "additionalProperties": {
                                        "type": "boolean"
                                    }
                                }
                            }
                        }
                    },
                    "500": {
                        "description": "Service is not responding"
                    }
                }
            }
        },
        "/recognize": {
            "post": {
                "tags": ["recognition"],
                "summary": "Recognize speech from audio",
                "description": "Transcribe speech from audio data (base64 encoded) or URL",
                "operationId": "recognizeAudio",
                "requestBody": {
                    "required": true,
                    "content": {
                        "application/json": {
                            "schema": {
                                "$ref": "#/components/schemas/RecognitionRequest"
                            }
                        }
                    }
                },
                "responses": {
                    "200": {
                        "description": "Recognition successful",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/RecognitionResponse"
                                }
                            }
                        }
                    },
                    "400": {
                        "$ref": "#/components/responses/BadRequest"
                    },
                    "500": {
                        "$ref": "#/components/responses/InternalError"
                    }
                },
                "security": [
                    {
                        "ApiKeyAuth": []
                    },
                    {
                        "BearerAuth": []
                    }
                ]
            }
        },
        "/recognize/batch": {
            "post": {
                "tags": ["recognition"],
                "summary": "Batch recognition",
                "description": "Process multiple audio files in batch",
                "operationId": "batchRecognize",
                "requestBody": {
                    "required": true,
                    "content": {
                        "application/json": {
                            "schema": {
                                "$ref": "#/components/schemas/BatchRecognitionRequest"
                            }
                        }
                    }
                },
                "responses": {
                    "200": {
                        "description": "Batch processing initiated",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/BatchRecognitionResponse"
                                }
                            }
                        }
                    },
                    "501": {
                        "description": "Not implemented yet"
                    }
                }
            }
        },
        "/models": {
            "get": {
                "tags": ["models"],
                "summary": "List available models",
                "description": "Get information about all available speech recognition models",
                "operationId": "listModels",
                "responses": {
                    "200": {
                        "description": "List of models",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/ModelStatusResponse"
                                }
                            }
                        }
                    }
                }
            }
        },
        "/models/{modelName}": {
            "get": {
                "tags": ["models"],
                "summary": "Get model information",
                "description": "Get detailed information about a specific model",
                "operationId": "getModelInfo",
                "parameters": [
                    {
                        "name": "modelName",
                        "in": "path",
                        "required": true,
                        "schema": {
                            "type": "string"
                        },
                        "description": "Name of the model"
                    }
                ],
                "responses": {
                    "200": {
                        "description": "Model information",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/ModelInfoResponse"
                                }
                            }
                        }
                    },
                    "404": {
                        "description": "Model not found"
                    }
                }
            }
        },
        "/models/{modelName}/load": {
            "post": {
                "tags": ["models"],
                "summary": "Load a model",
                "description": "Load a specific speech recognition model",
                "operationId": "loadModel",
                "parameters": [
                    {
                        "name": "modelName",
                        "in": "path",
                        "required": true,
                        "schema": {
                            "type": "string"
                        },
                        "description": "Name of the model to load"
                    }
                ],
                "responses": {
                    "200": {
                        "description": "Model loaded successfully",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/ModelManagementResponse"
                                }
                            }
                        }
                    },
                    "501": {
                        "description": "Not implemented yet"
                    }
                }
            }
        },
        "/ws": {
            "get": {
                "tags": ["websocket"],
                "summary": "WebSocket streaming endpoint",
                "description": "Establish WebSocket connection for real-time speech recognition",
                "operationId": "connectWebSocket",
                "parameters": [
                    {
                        "name": "session_id",
                        "in": "query",
                        "required": false,
                        "schema": {
                            "type": "string"
                        },
                        "description": "Optional session ID"
                    },
                    {
                        "name": "model",
                        "in": "query",
                        "required": false,
                        "schema": {
                            "type": "string"
                        },
                        "description": "Model to use for recognition"
                    },
                    {
                        "name": "language",
                        "in": "query",
                        "required": false,
                        "schema": {
                            "type": "string"
                        },
                        "description": "Language code"
                    },
                    {
                        "name": "interim_results",
                        "in": "query",
                        "required": false,
                        "schema": {
                            "type": "boolean"
                        },
                        "description": "Enable interim results"
                    }
                ],
                "responses": {
                    "101": {
                        "description": "WebSocket connection established"
                    },
                    "400": {
                        "description": "Invalid WebSocket request"
                    }
                }
            }
        },
        "/ws/sessions": {
            "get": {
                "tags": ["websocket"],
                "summary": "List active WebSocket sessions",
                "description": "Get information about all active streaming sessions",
                "operationId": "listActiveSessions",
                "responses": {
                    "200": {
                        "description": "List of active sessions",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "additionalProperties": true
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    })
}

/// Generate schema definitions
fn generate_schemas() -> Value {
    json!({
        "ApiResponse": {
            "type": "object",
            "properties": {
                "success": {
                    "type": "boolean",
                    "description": "Whether the request was successful"
                },
                "data": {
                    "description": "Response data (if successful)"
                },
                "error": {
                    "type": "string",
                    "description": "Error message (if unsuccessful)"
                },
                "request_id": {
                    "type": "string",
                    "description": "Unique request identifier"
                },
                "timestamp": {
                    "type": "string",
                    "format": "date-time",
                    "description": "Response timestamp"
                }
            },
            "required": ["success", "request_id", "timestamp"]
        },
        "HealthResponse": {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "description": "Overall health status",
                    "enum": ["healthy", "degraded", "unhealthy"]
                },
                "version": {
                    "type": "string",
                    "description": "Service version"
                },
                "uptime_seconds": {
                    "type": "number",
                    "description": "Service uptime in seconds"
                },
                "memory_usage": {
                    "$ref": "#/components/schemas/MemoryUsageResponse"
                },
                "model_status": {
                    "$ref": "#/components/schemas/ModelStatusResponse"
                },
                "performance_metrics": {
                    "$ref": "#/components/schemas/PerformanceMetricsResponse"
                }
            },
            "required": ["status", "version", "uptime_seconds"]
        },
        "MemoryUsageResponse": {
            "type": "object",
            "properties": {
                "used_mb": {
                    "type": "number",
                    "description": "Used memory in MB"
                },
                "available_mb": {
                    "type": "number",
                    "description": "Available memory in MB"
                },
                "cache_size_mb": {
                    "type": "number",
                    "description": "Cache size in MB"
                },
                "usage_percent": {
                    "type": "number",
                    "description": "Memory usage percentage"
                }
            }
        },
        "ModelStatusResponse": {
            "type": "object",
            "properties": {
                "loaded_models": {
                    "type": "array",
                    "items": {
                        "$ref": "#/components/schemas/ModelInfoResponse"
                    }
                },
                "loading_status": {
                    "type": "string"
                },
                "default_model": {
                    "type": "string"
                },
                "supported_models": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    }
                },
                "supported_languages": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    }
                }
            }
        },
        "ModelInfoResponse": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Model name"
                },
                "model_type": {
                    "type": "string",
                    "description": "Model type (e.g., whisper, deepspeech)"
                },
                "size_mb": {
                    "type": "number",
                    "description": "Model size in MB"
                },
                "is_loaded": {
                    "type": "boolean",
                    "description": "Whether the model is currently loaded"
                },
                "languages": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "Supported languages"
                },
                "version": {
                    "type": "string",
                    "description": "Model version"
                }
            }
        },
        "PerformanceMetricsResponse": {
            "type": "object",
            "properties": {
                "total_recognitions": {
                    "type": "integer",
                    "description": "Total number of recognitions performed"
                },
                "total_audio_duration": {
                    "type": "number",
                    "description": "Total audio duration processed (seconds)"
                },
                "avg_processing_time_ms": {
                    "type": "number",
                    "description": "Average processing time in milliseconds"
                },
                "real_time_factor": {
                    "type": "number",
                    "description": "Real-time factor (processing time / audio duration)"
                },
                "active_sessions": {
                    "type": "integer",
                    "description": "Number of active streaming sessions"
                },
                "cache_hit_rate": {
                    "type": "number",
                    "description": "Cache hit rate (0.0 to 1.0)"
                },
                "error_rate": {
                    "type": "number",
                    "description": "Error rate (0.0 to 1.0)"
                }
            }
        },
        "RecognitionRequest": {
            "type": "object",
            "properties": {
                "audio_data": {
                    "type": "string",
                    "description": "Base64 encoded audio data"
                },
                "audio_url": {
                    "type": "string",
                    "description": "URL to audio file (alternative to audio_data)"
                },
                "audio_format": {
                    "$ref": "#/components/schemas/AudioFormatRequest"
                },
                "config": {
                    "$ref": "#/components/schemas/RecognitionConfigRequest"
                },
                "include_segments": {
                    "type": "boolean",
                    "description": "Whether to return detailed segments"
                },
                "include_confidence": {
                    "type": "boolean",
                    "description": "Whether to include confidence scores"
                },
                "include_timestamps": {
                    "type": "boolean",
                    "description": "Whether to include timestamps"
                }
            }
        },
        "AudioFormatRequest": {
            "type": "object",
            "properties": {
                "sample_rate": {
                    "type": "integer",
                    "description": "Sample rate in Hz"
                },
                "channels": {
                    "type": "integer",
                    "description": "Number of channels"
                },
                "bits_per_sample": {
                    "type": "integer",
                    "description": "Bits per sample"
                },
                "format": {
                    "type": "string",
                    "enum": ["wav", "mp3", "flac", "ogg", "m4a"],
                    "description": "Audio format"
                }
            }
        },
        "RecognitionConfigRequest": {
            "type": "object",
            "properties": {
                "model": {
                    "type": "string",
                    "description": "Model name to use"
                },
                "language": {
                    "type": "string",
                    "description": "Language code"
                },
                "enable_vad": {
                    "type": "boolean",
                    "description": "Enable voice activity detection"
                },
                "confidence_threshold": {
                    "type": "number",
                    "description": "Confidence threshold (0.0 to 1.0)"
                },
                "beam_size": {
                    "type": "integer",
                    "description": "Beam size for decoding"
                },
                "temperature": {
                    "type": "number",
                    "description": "Temperature for sampling"
                },
                "suppress_blank": {
                    "type": "boolean",
                    "description": "Whether to suppress blank tokens"
                },
                "suppress_tokens": {
                    "type": "array",
                    "items": {
                        "type": "integer"
                    },
                    "description": "List of token IDs to suppress"
                }
            }
        },
        "RecognitionResponse": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Recognized text"
                },
                "confidence": {
                    "type": "number",
                    "description": "Overall confidence score"
                },
                "detected_language": {
                    "type": "string",
                    "description": "Detected language (if auto-detection was used)"
                },
                "processing_time_ms": {
                    "type": "number",
                    "description": "Processing time in milliseconds"
                },
                "audio_duration_s": {
                    "type": "number",
                    "description": "Audio duration in seconds"
                },
                "segment_count": {
                    "type": "integer",
                    "description": "Number of segments"
                },
                "segments": {
                    "type": "array",
                    "items": {
                        "$ref": "#/components/schemas/SegmentResponse"
                    },
                    "description": "Detailed segments (if requested)"
                },
                "audio_metadata": {
                    "$ref": "#/components/schemas/AudioMetadataResponse"
                },
                "metadata": {
                    "$ref": "#/components/schemas/RecognitionMetadataResponse"
                }
            },
            "required": ["text", "confidence", "processing_time_ms", "audio_duration_s", "segment_count"]
        },
        "SegmentResponse": {
            "type": "object",
            "properties": {
                "start_time": {
                    "type": "number",
                    "description": "Segment start time in seconds"
                },
                "end_time": {
                    "type": "number",
                    "description": "Segment end time in seconds"
                },
                "text": {
                    "type": "string",
                    "description": "Segment text"
                },
                "confidence": {
                    "type": "number",
                    "description": "Segment confidence score"
                },
                "no_speech_prob": {
                    "type": "number",
                    "description": "No speech probability"
                },
                "tokens": {
                    "type": "array",
                    "items": {
                        "$ref": "#/components/schemas/TokenResponse"
                    },
                    "description": "Token-level information"
                }
            }
        },
        "TokenResponse": {
            "type": "object",
            "properties": {
                "id": {
                    "type": "integer",
                    "description": "Token ID"
                },
                "text": {
                    "type": "string",
                    "description": "Token text"
                },
                "probability": {
                    "type": "number",
                    "description": "Token probability"
                },
                "start_time": {
                    "type": "number",
                    "description": "Token start time"
                },
                "end_time": {
                    "type": "number",
                    "description": "Token end time"
                }
            }
        },
        "AudioMetadataResponse": {
            "type": "object",
            "properties": {
                "sample_rate": {
                    "type": "integer",
                    "description": "Sample rate in Hz"
                },
                "channels": {
                    "type": "integer",
                    "description": "Number of channels"
                },
                "duration": {
                    "type": "number",
                    "description": "Duration in seconds"
                },
                "format": {
                    "type": "string",
                    "description": "Audio format"
                },
                "size_bytes": {
                    "type": "integer",
                    "description": "File size in bytes"
                },
                "bit_rate": {
                    "type": "integer",
                    "description": "Bit rate (if applicable)"
                }
            }
        },
        "RecognitionMetadataResponse": {
            "type": "object",
            "properties": {
                "model": {
                    "type": "string",
                    "description": "Model used for recognition"
                },
                "language": {
                    "type": "string",
                    "description": "Language code used"
                },
                "vad_enabled": {
                    "type": "boolean",
                    "description": "Whether VAD was enabled"
                },
                "beam_size": {
                    "type": "integer",
                    "description": "Beam size used"
                },
                "temperature": {
                    "type": "number",
                    "description": "Temperature used"
                },
                "processing_stats": {
                    "$ref": "#/components/schemas/ProcessingStatsResponse"
                }
            }
        },
        "ProcessingStatsResponse": {
            "type": "object",
            "properties": {
                "real_time_factor": {
                    "type": "number",
                    "description": "Real-time factor"
                },
                "memory_usage_mb": {
                    "type": "number",
                    "description": "Memory usage in MB"
                },
                "cpu_usage_percent": {
                    "type": "number",
                    "description": "CPU usage percentage"
                },
                "gpu_usage_percent": {
                    "type": "number",
                    "description": "GPU usage percentage"
                }
            }
        },
        "BatchRecognitionRequest": {
            "type": "object",
            "properties": {
                "inputs": {
                    "type": "array",
                    "items": {
                        "$ref": "#/components/schemas/RecognitionRequest"
                    },
                    "description": "List of recognition requests"
                },
                "config": {
                    "$ref": "#/components/schemas/BatchConfigRequest"
                },
                "parallel": {
                    "type": "boolean",
                    "description": "Whether to process in parallel"
                },
                "max_concurrency": {
                    "type": "integer",
                    "description": "Maximum number of concurrent jobs"
                }
            },
            "required": ["inputs"]
        },
        "BatchConfigRequest": {
            "type": "object",
            "properties": {
                "default_config": {
                    "$ref": "#/components/schemas/RecognitionConfigRequest"
                },
                "continue_on_error": {
                    "type": "boolean",
                    "description": "Whether to continue on errors"
                },
                "timeout_per_job": {
                    "type": "integer",
                    "description": "Timeout per job in seconds"
                },
                "priority": {
                    "type": "string",
                    "description": "Priority level"
                }
            }
        },
        "BatchRecognitionResponse": {
            "type": "object",
            "properties": {
                "batch_id": {
                    "type": "string",
                    "description": "Batch job ID"
                },
                "results": {
                    "type": "array",
                    "items": {
                        "$ref": "#/components/schemas/BatchResultResponse"
                    }
                },
                "statistics": {
                    "$ref": "#/components/schemas/BatchStatisticsResponse"
                },
                "status": {
                    "type": "string",
                    "description": "Overall batch status"
                }
            }
        },
        "BatchResultResponse": {
            "type": "object",
            "properties": {
                "index": {
                    "type": "integer",
                    "description": "Input index"
                },
                "success": {
                    "type": "boolean",
                    "description": "Whether this result is successful"
                },
                "result": {
                    "$ref": "#/components/schemas/RecognitionResponse"
                },
                "error": {
                    "type": "string",
                    "description": "Error message (if failed)"
                },
                "processing_time_ms": {
                    "type": "number",
                    "description": "Processing time for this input"
                }
            }
        },
        "BatchStatisticsResponse": {
            "type": "object",
            "properties": {
                "total_inputs": {
                    "type": "integer"
                },
                "successful": {
                    "type": "integer"
                },
                "failed": {
                    "type": "integer"
                },
                "total_processing_time_ms": {
                    "type": "number"
                },
                "avg_processing_time_ms": {
                    "type": "number"
                },
                "start_time": {
                    "type": "string",
                    "format": "date-time"
                },
                "end_time": {
                    "type": "string",
                    "format": "date-time"
                }
            }
        },
        "ModelManagementResponse": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "Action performed"
                },
                "model_name": {
                    "type": "string",
                    "description": "Model name"
                },
                "success": {
                    "type": "boolean",
                    "description": "Whether the action was successful"
                },
                "message": {
                    "type": "string",
                    "description": "Status message"
                },
                "time_taken_ms": {
                    "type": "number",
                    "description": "Time taken for the action"
                }
            }
        },
        "ErrorResponse": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Error code"
                },
                "message": {
                    "type": "string",
                    "description": "Error message"
                },
                "details": {
                    "type": "object",
                    "additionalProperties": true,
                    "description": "Additional error details"
                },
                "request_id": {
                    "type": "string",
                    "description": "Request ID for tracking"
                },
                "timestamp": {
                    "type": "string",
                    "format": "date-time",
                    "description": "Error timestamp"
                }
            },
            "required": ["code", "message", "request_id", "timestamp"]
        }
    })
}

/// Generate security schemes
fn generate_security_schemes() -> Value {
    json!({
        "ApiKeyAuth": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key",
            "description": "API key authentication"
        },
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
            "description": "Bearer token authentication"
        }
    })
}

/// Generate common responses
fn generate_common_responses() -> Value {
    json!({
        "BadRequest": {
            "description": "Invalid request",
            "content": {
                "application/json": {
                    "schema": {
                        "$ref": "#/components/schemas/ErrorResponse"
                    }
                }
            }
        },
        "Unauthorized": {
            "description": "Authentication required",
            "content": {
                "application/json": {
                    "schema": {
                        "$ref": "#/components/schemas/ErrorResponse"
                    }
                }
            }
        },
        "NotFound": {
            "description": "Resource not found",
            "content": {
                "application/json": {
                    "schema": {
                        "$ref": "#/components/schemas/ErrorResponse"
                    }
                }
            }
        },
        "InternalError": {
            "description": "Internal server error",
            "content": {
                "application/json": {
                    "schema": {
                        "$ref": "#/components/schemas/ErrorResponse"
                    }
                }
            }
        }
    })
}

/// Generate OpenAPI specification as JSON string
pub fn generate_openapi_json() -> String {
    serde_json::to_string_pretty(&generate_openapi_spec()).unwrap_or_else(|_| "{}".to_string())
}

/// Generate OpenAPI specification as YAML string
pub fn generate_openapi_yaml() -> String {
    // For now, return JSON (would need yaml crate for proper YAML generation)
    generate_openapi_json()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_openapi_generation() {
        let spec = generate_openapi_spec();

        // Verify basic structure
        assert!(spec["openapi"].is_string());
        assert!(spec["info"].is_object());
        assert!(spec["paths"].is_object());
        assert!(spec["components"].is_object());

        // Verify specific paths exist
        assert!(spec["paths"]["/health"].is_object());
        assert!(spec["paths"]["/recognize"].is_object());
        assert!(spec["paths"]["/models"].is_object());
        assert!(spec["paths"]["/ws"].is_object());
    }

    #[test]
    fn test_json_generation() {
        let json = generate_openapi_json();
        assert!(!json.is_empty());
        assert!(serde_json::from_str::<Value>(&json).is_ok());
    }
}

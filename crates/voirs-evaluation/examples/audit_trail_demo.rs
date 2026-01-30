//! Example demonstrating the audit trail system for compliance and security
//!
//! This example shows how to use the audit trail system to log evaluation activities,
//! track user actions, and generate compliance reports.

use std::collections::HashMap;
use std::env;
use std::time::{Duration, SystemTime};
use voirs_evaluation::audit::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== VoiRS Audit Trail System Demo ===\n");

    // Create audit configuration
    let temp_dir = env::temp_dir().join("voirs_audit_demo");
    let config = AuditConfig {
        enabled: true,
        log_directory: temp_dir.clone(),
        max_file_size: 10 * 1024 * 1024, // 10MB
        rotation_policy: RotationPolicy::Daily,
        retention_days: 90,
        log_sensitive_data: false,
        encryption: None,
        tamper_protection: true,
        realtime_monitoring: false,
    };

    println!("✓ Audit configuration created:");
    println!("  - Log directory: {:?}", config.log_directory);
    println!("  - Rotation policy: {:?}", config.rotation_policy);
    println!("  - Tamper protection: {}", config.tamper_protection);
    println!("  - Retention: {} days\n", config.retention_days);

    // Create audit trail manager
    let trail = AuditTrail::new(config)?;
    println!("✓ Audit trail manager initialized\n");

    // Example 1: Log user authentication
    println!("Example 1: User Authentication Logging");
    println!("---------------------------------------");

    let user_actor = Actor {
        id: "user_alice".to_string(),
        actor_type: ActorType::User,
        name: Some("Alice Johnson".to_string()),
        ip_address: Some("192.168.1.100".to_string()),
        session_id: Some("sess_abc123".to_string()),
        attributes: HashMap::from([
            ("department".to_string(), "research".to_string()),
            ("role".to_string(), "researcher".to_string()),
        ]),
    };

    // Successful authentication
    trail.log_authentication(user_actor.clone(), true, "api_key")?;
    println!("✓ Logged successful authentication for Alice");

    // Failed authentication attempt
    let attacker_actor = Actor {
        id: "unknown".to_string(),
        actor_type: ActorType::Unknown,
        name: None,
        ip_address: Some("10.0.0.5".to_string()),
        session_id: None,
        attributes: HashMap::new(),
    };

    trail.log_authentication(attacker_actor, false, "password")?;
    println!("✓ Logged failed authentication attempt\n");

    // Example 2: Log evaluation execution
    println!("Example 2: Evaluation Execution Logging");
    println!("----------------------------------------");

    let audio_resource = Resource {
        id: "audio_001".to_string(),
        resource_type: ResourceType::AudioFile,
        name: Some("sample_speech.wav".to_string()),
        parent: Some("dataset_english_001".to_string()),
        attributes: HashMap::from([
            ("duration".to_string(), "5.2s".to_string()),
            ("sample_rate".to_string(), "22050".to_string()),
            ("channels".to_string(), "1".to_string()),
        ]),
    };

    let eval_result = ActionResult {
        success: true,
        status_code: 200,
        message: Some("Quality evaluation completed successfully".to_string()),
        error_details: None,
    };

    let eval_context = HashMap::from([
        (
            "metrics".to_string(),
            serde_json::json!(["PESQ", "STOI", "MCD"]),
        ),
        ("quality_score".to_string(), serde_json::json!(4.5)),
        ("processing_node".to_string(), serde_json::json!("node-001")),
    ]);

    trail.log_evaluation(
        user_actor.clone(),
        audio_resource.clone(),
        eval_result,
        150, // 150ms duration
        eval_context,
    )?;
    println!("✓ Logged evaluation execution with metrics\n");

    // Example 3: Log data access
    println!("Example 3: Data Access Logging");
    println!("-------------------------------");

    let dataset_resource = Resource {
        id: "dataset_001".to_string(),
        resource_type: ResourceType::Dataset,
        name: Some("English Speech Dataset".to_string()),
        parent: None,
        attributes: HashMap::from([
            ("size".to_string(), "10GB".to_string()),
            ("samples".to_string(), "5000".to_string()),
        ]),
    };

    trail.log_data_access(user_actor.clone(), dataset_resource, "read", true)?;
    println!("✓ Logged data access for dataset\n");

    // Example 4: Log security events
    println!("Example 4: Security Event Logging");
    println!("----------------------------------");

    let security_details = HashMap::from([
        (
            "event_type".to_string(),
            serde_json::json!("rate_limit_exceeded"),
        ),
        ("request_count".to_string(), serde_json::json!(105)),
        ("threshold".to_string(), serde_json::json!(100)),
        ("time_window".to_string(), serde_json::json!("1 minute")),
    ]);

    trail.log_security_event(
        user_actor.clone(),
        "Rate limit exceeded",
        SeverityLevel::Warning,
        security_details,
    )?;
    println!("✓ Logged security event (rate limit exceeded)\n");

    // Log a critical security event
    let critical_details = HashMap::from([
        (
            "event_type".to_string(),
            serde_json::json!("unauthorized_access_attempt"),
        ),
        (
            "target_resource".to_string(),
            serde_json::json!("admin_panel"),
        ),
        ("source_ip".to_string(), serde_json::json!("10.0.0.5")),
    ]);

    let suspicious_actor = Actor {
        id: "user_suspicious".to_string(),
        actor_type: ActorType::User,
        name: None,
        ip_address: Some("10.0.0.5".to_string()),
        session_id: Some("sess_xyz789".to_string()),
        attributes: HashMap::new(),
    };

    trail.log_security_event(
        suspicious_actor,
        "Unauthorized access attempt to admin panel",
        SeverityLevel::Critical,
        critical_details,
    )?;
    println!("✓ Logged critical security event\n");

    // Example 5: View statistics
    println!("Example 5: Audit Statistics");
    println!("----------------------------");

    let stats = trail.get_statistics();
    println!("Total events logged: {}", stats.total_events);
    println!("\nEvents by type:");
    for (event_type, count) in &stats.events_by_type {
        println!("  - {}: {}", event_type, count);
    }

    println!("\nEvents by severity:");
    for (severity, count) in &stats.events_by_severity {
        println!("  - {}: {}", severity, count);
    }

    if let Some(last_time) = stats.last_event_time {
        let elapsed = SystemTime::now()
            .duration_since(last_time)
            .unwrap_or(Duration::from_secs(0));
        println!("\nLast event: {:.2}s ago", elapsed.as_secs_f64());
    }
    println!();

    // Example 6: Query events
    println!("Example 6: Query Audit Events");
    println!("------------------------------");

    let query = AuditQuery {
        start_time: Some(SystemTime::now() - Duration::from_secs(3600)),
        end_time: Some(SystemTime::now()),
        event_types: Some(vec![AuditEventType::Security]),
        severity_levels: None,
        actors: None,
        limit: Some(10),
    };

    let security_events = trail.query_events(query)?;
    println!(
        "Found {} security events in the last hour:",
        security_events.len()
    );
    for event in security_events.iter().take(3) {
        println!(
            "  - [{}] {} ({})",
            match event.severity {
                SeverityLevel::Critical => "CRITICAL",
                SeverityLevel::Error => "ERROR",
                SeverityLevel::Warning => "WARNING",
                SeverityLevel::Info => "INFO",
                SeverityLevel::Debug => "DEBUG",
            },
            event.action,
            event.timestamp_iso
        );
    }
    println!();

    // Example 7: Generate compliance report
    println!("Example 7: Compliance Report");
    println!("----------------------------");

    let start_time = SystemTime::now() - Duration::from_secs(3600);
    let end_time = SystemTime::now();

    let report = trail.generate_compliance_report(start_time, end_time)?;

    println!("Compliance Report (Last Hour):");
    println!("  Total events: {}", report.total_events);
    println!("  Security events: {}", report.security_events);
    println!(
        "  Failed authentications: {}",
        report.failed_authentications
    );
    println!("  Data access events: {}", report.data_access_events);

    if !report.top_actors.is_empty() {
        println!("\n  Top active users:");
        for (actor_id, count) in report.top_actors.iter().take(5) {
            println!("    - {}: {} events", actor_id, count);
        }
    }

    if !report.anomalies.is_empty() {
        println!("\n  ⚠ Anomalies detected:");
        for anomaly in &report.anomalies {
            println!("    - {}", anomaly);
        }
    } else {
        println!("\n  ✓ No anomalies detected");
    }
    println!();

    // Example 8: Advanced logging with custom context
    println!("Example 8: Advanced Logging with Context");
    println!("-----------------------------------------");

    let service_actor = Actor {
        id: "service_eval_worker_03".to_string(),
        actor_type: ActorType::Service,
        name: Some("Evaluation Worker #3".to_string()),
        ip_address: Some("172.16.0.23".to_string()),
        session_id: None,
        attributes: HashMap::from([
            ("cluster".to_string(), "production".to_string()),
            ("region".to_string(), "us-east-1".to_string()),
        ]),
    };

    let model_resource = Resource {
        id: "model_vits_v2".to_string(),
        resource_type: ResourceType::Model,
        name: Some("VITS v2 English".to_string()),
        parent: None,
        attributes: HashMap::from([
            ("architecture".to_string(), "VITS".to_string()),
            ("language".to_string(), "en-US".to_string()),
            ("version".to_string(), "2.0.1".to_string()),
        ]),
    };

    let batch_result = ActionResult {
        success: true,
        status_code: 200,
        message: Some("Batch evaluation completed".to_string()),
        error_details: None,
    };

    let batch_context = HashMap::from([
        ("batch_size".to_string(), serde_json::json!(100)),
        ("avg_quality_score".to_string(), serde_json::json!(4.3)),
        ("total_duration_ms".to_string(), serde_json::json!(15234)),
        ("successful".to_string(), serde_json::json!(98)),
        ("failed".to_string(), serde_json::json!(2)),
    ]);

    trail.log_evaluation(
        service_actor,
        model_resource,
        batch_result,
        15234,
        batch_context,
    )?;
    println!("✓ Logged batch evaluation with detailed context\n");

    // Flush all pending events
    trail.flush()?;
    println!("✓ All audit events flushed to disk");

    // Final statistics
    let final_stats = trail.get_statistics();
    println!("\n=== Final Statistics ===");
    println!("Total events logged: {}", final_stats.total_events);
    println!("Failed writes: {}", final_stats.failed_writes);
    println!("Buffer overflows: {}", final_stats.buffer_overflows);

    println!("\n✓ Audit trail demo completed successfully");
    println!("  Logs written to: {:?}", temp_dir);

    // Cleanup (optional - comment out to inspect logs)
    // std::fs::remove_dir_all(temp_dir)?;

    Ok(())
}

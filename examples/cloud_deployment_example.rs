/// Comprehensive Cloud Deployment Example for VoiRS
///
/// This example demonstrates enterprise-scale cloud deployment of VoiRS
/// across multiple cloud platforms including AWS, Azure, Google Cloud,
/// and Kubernetes orchestration for handling millions of synthesis requests.
///
/// Features Demonstrated:
/// - AWS deployment with ECS, Lambda, and S3 integration
/// - Azure deployment with Container Instances and Blob Storage
/// - Google Cloud deployment with Cloud Run and Cloud Storage
/// - Kubernetes orchestration with auto-scaling
/// - Load balancing and traffic distribution
/// - Multi-region deployment for global availability
/// - CDN integration for audio delivery
/// - Monitoring, logging, and observability
/// - Cost optimization and resource management
/// - Disaster recovery and high availability
///
/// This example's implementation is split across the sibling
/// `cloud_deployment_example/` module directory (see individual files for
/// details) to keep every source file under the project's 2000-line
/// refactor threshold:
/// - `config`: deployment/cloud-provider configuration types
/// - `service_types`: core service, request, and instance runtime types
/// - `service_core`: `CloudVoiRSService` lifecycle, scaling, and request handling
/// - `iac_templates`: Infrastructure-as-Code template generation (CloudFormation,
///   ARM, Deployment Manager, Kubernetes YAML, Terraform, Docker Compose)
/// - `infrastructure_support`: load balancer, storage manager, and auto-scaler
/// - `monitoring`: cloud monitoring/alerting
/// - `results`: deployment/processing result and status types
/// - `error`: `CloudError` and its trait implementations
/// - `scenarios`: example deployment scenario configurations
/// - `runner`: the main demonstration orchestration logic
// NOTE: this file is registered as the `path` of a `[[example]]` target, which
// makes it a crate root in its own right. Crate roots resolve bare `mod foo;`
// declarations directly inside their *containing* directory (`examples/foo.rs`),
// not inside a same-named sibling directory (`examples/cloud_deployment_example/foo.rs`)
// -- that convention only applies to non-root modules. `#[path = "..."]` is the
// standard, documented way to keep this file at its original top-level path while
// still storing its submodules under `cloud_deployment_example/`.
//
// The submodules are declared `pub mod` (rather than private `mod`) so that the
// `pub` types/fns they contain stay reachable from the crate root exactly as they
// were in the original single-file layout -- a private `mod` here would make
// rustc's dead-code reachability analysis treat everything inside as effectively
// private, spuriously flagging never-externally-read `pub` config fields as dead
// code (this file has none of its own state that's read outside the module tree,
// so nothing is actually gained by keeping the modules private).
#[path = "cloud_deployment_example/config.rs"]
pub mod config;
#[path = "cloud_deployment_example/error.rs"]
pub mod error;
#[path = "cloud_deployment_example/iac_templates.rs"]
pub mod iac_templates;
#[path = "cloud_deployment_example/infrastructure_support.rs"]
pub mod infrastructure_support;
#[path = "cloud_deployment_example/monitoring.rs"]
pub mod monitoring;
#[path = "cloud_deployment_example/results.rs"]
pub mod results;
#[path = "cloud_deployment_example/runner.rs"]
pub mod runner;
#[path = "cloud_deployment_example/scenarios.rs"]
pub mod scenarios;
#[path = "cloud_deployment_example/service_core.rs"]
pub mod service_core;
#[path = "cloud_deployment_example/service_types.rs"]
pub mod service_types;

#[cfg(test)]
#[path = "cloud_deployment_example/tests.rs"]
mod tests;

use error::CloudError;
use runner::run_cloud_deployment_example;

fn main() -> Result<(), CloudError> {
    run_cloud_deployment_example()
}

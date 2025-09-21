# VoiRS Recognizer CI/CD Infrastructure

This document describes the comprehensive CI/CD pipeline infrastructure for the VoiRS Recognizer project.

## 🚀 Quick Start

### Automated Setup

Run the setup script to configure the entire CI/CD infrastructure:

```bash
# Full setup (recommended)
./scripts/setup-ci.sh

# Development only
./scripts/setup-ci.sh --dev-only

# Validate existing setup
./scripts/setup-ci.sh --validate-only
```

### Manual Setup

1. **Copy environment configuration:**
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

2. **Start development environment:**
   ```bash
   docker-compose --profile development up
   ```

3. **Start production environment:**
   ```bash
   docker-compose --profile production up -d
   ```

## 📋 CI/CD Components

### GitHub Actions Workflows

#### 1. Main CI Pipeline (`.github/workflows/ci.yml`)
- **Triggers:** Push to main/develop, Pull requests
- **Features:**
  - Multi-platform testing (Ubuntu, Windows, macOS)
  - Multi-Rust version testing (stable, beta, nightly)
  - Feature-specific testing
  - Security auditing
  - Code coverage
  - WASM builds
  - C API builds
  - Docker builds
  - Documentation generation

#### 2. Release Pipeline (`.github/workflows/release.yml`)
- **Triggers:** Git tags (v*.*.*), Manual dispatch
- **Features:**
  - Automated GitHub releases
  - Multi-platform binary builds
  - Rust crate publishing
  - npm package publishing
  - Docker image publishing
  - Documentation deployment

#### 3. Performance Monitoring (`.github/workflows/performance.yml`)
- **Triggers:** Push to main, Pull requests, Daily schedule
- **Features:**
  - Performance regression testing
  - Comprehensive benchmarks
  - Memory profiling
  - Load testing
  - Performance comparison
  - Dashboard updates

### Docker Infrastructure

#### Multi-stage Dockerfile
- **Base:** System dependencies and Rust toolchain
- **Development:** Additional dev tools and hot-reload
- **Builder:** Optimized build environment
- **Production:** Minimal runtime image
- **REST API:** API server configuration
- **WASM Dev:** WASM development server

#### Docker Compose Profiles

```bash
# Core service only
docker-compose --profile core up

# API with dependencies
docker-compose --profile api up

# Full development environment
docker-compose --profile development up

# Complete production stack
docker-compose --profile production up

# Full stack with monitoring
docker-compose --profile full up
```

## ⚙️ Configuration

### Environment Variables

```env
# Application
VERSION=0.1.0-alpha.1
RUST_LOG=info

# Ports
API_PORT=8080
WASM_PORT=8081
REDIS_PORT=6379
POSTGRES_PORT=5432

# Security
MAX_UPLOAD_SIZE=100MB
CORS_ORIGIN=*

# Database
POSTGRES_DB=voirs
POSTGRES_USER=voirs
POSTGRES_PASSWORD=voirs123
```

### GitHub Repository Secrets

Required secrets for full CI/CD functionality:

| Secret | Purpose | Required |
|--------|---------|----------|
| `CARGO_REGISTRY_TOKEN` | Publishing to crates.io | ✅ |
| `NPM_TOKEN` | Publishing WASM to npm | ✅ |
| `DOCKERHUB_USERNAME` | Docker Hub publishing | ✅ |
| `DOCKERHUB_TOKEN` | Docker Hub publishing | ✅ |
| `CODECOV_TOKEN` | Code coverage reporting | ⚪ |

### Setting up secrets:

```bash
# Using GitHub CLI
gh secret set CARGO_REGISTRY_TOKEN
gh secret set NPM_TOKEN
gh secret set DOCKERHUB_USERNAME
gh secret set DOCKERHUB_TOKEN

# Or manually via GitHub web interface:
# Repository Settings > Secrets and variables > Actions
```

## 🔧 Service Configuration

### Redis (Caching)
- **File:** `config/redis.conf`
- **Purpose:** API response caching, session storage
- **Memory limit:** 256MB with LRU eviction

### PostgreSQL (Analytics)
- **File:** `scripts/init-db.sql`
- **Purpose:** Recognition session metadata, performance metrics
- **Tables:** `recognition_sessions`, `performance_metrics`, `recognition_results`

### Nginx (Reverse Proxy)
- **File:** `config/nginx.conf`
- **Routes:**
  - `/api/*` → API server
  - `/demo/*` → WASM demo
  - `/health` → Health checks

### Monitoring Stack

#### Prometheus (Metrics)
- **File:** `config/prometheus.yml`
- **Endpoints:** Application metrics, system metrics
- **Retention:** 200 hours

#### Grafana (Dashboards)
- **Port:** 3001
- **Default credentials:** admin/admin
- **Dashboards:** Auto-provisioned

#### Loki (Logs)
- **File:** `config/loki.yml`
- **Storage:** Local filesystem
- **Retention:** 7 days

## 🧪 Testing Strategy

### Test Matrix

| Platform | Rust Version | Features |
|----------|--------------|----------|
| Ubuntu | stable, beta, nightly | All combinations |
| Windows | stable, beta, nightly | Core features |
| macOS | stable, beta, nightly | Core features |

### Test Types

1. **Unit Tests:** `cargo test`
2. **Integration Tests:** `cargo test --test automated_regression_suite`
3. **Performance Tests:** `cargo test --test performance_tests`
4. **Documentation Tests:** `cargo test --doc`
5. **Benchmark Tests:** `cargo bench`

### Feature Testing

Each feature is tested independently:
- `default`, `whisper`, `whisper-pure`
- `deepspeech`, `wav2vec2`, `analysis`
- `python`, `wasm`, `c-api`, `rest-api`

### Performance Monitoring

- **Real-time Factor (RTF):** < 0.3
- **Memory Usage:** < 2GB
- **Startup Time:** < 5 seconds
- **Streaming Latency:** < 200ms

## 📦 Deployment Strategies

### Development Deployment

```bash
# Hot-reload development
docker-compose --profile development up

# Local testing
cargo watch -x "test --all-features"

# WASM development
./build-wasm.sh && npm run serve
```

### Staging Deployment

```bash
# API with monitoring
docker-compose --profile api --profile monitoring up -d

# Full stack without production optimizations
docker-compose --profile full up -d
```

### Production Deployment

```bash
# Production optimized stack
docker-compose --profile production up -d

# With complete monitoring
docker-compose --profile production --profile monitoring up -d

# Scale API instances
docker-compose up -d --scale voirs-api=3
```

### Kubernetes Deployment

The Docker images are compatible with Kubernetes deployment:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: voirs-recognizer-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: voirs-recognizer-api
  template:
    metadata:
      labels:
        app: voirs-recognizer-api
    spec:
      containers:
      - name: api
        image: ghcr.io/cool-japan/voirs-recognizer:latest
        ports:
        - containerPort: 8080
        env:
        - name: RUST_LOG
          value: "info"
```

## 🔍 Monitoring and Observability

### Health Checks

- **Application:** `/health` endpoint
- **Container:** Built-in health checks
- **Dependencies:** Redis, PostgreSQL connectivity

### Metrics Collection

- **Application metrics:** Performance, accuracy, usage
- **System metrics:** CPU, memory, disk, network
- **Business metrics:** Recognition sessions, model usage

### Log Aggregation

- **Application logs:** Structured JSON logging
- **Access logs:** Nginx request logs
- **System logs:** Container and host logs

### Alerting

Configure alerts in Grafana for:
- High error rates
- Performance degradation
- Resource exhaustion
- Service unavailability

## 🚨 Troubleshooting

### Common Issues

#### Build Failures
```bash
# Clean and rebuild
cargo clean
docker-compose down --volumes
docker-compose build --no-cache
```

#### Performance Issues
```bash
# Check system resources
docker stats

# Review performance logs
docker-compose logs voirs-api | grep Performance

# Run performance tests
cargo test --test performance_tests
```

#### Memory Issues
```bash
# Monitor memory usage
docker exec voirs-recognizer-api top -b -n 1

# Check for memory leaks
cargo test test_memory_pressure_scenarios
```

#### Network Issues
```bash
# Check service connectivity
docker-compose exec voirs-api curl http://redis:6379
docker-compose exec voirs-api pg_isready -h postgres

# Review network configuration
docker network ls
docker network inspect voirs-network
```

### Debugging Commands

```bash
# View all logs
docker-compose logs -f

# Access running container
docker-compose exec voirs-api bash

# Check service status
docker-compose ps

# View resource usage
docker-compose top
```

### Performance Debugging

```bash
# Run benchmarks
cargo bench

# Profile memory usage
cargo test --test performance_tests -- --nocapture

# Check regression tests
cargo test --test automated_regression_suite
```

## 📚 Additional Resources

### Documentation

- [API Documentation](https://cool-japan.github.io/voirs/docs/recognizer/)
- [Performance Dashboard](https://cool-japan.github.io/voirs/performance-dashboard/)
- [Architecture Overview](./README.md)

### Development Tools

- [Cargo Watch](https://github.com/watchexec/cargo-watch) - Hot reload
- [wasm-pack](https://rustwasm.github.io/wasm-pack/) - WASM builds
- [Docker Compose](https://docs.docker.com/compose/) - Service orchestration

### Monitoring Tools

- [Grafana](http://localhost:3001) - Dashboards
- [Prometheus](http://localhost:9090) - Metrics
- [Loki](http://localhost:3100) - Logs

## 🤝 Contributing

### Pre-commit Hooks

The setup script installs Git hooks that run:
1. Code formatting check (`cargo fmt`)
2. Linting (`cargo clippy`)
3. Tests (`cargo test`)

### CI/CD Pipeline Testing

Test your changes locally before pushing:

```bash
# Run the full test suite
./scripts/setup-ci.sh --validate-only

# Test Docker builds
docker-compose build

# Test all profiles
docker-compose --profile development config
docker-compose --profile production config
```

### Adding New Features

1. Add feature tests to the CI matrix
2. Update documentation
3. Add performance benchmarks if applicable
4. Test cross-platform compatibility

---

## 📞 Support

For issues related to CI/CD infrastructure:
1. Check the [troubleshooting section](#-troubleshooting)
2. Review GitHub Actions logs
3. Open an issue with detailed logs and environment information

**Happy building! 🚀**
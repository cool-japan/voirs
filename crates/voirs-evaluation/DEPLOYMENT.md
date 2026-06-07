# VoiRS Evaluation Deployment Guide

This guide covers deploying the VoiRS Evaluation distributed system on Kubernetes using either Kustomize or Helm.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Deployment Methods](#deployment-methods)
  - [Helm Deployment](#helm-deployment)
  - [Kustomize Deployment](#kustomize-deployment)
  - [kubectl Deployment](#kubectl-deployment)
- [Configuration](#configuration)
- [Monitoring](#monitoring)
- [Scaling](#scaling)
- [Troubleshooting](#troubleshooting)
- [Production Considerations](#production-considerations)

## Prerequisites

### Required Software

- Kubernetes 1.19+ cluster
- `kubectl` CLI tool
- `helm` 3.0+ (for Helm deployments)
- `kustomize` 4.0+ (optional, for Kustomize deployments)
- Docker (for building custom images)

### Cluster Requirements

- **Minimum Resources**:
  - 2 vCPUs
  - 4GB RAM
  - 20GB storage

- **Recommended for Production**:
  - 8+ vCPUs
  - 16GB+ RAM
  - 100GB+ storage
  - LoadBalancer support
  - PersistentVolume provisioner
  - Prometheus Operator (for monitoring)

### Optional Dependencies

- **Ingress Controller** (nginx, traefik, etc.) for external access
- **Cert-Manager** for automatic TLS certificate management
- **Prometheus Operator** for metrics collection
- **Network Policy** support for enhanced security

## Quick Start

The fastest way to get started is using Helm:

```bash
# Build and push Docker image (or use pre-built image)
docker build -t voirs/evaluation:v0.1.0 .
docker push voirs/evaluation:v0.1.0

# Install with Helm
helm install voirs-evaluation ./helm/voirs-evaluation

# Check deployment status
kubectl get pods -l app.kubernetes.io/name=voirs-evaluation
```

## Deployment Methods

### Helm Deployment

Helm is the recommended deployment method for production environments.

#### 1. Install Chart

```bash
# Development deployment
helm install voirs-evaluation ./helm/voirs-evaluation \
  --set worker.replicas=1 \
  --set autoscaling.enabled=false

# Production deployment
helm install voirs-evaluation ./helm/voirs-evaluation \
  --namespace voirs-prod --create-namespace \
  --set worker.replicas=5 \
  --set autoscaling.maxReplicas=50 \
  --set ingress.enabled=true \
  --set ingress.hosts[0].host=evaluation.example.com
```

#### 2. Upgrade Release

```bash
helm upgrade voirs-evaluation ./helm/voirs-evaluation \
  --namespace voirs-prod
```

#### 3. Custom Values

Create a custom `values.yaml`:

```yaml
# custom-values.yaml
worker:
  replicas: 10
  resources:
    requests:
      cpu: 2000m
      memory: 4Gi
    limits:
      cpu: 8000m
      memory: 16Gi

autoscaling:
  enabled: true
  minReplicas: 10
  maxReplicas: 100

ingress:
  enabled: true
  className: nginx
  hosts:
    - host: evaluation.prod.example.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: voirs-evaluation-tls
      hosts:
        - evaluation.prod.example.com

monitoring:
  enabled: true
  serviceMonitor:
    enabled: true
```

Deploy with custom values:

```bash
helm install voirs-evaluation ./helm/voirs-evaluation \
  --values custom-values.yaml
```

### Kustomize Deployment

Kustomize is ideal for GitOps workflows.

#### 1. Development Environment

```bash
kubectl apply -k k8s/overlays/development
```

#### 2. Production Environment

```bash
kubectl apply -k k8s/overlays/production
```

#### 3. Custom Overlays

Create a custom overlay:

```bash
mkdir -p k8s/overlays/staging
cat > k8s/overlays/staging/kustomization.yaml <<EOF
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

namespace: voirs-staging

bases:
- ../../base

patches:
- target:
    kind: Deployment
    name: voirs-evaluation-worker
  patch: |-
    - op: replace
      path: /spec/replicas
      value: 3

images:
- name: voirs/evaluation
  newTag: staging-latest
EOF

kubectl apply -k k8s/overlays/staging
```

### kubectl Deployment

For manual deployment using raw manifests:

```bash
# Apply base manifests
kubectl apply -f k8s/base/

# Or apply specific resources
kubectl apply -f k8s/base/rbac.yaml
kubectl apply -f k8s/base/configmap.yaml
kubectl apply -f k8s/base/pvc.yaml
kubectl apply -f k8s/base/deployment.yaml
kubectl apply -f k8s/base/service.yaml
kubectl apply -f k8s/base/hpa.yaml
kubectl apply -f k8s/base/networkpolicy.yaml
```

## Configuration

### Environment Variables

Key configuration options:

| Variable | Description | Default |
|----------|-------------|---------|
| `RUST_LOG` | Logging level | `info` |
| `COORDINATOR_URL` | Coordinator endpoint | `http://voirs-evaluation-coordinator:8080` |
| `MAX_CONCURRENT_TASKS` | Max tasks per worker | `10` |
| `TASK_TIMEOUT_SECONDS` | Task timeout | `300` |
| `HEARTBEAT_INTERVAL_SECONDS` | Heartbeat interval | `30` |

### ConfigMap

Edit the ConfigMap for application configuration:

```bash
kubectl edit configmap voirs-evaluation-config
```

Or update via Helm values:

```yaml
config:
  distributed:
    maxWorkers: 50
    loadBalancing: "adaptive"
  quality:
    defaultSampleRate: 16000
    enableGpu: true
```

### Secrets

Store sensitive data in Kubernetes Secrets:

```bash
kubectl create secret generic voirs-evaluation-secrets \
  --from-literal=api-key=your-api-key \
  --from-literal=storage-key=your-storage-key
```

## Monitoring

### Metrics Endpoint

Metrics are available at:

- **Coordinator**: `http://<coordinator-pod>:9090/metrics`
- **Workers**: `http://<worker-pod>:9090/metrics`

### Prometheus ServiceMonitor

If using Prometheus Operator, enable ServiceMonitor:

```yaml
monitoring:
  enabled: true
  serviceMonitor:
    enabled: true
    interval: 30s
```

### Key Metrics

Monitor these metrics:

- `voirs_evaluation_queue_depth` - Task queue depth
- `voirs_evaluation_active_tasks` - Active tasks count
- `voirs_evaluation_task_duration_seconds` - Task execution time
- `voirs_evaluation_errors_total` - Error counter
- `voirs_evaluation_worker_status` - Worker health status

### Grafana Dashboard

Import the provided Grafana dashboard:

```bash
kubectl apply -f monitoring/grafana-dashboard.json
```

## Scaling

### Manual Scaling

Scale workers manually:

```bash
kubectl scale deployment voirs-evaluation-worker --replicas=10
```

### Horizontal Pod Autoscaling

HPA automatically scales based on metrics:

```bash
# Check HPA status
kubectl get hpa voirs-evaluation-worker-hpa

# Modify HPA
kubectl edit hpa voirs-evaluation-worker-hpa
```

### Vertical Pod Autoscaling

Install VPA (optional):

```bash
kubectl apply -f https://github.com/kubernetes/autoscaler/releases/download/vertical-pod-autoscaler-0.13.0/vpa-v0.13.0-crd.yaml
kubectl apply -f monitoring/vpa.yaml
```

## Troubleshooting

### Check Pod Status

```bash
# List all pods
kubectl get pods -l app.kubernetes.io/name=voirs-evaluation

# Describe pod
kubectl describe pod <pod-name>

# View pod logs
kubectl logs <pod-name>

# Follow logs
kubectl logs -f <pod-name>
```

### Common Issues

#### 1. Workers Not Connecting to Coordinator

**Symptom**: Workers show connection errors in logs

**Solution**:
```bash
# Check service DNS resolution
kubectl exec -it <worker-pod> -- nslookup voirs-evaluation-coordinator

# Check network policy
kubectl get networkpolicy
kubectl describe networkpolicy voirs-evaluation-worker
```

#### 2. High Memory Usage

**Symptom**: OOMKilled pods

**Solution**:
```bash
# Increase memory limits
kubectl edit deployment voirs-evaluation-worker
# Or via Helm
helm upgrade voirs-evaluation ./helm/voirs-evaluation \
  --set worker.resources.limits.memory=8Gi
```

#### 3. Autoscaling Not Working

**Symptom**: HPA not scaling pods

**Solution**:
```bash
# Check HPA status
kubectl get hpa voirs-evaluation-worker-hpa
kubectl describe hpa voirs-evaluation-worker-hpa

# Verify metrics server
kubectl top pods
```

### Debug Mode

Enable debug logging:

```bash
kubectl set env deployment/voirs-evaluation-worker RUST_LOG=debug
kubectl set env deployment/voirs-evaluation-coordinator RUST_LOG=debug
```

## Production Considerations

### High Availability

- **Coordinator**: Use StatefulSet for coordinator with multiple replicas
- **Workers**: Deploy across multiple availability zones
- **Storage**: Use replicated storage backends (S3, GCS)

### Security

- Enable NetworkPolicy for traffic control
- Use Pod Security Policies/Pod Security Standards
- Rotate secrets regularly
- Enable audit logging
- Use private image registries

### Resource Management

- Set appropriate resource requests/limits
- Use ResourceQuotas for namespace isolation
- Monitor resource utilization
- Implement cost allocation tags

### Backup and Recovery

- Backup PersistentVolumes regularly
- Export configurations to version control
- Document disaster recovery procedures
- Test recovery procedures regularly

### Performance Optimization

- Enable GPU acceleration for compute-intensive workloads
- Use node affinity for workload placement
- Configure topology spread constraints
- Optimize network policies

### Cost Optimization

- Use spot/preemptible instances for workers
- Implement aggressive scale-down policies
- Enable cluster autoscaling
- Monitor and optimize resource usage

## Additional Resources

- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Helm Documentation](https://helm.sh/docs/)
- [Kustomize Documentation](https://kustomize.io/)
- [VoiRS Evaluation README](./README.md)
- [VoiRS GitHub](https://github.com/cool-japan/voirs)

## Support

For issues and questions:

- GitHub Issues: https://github.com/cool-japan/voirs/issues
- Documentation: https://github.com/cool-japan/voirs/tree/main/crates/voirs-evaluation

## License

Apache-2.0

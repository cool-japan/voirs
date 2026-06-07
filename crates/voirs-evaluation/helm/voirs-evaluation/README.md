# VoiRS Evaluation Helm Chart

This Helm chart deploys the VoiRS Evaluation distributed system on Kubernetes.

## TL;DR

```bash
helm repo add voirs https://charts.voirs.example.com
helm install voirs-evaluation voirs/voirs-evaluation
```

## Introduction

This chart bootstraps a VoiRS Evaluation deployment on a Kubernetes cluster using the Helm package manager. It deploys:

- Evaluation coordinator (manages task distribution)
- Evaluation workers (execute evaluation tasks)
- Horizontal Pod Autoscaler (auto-scaling based on load)
- ConfigMaps and Secrets (configuration management)
- Services (internal and external access)
- Network Policies (security)
- RBAC (role-based access control)

## Prerequisites

- Kubernetes 1.19+
- Helm 3.0+
- PV provisioner support in the underlying infrastructure (for persistence)
- (Optional) Prometheus Operator for monitoring

## Installing the Chart

To install the chart with the release name `voirs-evaluation`:

```bash
helm install voirs-evaluation ./helm/voirs-evaluation
```

The command deploys VoiRS Evaluation on the Kubernetes cluster with default configuration. The [Parameters](#parameters) section lists the parameters that can be configured during installation.

## Uninstalling the Chart

To uninstall/delete the `voirs-evaluation` deployment:

```bash
helm delete voirs-evaluation
```

This command removes all the Kubernetes components associated with the chart and deletes the release.

## Parameters

### Global Parameters

| Name                      | Description                         | Value |
|---------------------------|-------------------------------------|-------|
| `global.imageRegistry`    | Global Docker image registry        | `""`  |
| `global.imagePullSecrets` | Global Docker registry secret names | `[]`  |
| `global.storageClass`     | Global storage class                | `""`  |

### Image Parameters

| Name                | Description                          | Value                     |
|---------------------|--------------------------------------|---------------------------|
| `image.registry`    | Image registry                       | `docker.io`               |
| `image.repository`  | Image repository                     | `voirs/evaluation`        |
| `image.tag`         | Image tag                            | `v0.1.0`          |
| `image.pullPolicy`  | Image pull policy                    | `IfNotPresent`            |
| `image.pullSecrets` | Image pull secrets                   | `[]`                      |

### Coordinator Parameters

| Name                                | Description                           | Value          |
|-------------------------------------|---------------------------------------|----------------|
| `coordinator.enabled`               | Enable coordinator deployment         | `true`         |
| `coordinator.replicas`              | Number of coordinator replicas        | `1`            |
| `coordinator.resources.requests.cpu`| CPU request                           | `250m`         |
| `coordinator.resources.requests.memory` | Memory request                    | `512Mi`        |
| `coordinator.resources.limits.cpu`  | CPU limit                             | `1000m`        |
| `coordinator.resources.limits.memory` | Memory limit                        | `2Gi`          |
| `coordinator.persistence.enabled`   | Enable persistence                    | `true`         |
| `coordinator.persistence.size`      | PVC size                              | `10Gi`         |
| `coordinator.service.type`          | Service type                          | `ClusterIP`    |
| `coordinator.service.port`          | Service port                          | `8080`         |

### Worker Parameters

| Name                          | Description                    | Value      |
|-------------------------------|--------------------------------|------------|
| `worker.enabled`              | Enable worker deployment       | `true`     |
| `worker.replicas`             | Number of worker replicas      | `3`        |
| `worker.resources.requests.cpu` | CPU request                  | `500m`     |
| `worker.resources.requests.memory` | Memory request            | `1Gi`      |
| `worker.resources.limits.cpu` | CPU limit                      | `2000m`    |
| `worker.resources.limits.memory` | Memory limit                | `4Gi`      |
| `worker.cache.enabled`        | Enable worker cache            | `true`     |
| `worker.cache.sizeLimit`      | Cache size limit               | `5Gi`      |

### Autoscaling Parameters

| Name                                      | Description                         | Value   |
|-------------------------------------------|-------------------------------------|---------|
| `autoscaling.enabled`                     | Enable autoscaling                  | `true`  |
| `autoscaling.minReplicas`                 | Minimum replicas                    | `3`     |
| `autoscaling.maxReplicas`                 | Maximum replicas                    | `20`    |
| `autoscaling.targetCPUUtilizationPercentage` | Target CPU utilization           | `70`    |
| `autoscaling.targetMemoryUtilizationPercentage` | Target memory utilization     | `80`    |

### Service Parameters

| Name           | Description          | Value           |
|----------------|----------------------|-----------------|
| `service.type` | Kubernetes Service type | `LoadBalancer` |
| `service.port` | Service HTTP port    | `80`            |

### Ingress Parameters

| Name               | Description          | Value                     |
|--------------------|----------------------|---------------------------|
| `ingress.enabled`  | Enable ingress       | `false`                   |
| `ingress.className`| Ingress class name   | `nginx`                   |
| `ingress.hosts[0].host` | Hostname        | `evaluation.voirs.example.com` |

### RBAC Parameters

| Name                           | Description                  | Value  |
|--------------------------------|------------------------------|--------|
| `rbac.create`                  | Create RBAC resources        | `true` |
| `rbac.serviceAccount.create`   | Create service account       | `true` |
| `rbac.serviceAccount.name`     | Service account name         | `""`   |

### Monitoring Parameters

| Name                      | Description              | Value  |
|---------------------------|--------------------------|--------|
| `monitoring.enabled`      | Enable monitoring        | `true` |
| `monitoring.serviceMonitor.enabled` | Enable ServiceMonitor | `true` |
| `monitoring.prometheusRule.enabled` | Enable PrometheusRule | `true` |

### Configuration Parameters

See `values.yaml` for the complete list of configuration parameters.

## Configuration and Installation Details

### Resource Limits

The default resource limits are suitable for development and testing. For production deployments, adjust the resource limits based on your workload:

```yaml
worker:
  resources:
    requests:
      cpu: 2000m
      memory: 4Gi
    limits:
      cpu: 8000m
      memory: 16Gi
```

### Persistence

Coordinator state is stored in a PersistentVolumeClaim. To disable persistence:

```yaml
coordinator:
  persistence:
    enabled: false
```

### Autoscaling

The HorizontalPodAutoscaler automatically scales workers based on CPU and memory utilization. To adjust the scaling behavior:

```yaml
autoscaling:
  minReplicas: 5
  maxReplicas: 50
  targetCPUUtilizationPercentage: 60
```

### Monitoring

The chart includes ServiceMonitor and PrometheusRule resources for Prometheus Operator. Ensure Prometheus Operator is installed:

```bash
helm install prometheus prometheus-community/kube-prometheus-stack
```

### Security

The chart follows Kubernetes security best practices:

- Non-root user
- Read-only root filesystem
- Dropped capabilities
- NetworkPolicy for network isolation
- RBAC for service accounts

### Examples

#### Development Deployment

```bash
helm install voirs-evaluation ./helm/voirs-evaluation \
  --set worker.replicas=1 \
  --set autoscaling.enabled=false \
  --set coordinator.persistence.enabled=false
```

#### Production Deployment

```bash
helm install voirs-evaluation ./helm/voirs-evaluation \
  --set worker.replicas=10 \
  --set autoscaling.maxReplicas=100 \
  --set ingress.enabled=true \
  --set ingress.hosts[0].host=evaluation.prod.example.com \
  --set monitoring.enabled=true
```

#### With Custom Configuration

```bash
helm install voirs-evaluation ./helm/voirs-evaluation \
  --values my-values.yaml
```

## Upgrading

To upgrade the release:

```bash
helm upgrade voirs-evaluation ./helm/voirs-evaluation
```

## Troubleshooting

### Check Pod Status

```bash
kubectl get pods -l app.kubernetes.io/name=voirs-evaluation
```

### View Logs

```bash
# Coordinator logs
kubectl logs -l app.kubernetes.io/name=voirs-evaluation,component=coordinator

# Worker logs
kubectl logs -l app.kubernetes.io/name=voirs-evaluation,component=worker
```

### Port Forward for Local Access

```bash
kubectl port-forward svc/voirs-evaluation-coordinator 8080:8080
```

### Check Autoscaling Status

```bash
kubectl get hpa voirs-evaluation-worker-hpa
```

## License

Apache-2.0

## Support

For issues and questions:

- GitHub Issues: https://github.com/cool-japan/voirs/issues
- Documentation: https://github.com/cool-japan/voirs/tree/main/crates/voirs-evaluation

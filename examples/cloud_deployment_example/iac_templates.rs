use crate::results::InfrastructureCode;
use crate::service_types::CloudVoiRSService;

impl CloudVoiRSService {
    pub fn generate_infrastructure_code(&self) -> InfrastructureCode {
        InfrastructureCode {
            aws_cloudformation: self.generate_aws_template(),
            azure_arm: self.generate_azure_template(),
            gcp_deployment_manager: self.generate_gcp_template(),
            kubernetes_yaml: self.generate_k8s_yaml(),
            terraform: self.generate_terraform(),
            docker_compose: self.generate_docker_compose(),
        }
    }

    fn generate_aws_template(&self) -> String {
        r#"
AWSTemplateFormatVersion: '2010-09-09'
Description: 'VoiRS Cloud Deployment on AWS'

Parameters:
  MinInstances:
    Type: Number
    Default: 2
  MaxInstances:
    Type: Number
    Default: 20
  InstanceType:
    Type: String
    Default: c5.xlarge

Resources:
  VoiRSCluster:
    Type: AWS::ECS::Cluster
    Properties:
      ClusterName: voirs-cluster
      CapacityProviders:
        - EC2
        - FARGATE
        - FARGATE_SPOT

  VoiRSTaskDefinition:
    Type: AWS::ECS::TaskDefinition
    Properties:
      Family: voirs-task
      RequiresCompatibilities:
        - FARGATE
      NetworkMode: awsvpc
      Cpu: 1024
      Memory: 2048
      ExecutionRoleArn: !Ref VoiRSExecutionRole
      TaskRoleArn: !Ref VoiRSTaskRole
      ContainerDefinitions:
        - Name: voirs-container
          Image: voirs/cloud:latest
          Essential: true
          PortMappings:
            - ContainerPort: 8080
              Protocol: tcp
          Environment:
            - Name: RUST_LOG
              Value: info
            - Name: AUDIO_STORAGE_BUCKET
              Value: !Ref VoiRSAudioBucket
          LogConfiguration:
            LogDriver: awslogs
            Options:
              awslogs-group: /ecs/voirs
              awslogs-region: !Ref AWS::Region
              awslogs-stream-prefix: ecs

  VoiRSService:
    Type: AWS::ECS::Service
    DependsOn: VoiRSListener
    Properties:
      Cluster: !Ref VoiRSCluster
      TaskDefinition: !Ref VoiRSTaskDefinition
      DesiredCount: !Ref MinInstances
      LaunchType: FARGATE
      NetworkConfiguration:
        AwsvpcConfiguration:
          SecurityGroups:
            - !Ref VoiRSSecurityGroup
          Subnets:
            - !Ref PrivateSubnet1
            - !Ref PrivateSubnet2
          AssignPublicIp: DISABLED
      LoadBalancers:
        - ContainerName: voirs-container
          ContainerPort: 8080
          TargetGroupArn: !Ref VoiRSTargetGroup

  VoiRSLoadBalancer:
    Type: AWS::ElasticLoadBalancingV2::LoadBalancer
    Properties:
      Type: application
      Scheme: internet-facing
      SecurityGroups:
        - !Ref VoiRSALBSecurityGroup
      Subnets:
        - !Ref PublicSubnet1
        - !Ref PublicSubnet2

  VoiRSTargetGroup:
    Type: AWS::ElasticLoadBalancingV2::TargetGroup
    Properties:
      Port: 8080
      Protocol: HTTP
      VpcId: !Ref VPC
      TargetType: ip
      HealthCheckPath: /health
      HealthCheckIntervalSeconds: 30
      HealthyThresholdCount: 2
      UnhealthyThresholdCount: 3

  VoiRSListener:
    Type: AWS::ElasticLoadBalancingV2::Listener
    Properties:
      LoadBalancerArn: !Ref VoiRSLoadBalancer
      Port: 443
      Protocol: HTTPS
      Certificates:
        - CertificateArn: !Ref SSLCertificate
      DefaultActions:
        - Type: forward
          TargetGroupArn: !Ref VoiRSTargetGroup

  VoiRSAutoScalingTarget:
    Type: AWS::ApplicationAutoScaling::ScalableTarget
    Properties:
      MaxCapacity: !Ref MaxInstances
      MinCapacity: !Ref MinInstances
      ResourceId: !Sub service/${VoiRSCluster}/${VoiRSService.Name}
      RoleARN: !GetAtt ApplicationAutoScalingRole.Arn
      ScalableDimension: ecs:service:DesiredCount
      ServiceNamespace: ecs

  VoiRSScalingPolicy:
    Type: AWS::ApplicationAutoScaling::ScalingPolicy
    Properties:
      PolicyName: VoiRSCPUScaling
      PolicyType: TargetTrackingScaling
      ScalingTargetId: !Ref VoiRSAutoScalingTarget
      TargetTrackingScalingPolicyConfiguration:
        PredefinedMetricSpecification:
          PredefinedMetricType: ECSServiceAverageCPUUtilization
        TargetValue: 70.0

  VoiRSAudioBucket:
    Type: AWS::S3::Bucket
    Properties:
      BucketName: !Sub voirs-audio-${AWS::AccountId}
      VersioningConfiguration:
        Status: Enabled
      PublicAccessBlockConfiguration:
        BlockPublicAcls: true
        BlockPublicPolicy: true
        IgnorePublicAcls: true
        RestrictPublicBuckets: true
      BucketEncryption:
        ServerSideEncryptionConfiguration:
          - ServerSideEncryptionByDefault:
              SSEAlgorithm: AES256

Outputs:
  LoadBalancerDNS:
    Description: DNS name of the load balancer
    Value: !GetAtt VoiRSLoadBalancer.DNSName
  ServiceEndpoint:
    Description: HTTPS endpoint for the service
    Value: !Sub https://${VoiRSLoadBalancer.DNSName}
"#
        .to_string()
    }

    fn generate_azure_template(&self) -> String {
        r#"
{
  "$schema": "https://schema.management.azure.com/schemas/2019-04-01/deploymentTemplate.json#",
  "contentVersion": "1.0.0.0",
  "parameters": {
    "minInstances": {
      "type": "int",
      "defaultValue": 2,
      "metadata": {
        "description": "Minimum number of container instances"
      }
    },
    "maxInstances": {
      "type": "int",
      "defaultValue": 20,
      "metadata": {
        "description": "Maximum number of container instances"
      }
    }
  },
  "variables": {
    "containerGroupName": "voirs-container-group",
    "storageAccountName": "[concat('voirsaudio', uniqueString(resourceGroup().id))]",
    "containerRegistryName": "[concat('voirsacr', uniqueString(resourceGroup().id))]"
  },
  "resources": [
    {
      "type": "Microsoft.Storage/storageAccounts",
      "apiVersion": "2019-06-01",
      "name": "[variables('storageAccountName')]",
      "location": "[resourceGroup().location]",
      "sku": {
        "name": "Standard_LRS"
      },
      "kind": "StorageV2",
      "properties": {
        "accessTier": "Hot",
        "supportsHttpsTrafficOnly": true,
        "encryption": {
          "services": {
            "blob": {
              "enabled": true
            }
          },
          "keySource": "Microsoft.Storage"
        }
      }
    },
    {
      "type": "Microsoft.ContainerRegistry/registries",
      "apiVersion": "2019-05-01",
      "name": "[variables('containerRegistryName')]",
      "location": "[resourceGroup().location]",
      "sku": {
        "name": "Basic"
      },
      "properties": {
        "adminUserEnabled": true
      }
    },
    {
      "type": "Microsoft.ContainerInstance/containerGroups",
      "apiVersion": "2019-12-01",
      "name": "[variables('containerGroupName')]",
      "location": "[resourceGroup().location]",
      "dependsOn": [
        "[resourceId('Microsoft.Storage/storageAccounts', variables('storageAccountName'))]"
      ],
      "properties": {
        "containers": [
          {
            "name": "voirs-container",
            "properties": {
              "image": "voirs/cloud:latest",
              "resources": {
                "requests": {
                  "cpu": 1,
                  "memoryInGB": 2
                }
              },
              "ports": [
                {
                  "port": 8080,
                  "protocol": "TCP"
                }
              ],
              "environmentVariables": [
                {
                  "name": "RUST_LOG",
                  "value": "info"
                },
                {
                  "name": "AUDIO_STORAGE_ACCOUNT",
                  "value": "[variables('storageAccountName')]"
                }
              ]
            }
          }
        ],
        "osType": "Linux",
        "restartPolicy": "Always",
        "ipAddress": {
          "type": "Public",
          "ports": [
            {
              "port": 8080,
              "protocol": "TCP"
            }
          ]
        }
      }
    },
    {
      "type": "Microsoft.Insights/autoscalesettings",
      "apiVersion": "2015-04-01",
      "name": "voirs-autoscale",
      "location": "[resourceGroup().location]",
      "dependsOn": [
        "[resourceId('Microsoft.ContainerInstance/containerGroups', variables('containerGroupName'))]"
      ],
      "properties": {
        "profiles": [
          {
            "name": "DefaultProfile",
            "capacity": {
              "minimum": "[parameters('minInstances')]",
              "maximum": "[parameters('maxInstances')]",
              "default": "[parameters('minInstances')]"
            },
            "rules": [
              {
                "metricTrigger": {
                  "metricName": "Percentage CPU",
                  "metricResourceUri": "[resourceId('Microsoft.ContainerInstance/containerGroups', variables('containerGroupName'))]",
                  "timeGrain": "PT1M",
                  "statistic": "Average",
                  "timeWindow": "PT5M",
                  "timeAggregation": "Average",
                  "operator": "GreaterThan",
                  "threshold": 70
                },
                "scaleAction": {
                  "direction": "Increase",
                  "type": "ChangeCount",
                  "value": "1",
                  "cooldown": "PT5M"
                }
              }
            ]
          }
        ],
        "enabled": true,
        "targetResourceUri": "[resourceId('Microsoft.ContainerInstance/containerGroups', variables('containerGroupName'))]"
      }
    }
  ],
  "outputs": {
    "containerGroupFQDN": {
      "type": "string",
      "value": "[reference(resourceId('Microsoft.ContainerInstance/containerGroups', variables('containerGroupName'))).ipAddress.fqdn]"
    },
    "serviceEndpoint": {
      "type": "string",
      "value": "[concat('https://', reference(resourceId('Microsoft.ContainerInstance/containerGroups', variables('containerGroupName'))).ipAddress.fqdn, ':8080')]"
    }
  }
}
"#.to_string()
    }

    fn generate_gcp_template(&self) -> String {
        r#"
resources:
- name: voirs-cloud-run-service
  type: gcp-types/run-v1:namespaces.services
  properties:
    parent: namespaces/[PROJECT_ID]
    location: us-central1
    body:
      apiVersion: serving.knative.dev/v1
      kind: Service
      metadata:
        name: voirs-service
        annotations:
          run.googleapis.com/ingress: all
          autoscaling.knative.dev/minScale: "2"
          autoscaling.knative.dev/maxScale: "100"
      spec:
        template:
          metadata:
            annotations:
              autoscaling.knative.dev/maxScale: "20"
              run.googleapis.com/cpu-throttling: "false"
              run.googleapis.com/memory: "2Gi"
              run.googleapis.com/cpu: "1000m"
          spec:
            containers:
            - image: gcr.io/[PROJECT_ID]/voirs:latest
              ports:
              - containerPort: 8080
              env:
              - name: RUST_LOG
                value: info
              - name: GOOGLE_CLOUD_PROJECT
                value: [PROJECT_ID]
              - name: AUDIO_STORAGE_BUCKET
                value: $(ref.voirs-audio-bucket.name)
              resources:
                limits:
                  cpu: 1000m
                  memory: 2Gi

- name: voirs-audio-bucket
  type: storage.v1.bucket
  properties:
    name: voirs-audio-[PROJECT_NUMBER]
    location: US-CENTRAL1
    storageClass: STANDARD
    versioning:
      enabled: true
    encryption:
      defaultKmsKeyName: $(ref.voirs-kms-key.name)

- name: voirs-kms-key
  type: gcp-types/cloudkms-v1:projects.locations.keyRings.cryptoKeys
  properties:
    parent: projects/[PROJECT_ID]/locations/global/keyRings/voirs-keyring
    cryptoKeyId: voirs-audio-key
    purpose: ENCRYPT_DECRYPT
    versionTemplate:
      algorithm: GOOGLE_SYMMETRIC_ENCRYPTION

- name: voirs-load-balancer
  type: compute.v1.globalForwardingRule
  properties:
    name: voirs-lb-forwarding-rule
    target: $(ref.voirs-target-proxy.selfLink)
    portRange: 443-443
    IPProtocol: TCP

- name: voirs-target-proxy
  type: compute.v1.targetHttpsProxy
  properties:
    name: voirs-target-proxy
    urlMap: $(ref.voirs-url-map.selfLink)
    sslCertificates:
    - $(ref.voirs-ssl-cert.selfLink)

- name: voirs-ssl-cert
  type: compute.v1.sslCertificate
  properties:
    name: voirs-ssl-certificate
    managed:
      domains:
      - api.voirs.example.com

- name: voirs-url-map
  type: compute.v1.urlMap
  properties:
    name: voirs-url-map
    defaultService: $(ref.voirs-backend-service.selfLink)

- name: voirs-backend-service
  type: compute.v1.backendService
  properties:
    name: voirs-backend-service
    protocol: HTTP
    timeoutSec: 30
    connectionDraining:
      drainingTimeoutSec: 300
    healthChecks:
    - $(ref.voirs-health-check.selfLink)
    backends:
    - group: $(ref.voirs-cloud-run-service.status.url)
      balancingMode: UTILIZATION
      maxUtilization: 0.8

- name: voirs-health-check
  type: compute.v1.healthCheck
  properties:
    name: voirs-health-check
    type: HTTP
    httpHealthCheck:
      port: 8080
      requestPath: /health
      checkIntervalSec: 30
      timeoutSec: 5

outputs:
- name: serviceUrl
  value: $(ref.voirs-cloud-run-service.status.url)
- name: loadBalancerIP
  value: $(ref.voirs-load-balancer.IPAddress)
"#
        .to_string()
    }

    fn generate_k8s_yaml(&self) -> String {
        r#"
apiVersion: v1
kind: Namespace
metadata:
  name: voirs
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: voirs-deployment
  namespace: voirs
  labels:
    app: voirs
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 1
      maxSurge: 2
  selector:
    matchLabels:
      app: voirs
  template:
    metadata:
      labels:
        app: voirs
    spec:
      containers:
      - name: voirs
        image: voirs/cloud:latest
        imagePullPolicy: Always
        ports:
        - containerPort: 8080
          name: http
        env:
        - name: RUST_LOG
          value: "info"
        - name: KUBERNETES_NAMESPACE
          valueFrom:
            fieldRef:
              fieldPath: metadata.namespace
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 30
          timeoutSeconds: 5
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 10
          timeoutSeconds: 3
---
apiVersion: v1
kind: Service
metadata:
  name: voirs-service
  namespace: voirs
  labels:
    app: voirs
spec:
  selector:
    app: voirs
  ports:
  - name: http
    port: 80
    targetPort: 8080
  type: ClusterIP
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: voirs-ingress
  namespace: voirs
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/rate-limit-window: "1m"
spec:
  tls:
  - hosts:
    - api.voirs.example.com
    secretName: voirs-tls
  rules:
  - host: api.voirs.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: voirs-service
            port:
              number: 80
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: voirs-hpa
  namespace: voirs
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: voirs-deployment
  minReplicas: 2
  maxReplicas: 50
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: voirs-audio-storage
  namespace: voirs
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 100Gi
  storageClassName: fast-ssd
---
apiVersion: v1
kind: Secret
metadata:
  name: voirs-secrets
  namespace: voirs
type: Opaque
stringData:
  database-url: "postgresql://user:password@voirs-db:5432/voirs"
  s3-access-key: "AKIAIOSFODNN7EXAMPLE"
  s3-secret-key: "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: voirs-config
  namespace: voirs
data:
  config.yaml: |
    server:
      bind: "0.0.0.0:8080"
      workers: 4
    storage:
      type: "s3"
      bucket: "voirs-audio"
      region: "us-east-1"
    synthesis:
      default_voice: "neutral"
      max_text_length: 5000
      cache_ttl: 3600
"#
        .to_string()
    }

    fn generate_terraform(&self) -> String {
        r#"
terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "min_instances" {
  description = "Minimum number of instances"
  type        = number
  default     = 2
}

variable "max_instances" {
  description = "Maximum number of instances"
  type        = number
  default     = 20
}

variable "instance_type" {
  description = "EC2 instance type"
  type        = string
  default     = "c5.xlarge"
}

# VPC and Networking
resource "aws_vpc" "voirs_vpc" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name = "voirs-vpc"
  }
}

resource "aws_internet_gateway" "voirs_igw" {
  vpc_id = aws_vpc.voirs_vpc.id

  tags = {
    Name = "voirs-igw"
  }
}

resource "aws_subnet" "public_subnet" {
  count = 2

  vpc_id                  = aws_vpc.voirs_vpc.id
  cidr_block              = "10.0.${count.index + 1}.0/24"
  availability_zone       = data.aws_availability_zones.available.names[count.index]
  map_public_ip_on_launch = true

  tags = {
    Name = "voirs-public-subnet-${count.index + 1}"
  }
}

resource "aws_subnet" "private_subnet" {
  count = 2

  vpc_id            = aws_vpc.voirs_vpc.id
  cidr_block        = "10.0.${count.index + 10}.0/24"
  availability_zone = data.aws_availability_zones.available.names[count.index]

  tags = {
    Name = "voirs-private-subnet-${count.index + 1}"
  }
}

# ECS Cluster
resource "aws_ecs_cluster" "voirs_cluster" {
  name = "voirs-cluster"

  setting {
    name  = "containerInsights"
    value = "enabled"
  }

  tags = {
    Name = "voirs-cluster"
  }
}

# Load Balancer
resource "aws_lb" "voirs_alb" {
  name               = "voirs-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb_sg.id]
  subnets            = aws_subnet.public_subnet[*].id

  enable_deletion_protection = false

  tags = {
    Name = "voirs-alb"
  }
}

resource "aws_lb_target_group" "voirs_tg" {
  name        = "voirs-tg"
  port        = 8080
  protocol    = "HTTP"
  vpc_id      = aws_vpc.voirs_vpc.id
  target_type = "ip"

  health_check {
    enabled             = true
    healthy_threshold   = 2
    interval            = 30
    matcher             = "200"
    path                = "/health"
    port                = "traffic-port"
    protocol            = "HTTP"
    timeout             = 5
    unhealthy_threshold = 3
  }

  tags = {
    Name = "voirs-tg"
  }
}

resource "aws_lb_listener" "voirs_listener" {
  load_balancer_arn = aws_lb.voirs_alb.arn
  port              = "443"
  protocol          = "HTTPS"
  ssl_policy        = "ELBSecurityPolicy-TLS-1-2-2017-01"
  certificate_arn   = aws_acm_certificate.voirs_cert.arn

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.voirs_tg.arn
  }
}

# ECS Task Definition
resource "aws_ecs_task_definition" "voirs_task" {
  family                   = "voirs-task"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = 1024
  memory                   = 2048
  execution_role_arn       = aws_iam_role.ecs_execution_role.arn
  task_role_arn           = aws_iam_role.ecs_task_role.arn

  container_definitions = jsonencode([
    {
      name  = "voirs-container"
      image = "voirs/cloud:latest"
      
      portMappings = [
        {
          containerPort = 8080
          protocol      = "tcp"
        }
      ]

      environment = [
        {
          name  = "RUST_LOG"
          value = "info"
        },
        {
          name  = "AUDIO_STORAGE_BUCKET"
          value = aws_s3_bucket.voirs_audio.id
        }
      ]

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          awslogs-group         = aws_cloudwatch_log_group.voirs_logs.name
          awslogs-region        = var.aws_region
          awslogs-stream-prefix = "ecs"
        }
      }

      essential = true
    }
  ])

  tags = {
    Name = "voirs-task"
  }
}

# ECS Service
resource "aws_ecs_service" "voirs_service" {
  name            = "voirs-service"
  cluster         = aws_ecs_cluster.voirs_cluster.id
  task_definition = aws_ecs_task_definition.voirs_task.arn
  desired_count   = var.min_instances
  launch_type     = "FARGATE"

  network_configuration {
    security_groups  = [aws_security_group.ecs_sg.id]
    subnets          = aws_subnet.private_subnet[*].id
    assign_public_ip = false
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.voirs_tg.arn
    container_name   = "voirs-container"
    container_port   = 8080
  }

  depends_on = [aws_lb_listener.voirs_listener]

  tags = {
    Name = "voirs-service"
  }
}

# Auto Scaling
resource "aws_appautoscaling_target" "ecs_target" {
  max_capacity       = var.max_instances
  min_capacity       = var.min_instances
  resource_id        = "service/${aws_ecs_cluster.voirs_cluster.name}/${aws_ecs_service.voirs_service.name}"
  scalable_dimension = "ecs:service:DesiredCount"
  service_namespace  = "ecs"
}

resource "aws_appautoscaling_policy" "ecs_cpu_policy" {
  name               = "voirs-cpu-scaling"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.ecs_target.resource_id
  scalable_dimension = aws_appautoscaling_target.ecs_target.scalable_dimension
  service_namespace  = aws_appautoscaling_target.ecs_target.service_namespace

  target_tracking_scaling_policy_configuration {
    predefined_metric_specification {
      predefined_metric_type = "ECSServiceAverageCPUUtilization"
    }
    target_value = 70.0
  }
}

# S3 Bucket for Audio Storage
resource "aws_s3_bucket" "voirs_audio" {
  bucket = "voirs-audio-${random_id.bucket_suffix.hex}"

  tags = {
    Name = "voirs-audio-storage"
  }
}

resource "aws_s3_bucket_versioning" "voirs_audio_versioning" {
  bucket = aws_s3_bucket.voirs_audio.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_encryption" "voirs_audio_encryption" {
  bucket = aws_s3_bucket.voirs_audio.id

  server_side_encryption_configuration {
    rule {
      apply_server_side_encryption_by_default {
        sse_algorithm = "AES256"
      }
    }
  }
}

# Outputs
output "load_balancer_dns" {
  description = "DNS name of the load balancer"
  value       = aws_lb.voirs_alb.dns_name
}

output "service_endpoint" {
  description = "HTTPS endpoint for the service"
  value       = "https://${aws_lb.voirs_alb.dns_name}"
}

output "s3_bucket_name" {
  description = "Name of the S3 bucket for audio storage"
  value       = aws_s3_bucket.voirs_audio.id
}

data "aws_availability_zones" "available" {
  state = "available"
}

resource "random_id" "bucket_suffix" {
  byte_length = 4
}
"#.to_string()
    }

    fn generate_docker_compose(&self) -> String {
        r#"
version: '3.8'

services:
  voirs-api:
    image: voirs/cloud:latest
    ports:
      - "8080:8080"
    environment:
      - RUST_LOG=info
      - DATABASE_URL=postgresql://voirs:password@postgres:5432/voirs
      - REDIS_URL=redis://redis:6379
      - S3_ENDPOINT=http://minio:9000
      - S3_BUCKET=voirs-audio
      - S3_ACCESS_KEY=minioadmin
      - S3_SECRET_KEY=minioadmin
    depends_on:
      - postgres
      - redis
      - minio
    restart: unless-stopped
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '1.0'
          memory: 2G
        reservations:
          cpus: '0.5'
          memory: 1G
      restart_policy:
        condition: on-failure
        delay: 5s
        max_attempts: 3
        window: 120s
    networks:
      - voirs-network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - voirs-api
    restart: unless-stopped
    networks:
      - voirs-network

  postgres:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=voirs
      - POSTGRES_USER=voirs
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
    restart: unless-stopped
    networks:
      - voirs-network
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U voirs"]
      interval: 30s
      timeout: 10s
      retries: 5

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes --requirepass password
    volumes:
      - redis_data:/data
    restart: unless-stopped
    networks:
      - voirs-network
    healthcheck:
      test: ["CMD", "redis-cli", "--raw", "incr", "ping"]
      interval: 30s
      timeout: 10s
      retries: 5

  minio:
    image: minio/minio:latest
    ports:
      - "9000:9000"
      - "9001:9001"
    environment:
      - MINIO_ROOT_USER=minioadmin
      - MINIO_ROOT_PASSWORD=minioadmin
    volumes:
      - minio_data:/data
    command: server /data --console-address ":9001"
    restart: unless-stopped
    networks:
      - voirs-network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 30s
      timeout: 20s
      retries: 3

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--web.enable-lifecycle'
    restart: unless-stopped
    networks:
      - voirs-network

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./grafana/datasources:/etc/grafana/provisioning/datasources
    restart: unless-stopped
    networks:
      - voirs-network

  jaeger:
    image: jaegertracing/all-in-one:latest
    ports:
      - "16686:16686"
      - "14268:14268"
    environment:
      - COLLECTOR_ZIPKIN_HTTP_PORT=9411
    restart: unless-stopped
    networks:
      - voirs-network

volumes:
  postgres_data:
  redis_data:
  minio_data:
  prometheus_data:
  grafana_data:

networks:
  voirs-network:
    driver: bridge
"#
        .to_string()
    }
}

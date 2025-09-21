# VoiRS Docker Configuration

This directory contains Docker configurations for running VoiRS in various environments.

## Files Overview

- `Dockerfile` - Multi-stage production build
- `Dockerfile.dev` - Development environment with hot reloading
- `docker-compose.yml` - Main compose file with all services
- `docker-compose.prod.yml` - Production overrides
- `.dockerignore` - Files to exclude from Docker build context
- `nginx/nginx.conf` - Nginx reverse proxy configuration

## Quick Start

### Development Environment

```bash
# Start development environment with hot reloading
docker-compose --profile dev up voirs-dev

# Or build and run the main service
docker-compose up voirs
```

### Production Environment

```bash
# Start production environment
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

## Service Profiles

### Default Services
- `voirs` - Main VoiRS application

### Development Profile (`--profile dev`)
- `voirs-dev` - Development environment with hot reloading

### Cache Profile (`--profile cache`)
- `redis` - Redis cache for improved performance

### Database Profile (`--profile database`)
- `postgres` - PostgreSQL database for advanced features

### Proxy Profile (`--profile proxy`)
- `nginx` - Nginx reverse proxy with SSL termination

## Environment Variables

Create a `.env` file in the project root:

```bash
# Database configuration
POSTGRES_DB=voirs
POSTGRES_USER=voirs
POSTGRES_PASSWORD=your_secure_password

# Redis configuration
REDIS_PASSWORD=your_redis_password

# Application configuration
RUST_LOG=info
VOIRS_ENV=production
```

## SSL Configuration

For HTTPS support, place your SSL certificates in `docker/nginx/ssl/`:
- `server.crt` - SSL certificate
- `server.key` - Private key

Generate self-signed certificates for testing:
```bash
mkdir -p docker/nginx/ssl
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout docker/nginx/ssl/server.key \
  -out docker/nginx/ssl/server.crt
```

## Volume Mounts

The Docker configuration expects these directories:
- `./models` - TTS models (read-only)
- `./data` - Application data (read-write)
- `./config` - Configuration files (read-only)

## Common Commands

```bash
# Start all services
docker-compose up -d

# Start with specific profiles
docker-compose --profile dev --profile cache up -d

# View logs
docker-compose logs -f voirs

# Stop services
docker-compose down

# Rebuild images
docker-compose build --no-cache

# Production deployment
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Scale services
docker-compose up -d --scale voirs=3
```

## Health Checks

All services include health checks:
- VoiRS: `voirs --version`
- Redis: Built-in Redis ping
- PostgreSQL: Built-in PostgreSQL health check
- Nginx: HTTP status check

## Security Features

### Production Security
- Non-root user execution
- Read-only filesystem
- Resource limits
- Security headers (Nginx)
- Rate limiting
- SSL/TLS encryption

### Development Security
- Isolated development user
- Limited container privileges
- Network isolation

## Troubleshooting

### Common Issues

1. **Permission errors**: Ensure proper file ownership
   ```bash
   sudo chown -R $(id -u):$(id -g) ./data
   ```

2. **Port conflicts**: Check if ports are already in use
   ```bash
   netstat -tulpn | grep :8080
   ```

3. **Build failures**: Clear Docker cache
   ```bash
   docker system prune -a
   ```

4. **SSL issues**: Verify certificate files exist and are readable

### Debug Commands

```bash
# Check container status
docker-compose ps

# View detailed logs
docker-compose logs --tail=100 voirs

# Execute shell in container
docker-compose exec voirs /bin/bash

# Check resource usage
docker stats
```

## Performance Tuning

### Memory Optimization
- Adjust `deploy.resources.limits` in production compose
- Use `--memory` flag for specific containers

### CPU Optimization
- Set `deploy.resources.limits.cpus` for CPU limits
- Use `--cpus` flag for specific containers

### Network Optimization
- Use custom networks for service isolation
- Configure nginx worker processes based on CPU cores

## Monitoring

### Container Metrics
```bash
# View real-time metrics
docker stats

# Export metrics to file
docker stats --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}" > metrics.txt
```

### Application Logs
```bash
# Follow application logs
docker-compose logs -f voirs

# Export logs
docker-compose logs voirs > voirs.log
```

## Backup and Recovery

### Data Backup
```bash
# Backup PostgreSQL
docker-compose exec postgres pg_dump -U voirs voirs > backup.sql

# Backup Redis
docker-compose exec redis redis-cli save
docker cp $(docker-compose ps -q redis):/data/dump.rdb ./redis-backup.rdb
```

### Data Recovery
```bash
# Restore PostgreSQL
docker-compose exec -T postgres psql -U voirs voirs < backup.sql

# Restore Redis
docker cp ./redis-backup.rdb $(docker-compose ps -q redis):/data/dump.rdb
docker-compose restart redis
```
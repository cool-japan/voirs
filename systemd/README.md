# VoiRS Systemd Integration

This directory contains systemd service files and installation scripts for running VoiRS as a system service.

## Files Overview

- `voirs.service` - Main systemd service unit
- `voirs.socket` - Socket activation unit
- `voirs-maintenance.service` - Maintenance tasks service
- `voirs-maintenance.timer` - Timer for periodic maintenance
- `install.sh` - Installation and setup script
- `README.md` - This documentation

## Installation

### Prerequisites

- systemd-based Linux distribution
- Root access
- VoiRS binary built and available

### Quick Installation

```bash
# Build VoiRS first
cargo build --release

# Install as systemd service
sudo ./systemd/install.sh install
```

### Manual Installation

```bash
# Create user and directories
sudo useradd --system --group --home-dir /var/lib/voirs --no-create-home --shell /bin/false voirs
sudo mkdir -p /opt/voirs /etc/voirs /var/lib/voirs /var/log/voirs /run/voirs

# Install binary
sudo cp target/release/voirs /opt/voirs/voirs
sudo chown root:root /opt/voirs/voirs
sudo chmod 755 /opt/voirs/voirs

# Install service files
sudo cp systemd/*.service /etc/systemd/system/
sudo cp systemd/*.socket /etc/systemd/system/
sudo cp systemd/*.timer /etc/systemd/system/

# Set permissions
sudo chown -R voirs:voirs /var/lib/voirs /var/log/voirs /run/voirs
sudo chmod 755 /opt/voirs /etc/voirs

# Reload systemd and enable services
sudo systemctl daemon-reload
sudo systemctl enable voirs.service voirs.socket voirs-maintenance.timer
```

## Service Management

### Basic Commands

```bash
# Start service
sudo systemctl start voirs

# Stop service
sudo systemctl stop voirs

# Restart service
sudo systemctl restart voirs

# Check status
sudo systemctl status voirs

# View logs
sudo journalctl -u voirs

# Follow logs
sudo journalctl -u voirs -f
```

### Socket Activation

VoiRS supports socket activation for better resource management:

```bash
# Start socket (service starts on demand)
sudo systemctl start voirs.socket

# Check socket status
sudo systemctl status voirs.socket

# Test socket activation
curl http://localhost:8080/health
```

### Maintenance Tasks

Periodic maintenance is handled by a systemd timer:

```bash
# Check timer status
sudo systemctl status voirs-maintenance.timer

# View maintenance logs
sudo journalctl -u voirs-maintenance

# Run maintenance manually
sudo systemctl start voirs-maintenance
```

## Configuration

### Main Configuration

Configuration file: `/etc/voirs/config.toml`

```toml
[server]
host = "0.0.0.0"
port = 8080
workers = 4

[logging]
level = "info"
path = "/var/log/voirs"

[models]
path = "/var/lib/voirs/models"
cache_size = "1GB"

[audio]
output_format = "wav"
sample_rate = 22050

[performance]
max_concurrent_requests = 100
request_timeout = 30
```

### Environment Variables

Environment file: `/etc/voirs/environment`

```bash
RUST_LOG=info
VOIRS_CONFIG_PATH=/etc/voirs
VOIRS_DATA_PATH=/var/lib/voirs
VOIRS_LOG_PATH=/var/log/voirs
VOIRS_MODELS_PATH=/var/lib/voirs/models
```

## Directory Structure

```
/opt/voirs/              # Installation directory
├── voirs               # Main binary

/etc/voirs/             # Configuration
├── config.toml         # Main configuration
└── environment         # Environment variables

/var/lib/voirs/         # Data directory
├── models/            # TTS models
└── cache/             # Application cache

/var/log/voirs/         # Log directory
├── voirs.log          # Application logs
└── error.log          # Error logs

/run/voirs/             # Runtime directory
└── voirs.socket       # Unix socket (if used)
```

## Security Features

### User Isolation
- Runs as dedicated `voirs` user
- No shell access for service user
- Minimal system privileges

### Filesystem Security
- Read-only system directories
- Restricted write access
- Private temporary directories

### Network Security
- Configurable bind address
- Optional Unix socket support
- Capability-based permissions

### Resource Limits
- Memory limits (4GB default)
- CPU quotas (200% default)
- Process limits
- File descriptor limits

## Monitoring

### Service Status
```bash
# Check if service is running
sudo systemctl is-active voirs

# Check if service is enabled
sudo systemctl is-enabled voirs

# Get detailed status
sudo systemctl show voirs
```

### Performance Monitoring
```bash
# View resource usage
sudo systemctl status voirs

# Check memory usage
sudo systemctl show voirs --property=MemoryCurrent

# Check CPU usage
sudo systemctl show voirs --property=CPUUsageNSec
```

### Health Checks
```bash
# HTTP health check
curl http://localhost:8080/health

# Service health check
sudo systemctl is-active voirs

# Socket health check
sudo systemctl is-active voirs.socket
```

## Troubleshooting

### Common Issues

1. **Service fails to start**
   ```bash
   # Check logs
   sudo journalctl -u voirs --no-pager
   
   # Check configuration
   sudo -u voirs /opt/voirs/voirs --config /etc/voirs/config.toml --check
   ```

2. **Permission denied errors**
   ```bash
   # Fix ownership
   sudo chown -R voirs:voirs /var/lib/voirs /var/log/voirs
   
   # Fix permissions
   sudo chmod 750 /var/lib/voirs /var/log/voirs
   ```

3. **Port already in use**
   ```bash
   # Check what's using the port
   sudo netstat -tlnp | grep :8080
   
   # Change port in configuration
   sudo nano /etc/voirs/config.toml
   ```

4. **High memory usage**
   ```bash
   # Check memory settings
   sudo systemctl show voirs --property=MemoryMax
   
   # Adjust memory limit
   sudo systemctl edit voirs
   ```

### Debug Commands

```bash
# Run service in foreground
sudo -u voirs /opt/voirs/voirs server --config /etc/voirs/config.toml

# Check service dependencies
sudo systemctl list-dependencies voirs

# Verify service file syntax
sudo systemd-analyze verify /etc/systemd/system/voirs.service

# Show service environment
sudo systemctl show-environment
```

## Uninstallation

```bash
# Using installation script
sudo ./systemd/install.sh uninstall

# Manual removal
sudo systemctl stop voirs voirs.socket voirs-maintenance.timer
sudo systemctl disable voirs voirs.socket voirs-maintenance.timer
sudo rm /etc/systemd/system/voirs.*
sudo rm /etc/logrotate.d/voirs
sudo systemctl daemon-reload
```

## Integration with Other Services

### Nginx Proxy
```bash
# Install nginx
sudo apt install nginx

# Configure upstream
echo "upstream voirs_backend { server localhost:8080; }" | sudo tee /etc/nginx/conf.d/voirs.conf

# Add to nginx config
location /voirs/ {
    proxy_pass http://voirs_backend/;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
}
```

### Firewall Configuration
```bash
# UFW
sudo ufw allow 8080/tcp

# firewalld
sudo firewall-cmd --permanent --add-port=8080/tcp
sudo firewall-cmd --reload

# iptables
sudo iptables -A INPUT -p tcp --dport 8080 -j ACCEPT
```

### Log Aggregation
```bash
# Rsyslog
echo "if $programname == 'voirs' then /var/log/voirs/voirs.log" | sudo tee /etc/rsyslog.d/voirs.conf

# Logrotate is automatically configured by the installer
```
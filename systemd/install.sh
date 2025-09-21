#!/bin/bash

# VoiRS Systemd Service Installation Script
# This script installs and configures VoiRS as a systemd service

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SERVICE_NAME="voirs"
BINARY_NAME="voirs"
INSTALL_DIR="/opt/voirs"
CONFIG_DIR="/etc/voirs"
DATA_DIR="/var/lib/voirs"
LOG_DIR="/var/log/voirs"
RUN_DIR="/run/voirs"
USER="voirs"
GROUP="voirs"

# Print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}=== $1 ===${NC}"
}

# Check if running as root
check_root() {
    if [[ $EUID -ne 0 ]]; then
        print_error "This script must be run as root"
        exit 1
    fi
}

# Check if systemd is available
check_systemd() {
    if ! command -v systemctl > /dev/null 2>&1; then
        print_error "systemctl not found. This system does not appear to use systemd."
        exit 1
    fi
}

# Create voirs user and group
create_user() {
    print_header "Creating VoiRS user and group"
    
    if ! getent group $GROUP > /dev/null 2>&1; then
        print_status "Creating group: $GROUP"
        groupadd --system $GROUP
    else
        print_status "Group $GROUP already exists"
    fi
    
    if ! getent passwd $USER > /dev/null 2>&1; then
        print_status "Creating user: $USER"
        useradd --system --gid $GROUP --home-dir $DATA_DIR --no-create-home --shell /bin/false $USER
    else
        print_status "User $USER already exists"
    fi
}

# Create directories
create_directories() {
    print_header "Creating directories"
    
    directories=("$INSTALL_DIR" "$CONFIG_DIR" "$DATA_DIR" "$LOG_DIR" "$RUN_DIR" "$DATA_DIR/models")
    
    for dir in "${directories[@]}"; do
        if [[ ! -d "$dir" ]]; then
            print_status "Creating directory: $dir"
            mkdir -p "$dir"
        else
            print_status "Directory already exists: $dir"
        fi
    done
    
    # Set permissions
    chown -R $USER:$GROUP $DATA_DIR $LOG_DIR $RUN_DIR
    chmod 755 $INSTALL_DIR $CONFIG_DIR
    chmod 750 $DATA_DIR $LOG_DIR $RUN_DIR
}

# Install binary
install_binary() {
    print_header "Installing VoiRS binary"
    
    local binary_path=""
    
    # Try to find the binary
    if [[ -f "./target/release/$BINARY_NAME" ]]; then
        binary_path="./target/release/$BINARY_NAME"
    elif [[ -f "./$BINARY_NAME" ]]; then
        binary_path="./$BINARY_NAME"
    elif command -v $BINARY_NAME > /dev/null 2>&1; then
        binary_path=$(which $BINARY_NAME)
    else
        print_error "VoiRS binary not found. Please build the project first."
        exit 1
    fi
    
    print_status "Installing binary from: $binary_path"
    cp "$binary_path" "$INSTALL_DIR/$BINARY_NAME"
    chown root:root "$INSTALL_DIR/$BINARY_NAME"
    chmod 755 "$INSTALL_DIR/$BINARY_NAME"
}

# Install systemd service files
install_service_files() {
    print_header "Installing systemd service files"
    
    local service_files=("$SERVICE_NAME.service" "$SERVICE_NAME.socket" "$SERVICE_NAME-maintenance.service" "$SERVICE_NAME-maintenance.timer")
    
    for file in "${service_files[@]}"; do
        if [[ -f "./systemd/$file" ]]; then
            print_status "Installing: $file"
            cp "./systemd/$file" "/etc/systemd/system/"
            chmod 644 "/etc/systemd/system/$file"
        else
            print_warning "Service file not found: ./systemd/$file"
        fi
    done
}

# Create configuration files
create_config() {
    print_header "Creating configuration files"
    
    # Main configuration file
    if [[ ! -f "$CONFIG_DIR/config.toml" ]]; then
        print_status "Creating configuration file: $CONFIG_DIR/config.toml"
        cat > "$CONFIG_DIR/config.toml" << EOF
[server]
host = "0.0.0.0"
port = 8080
workers = 4

[logging]
level = "info"
path = "$LOG_DIR"

[models]
path = "$DATA_DIR/models"
cache_size = "1GB"

[audio]
output_format = "wav"
sample_rate = 22050

[performance]
max_concurrent_requests = 100
request_timeout = 30
EOF
    else
        print_status "Configuration file already exists: $CONFIG_DIR/config.toml"
    fi
    
    # Environment file
    if [[ ! -f "$CONFIG_DIR/environment" ]]; then
        print_status "Creating environment file: $CONFIG_DIR/environment"
        cat > "$CONFIG_DIR/environment" << EOF
RUST_LOG=info
VOIRS_CONFIG_PATH=$CONFIG_DIR
VOIRS_DATA_PATH=$DATA_DIR
VOIRS_LOG_PATH=$LOG_DIR
VOIRS_MODELS_PATH=$DATA_DIR/models
EOF
    else
        print_status "Environment file already exists: $CONFIG_DIR/environment"
    fi
    
    # Set permissions
    chown -R root:$GROUP $CONFIG_DIR
    chmod 640 "$CONFIG_DIR/config.toml" "$CONFIG_DIR/environment"
}

# Create logrotate configuration
create_logrotate() {
    print_header "Creating logrotate configuration"
    
    cat > "/etc/logrotate.d/$SERVICE_NAME" << EOF
$LOG_DIR/*.log {
    daily
    missingok
    rotate 14
    compress
    delaycompress
    notifempty
    create 640 $USER $GROUP
    postrotate
        systemctl reload $SERVICE_NAME > /dev/null 2>&1 || true
    endscript
}
EOF
    
    print_status "Logrotate configuration created"
}

# Set up systemd services
setup_systemd() {
    print_header "Setting up systemd services"
    
    # Reload systemd
    print_status "Reloading systemd daemon"
    systemctl daemon-reload
    
    # Enable services
    print_status "Enabling services"
    systemctl enable $SERVICE_NAME.service
    systemctl enable $SERVICE_NAME.socket
    systemctl enable $SERVICE_NAME-maintenance.timer
    
    # Start socket (service will be started on demand)
    print_status "Starting socket"
    systemctl start $SERVICE_NAME.socket
    
    # Start maintenance timer
    print_status "Starting maintenance timer"
    systemctl start $SERVICE_NAME-maintenance.timer
    
    print_status "Services configured and started"
}

# Verify installation
verify_installation() {
    print_header "Verifying installation"
    
    # Check service status
    print_status "Checking service status"
    if systemctl is-enabled $SERVICE_NAME.service > /dev/null 2>&1; then
        print_status "Service is enabled"
    else
        print_error "Service is not enabled"
        return 1
    fi
    
    # Check socket status
    if systemctl is-active $SERVICE_NAME.socket > /dev/null 2>&1; then
        print_status "Socket is active"
    else
        print_error "Socket is not active"
        return 1
    fi
    
    # Check binary
    if [[ -x "$INSTALL_DIR/$BINARY_NAME" ]]; then
        print_status "Binary is executable"
    else
        print_error "Binary is not executable"
        return 1
    fi
    
    print_status "Installation verification completed successfully"
}

# Show usage information
show_usage() {
    print_header "VoiRS Service Management"
    cat << EOF

Service has been installed successfully!

Common commands:
  systemctl start $SERVICE_NAME          # Start the service
  systemctl stop $SERVICE_NAME           # Stop the service
  systemctl restart $SERVICE_NAME        # Restart the service
  systemctl status $SERVICE_NAME         # Check service status
  systemctl reload $SERVICE_NAME         # Reload configuration
  
  systemctl enable $SERVICE_NAME         # Enable auto-start
  systemctl disable $SERVICE_NAME        # Disable auto-start
  
  journalctl -u $SERVICE_NAME            # View logs
  journalctl -u $SERVICE_NAME -f         # Follow logs
  
Configuration files:
  $CONFIG_DIR/config.toml                # Main configuration
  $CONFIG_DIR/environment                # Environment variables
  
Data directories:
  $DATA_DIR                              # Application data
  $LOG_DIR                               # Log files
  $DATA_DIR/models                       # TTS models
  
Service files:
  /etc/systemd/system/$SERVICE_NAME.service
  /etc/systemd/system/$SERVICE_NAME.socket
  /etc/systemd/system/$SERVICE_NAME-maintenance.service
  /etc/systemd/system/$SERVICE_NAME-maintenance.timer

EOF
}

# Main installation function
main() {
    print_header "VoiRS Systemd Service Installation"
    
    check_root
    check_systemd
    
    case "${1:-install}" in
        install)
            create_user
            create_directories
            install_binary
            install_service_files
            create_config
            create_logrotate
            setup_systemd
            verify_installation
            show_usage
            ;;
        uninstall)
            print_header "Uninstalling VoiRS service"
            systemctl stop $SERVICE_NAME.service $SERVICE_NAME.socket $SERVICE_NAME-maintenance.timer || true
            systemctl disable $SERVICE_NAME.service $SERVICE_NAME.socket $SERVICE_NAME-maintenance.timer || true
            rm -f /etc/systemd/system/$SERVICE_NAME.service
            rm -f /etc/systemd/system/$SERVICE_NAME.socket
            rm -f /etc/systemd/system/$SERVICE_NAME-maintenance.service
            rm -f /etc/systemd/system/$SERVICE_NAME-maintenance.timer
            rm -f /etc/logrotate.d/$SERVICE_NAME
            systemctl daemon-reload
            print_status "Service uninstalled"
            ;;
        *)
            echo "Usage: $0 [install|uninstall]"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
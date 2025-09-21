# Linux Integration Guide

VoiRS provides comprehensive Linux integration through PulseAudio, ALSA, D-Bus, SystemD, and Linux-specific performance optimizations including real-time scheduling and NUMA awareness.

## Table of Contents

- [PulseAudio Integration](#pulseaudio-integration)
- [ALSA Support](#alsa-support)
- [D-Bus Integration](#d-bus-integration)
- [SystemD Integration](#systemd-integration)
- [Real-time Scheduling](#real-time-scheduling)
- [NUMA Optimization](#numa-optimization)
- [Build Configuration](#build-configuration)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

## PulseAudio Integration

PulseAudio is the standard audio server on most modern Linux distributions, providing high-level audio management.

### Basic PulseAudio Setup

```c
#include "voirs/platform/linux.h"

// Initialize PulseAudio connection
VoirsLinuxPulseAudio* pulse = voirs_linux_init_pulseaudio();
if (!pulse) {
    fprintf(stderr, "Failed to initialize PulseAudio\n");
    return -1;
}

// Check if PulseAudio is running
if (!voirs_linux_pulseaudio_is_running(pulse)) {
    printf("PulseAudio is not running, falling back to ALSA\n");
    voirs_linux_destroy_pulseaudio(pulse);
    return -1;
}

// Cleanup when done
voirs_linux_destroy_pulseaudio(pulse);
```

### Audio Device Management

```c
VoirsLinuxPulseAudio* pulse = voirs_linux_init_pulseaudio();

// Get PulseAudio server information
VoirsPulseServerInfo* server_info = voirs_linux_get_pulse_server_info(pulse);
if (server_info) {
    printf("PulseAudio Server: %s\n", server_info->server_name);
    printf("Version: %s\n", server_info->version);
    printf("Sample Rate: %u Hz\n", server_info->sample_rate);
    printf("Channels: %u\n", server_info->channels);
}

// Enumerate audio devices
VoirsLinuxAudioDevice* devices;
int device_count = voirs_linux_get_audio_devices(pulse, &devices);

for (int i = 0; i < device_count; i++) {
    printf("Device %d: %s\n", devices[i].id, devices[i].name);
    printf("  Driver: %s\n", devices[i].driver);
    printf("  Card: %s\n", devices[i].card_name);
    printf("  Sample Rate: %u Hz\n", devices[i].sample_rate);
    printf("  Channels: %u\n", devices[i].channels);
    printf("  Type: %s\n", devices[i].is_input ? "Input" : "Output");
    printf("  Default: %s\n", devices[i].is_default ? "Yes" : "No");
}

voirs_linux_free_device_list(devices, device_count);
```

### Volume Control

```c
// Get current volume for a device
float volume = voirs_linux_get_device_volume(pulse, 0);  // Device ID 0
printf("Current volume: %.0f%%\n", volume * 100);

// Set volume (0.0 to 1.0)
voirs_linux_set_volume(pulse, 0, 0.75f);  // 75% volume

// Mute/unmute device
voirs_linux_set_device_mute(pulse, 0, true);   // Mute
voirs_linux_set_device_mute(pulse, 0, false);  // Unmute

// Get mute status
bool is_muted = voirs_linux_get_device_mute(pulse, 0);
printf("Device is %s\n", is_muted ? "muted" : "unmuted");
```

### Stream Management

```c
// Create PulseAudio stream for synthesis
VoirsPulseAudioStream* stream = voirs_linux_create_pulse_stream(pulse);

// Configure stream
VoirsPulseStreamConfig config = {
    .sample_rate = 44100,
    .channels = 2,
    .sample_format = VOIRS_PULSE_FORMAT_FLOAT32LE,
    .buffer_target_length = 4096,
    .stream_name = "VoiRS Synthesis",
    .application_name = "VoiRS"
};

voirs_linux_configure_pulse_stream(stream, &config);

// Set stream callbacks
voirs_linux_set_stream_write_callback(stream, stream_write_callback, user_data);
voirs_linux_set_stream_state_callback(stream, stream_state_callback, user_data);

// Start streaming
voirs_linux_start_pulse_stream(stream);
```

## ALSA Support

Advanced Linux Sound Architecture (ALSA) provides low-level audio hardware access.

### ALSA Initialization

```c
// Initialize ALSA system
VoirsLinuxALSA* alsa = voirs_linux_init_alsa();
if (!alsa) {
    fprintf(stderr, "Failed to initialize ALSA\n");
    return -1;
}

// Check ALSA availability
if (!voirs_linux_alsa_is_available()) {
    printf("ALSA not available on this system\n");
    return -1;
}
```

### Sound Card Enumeration

```c
// Get ALSA sound cards
VoirsALSACard* cards;
int card_count = voirs_linux_get_alsa_cards(alsa, &cards);

for (int i = 0; i < card_count; i++) {
    printf("Card %d: %s\n", cards[i].id, cards[i].name);
    printf("  Driver: %s\n", cards[i].driver);
    
    // List devices on this card
    for (int j = 0; j < cards[i].device_count; j++) {
        VoirsALSADevice* device = &cards[i].devices[j];
        printf("  Device %d: %s (%s)\n", 
               device->id, device->name, device->device_type);
    }
}

voirs_linux_free_alsa_cards(cards, card_count);
```

### Device Capability Testing

```c
// Test device capabilities
uint32_t card_id = 0, device_id = 0;
VoirsALSADeviceCapability* caps = voirs_linux_test_alsa_device(alsa, card_id, device_id);

if (caps) {
    printf("Device %u:%u capabilities:\n", card_id, device_id);
    
    // Sample rates
    printf("Supported sample rates: ");
    for (int i = 0; i < caps->sample_rate_count; i++) {
        printf("%u ", caps->sample_rates[i]);
    }
    printf("\n");
    
    // Formats
    printf("Supported formats: ");
    for (int i = 0; i < caps->format_count; i++) {
        printf("%s ", caps->formats[i]);
    }
    printf("\n");
    
    // Channels
    printf("Supported channels: ");
    for (int i = 0; i < caps->channel_count; i++) {
        printf("%u ", caps->channels[i]);
    }
    printf("\n");
    
    // Buffer sizes
    printf("Supported buffer sizes: ");
    for (int i = 0; i < caps->buffer_size_count; i++) {
        printf("%u ", caps->buffer_sizes[i]);
    }
    printf("\n");
    
    voirs_linux_free_alsa_capabilities(caps);
}
```

### Direct ALSA PCM Access

```c
// Open ALSA PCM device for playback
VoirsALSAPCM* pcm = voirs_linux_open_alsa_pcm("default", VOIRS_ALSA_STREAM_PLAYBACK);

// Configure PCM parameters
VoirsALSAPCMConfig pcm_config = {
    .sample_rate = 44100,
    .channels = 2,
    .format = VOIRS_ALSA_FORMAT_S16_LE,
    .period_size = 1024,
    .buffer_size = 4096
};

voirs_linux_configure_alsa_pcm(pcm, &pcm_config);

// Write audio data
float* audio_data = /* your audio data */;
int frames_written = voirs_linux_write_alsa_pcm(pcm, audio_data, frame_count);

// Close PCM device
voirs_linux_close_alsa_pcm(pcm);
```

## D-Bus Integration

D-Bus provides inter-process communication and system service integration.

### Basic D-Bus Setup

```c
// Initialize D-Bus connection
VoirsLinuxDBus* dbus = voirs_linux_init_dbus();
if (!dbus) {
    fprintf(stderr, "Failed to initialize D-Bus\n");
    return -1;
}

// Check D-Bus availability
if (!voirs_linux_dbus_is_available()) {
    printf("D-Bus not available\n");
    return -1;
}
```

### Desktop Notifications

```c
// Send desktop notification
voirs_linux_send_notification(
    dbus,
    "VoiRS",                           // app_name
    "Synthesis Complete",              // title
    "Your audio file has been generated successfully."  // message
);

// Send notification with custom options
VoirsLinuxNotificationOptions options = {
    .app_name = "VoiRS",
    .title = "Processing Status",
    .body = "Speech synthesis in progress...",
    .icon = "audio-volume-high",
    .timeout = 5000,  // 5 seconds
    .urgency = VOIRS_NOTIFICATION_URGENCY_NORMAL,
    .category = "transfer.complete"
};

uint32_t notification_id = voirs_linux_send_notification_extended(dbus, &options);

// Update existing notification
options.body = "Speech synthesis completed!";
voirs_linux_update_notification(dbus, notification_id, &options);
```

### System Service Registration

```c
// Register VoiRS as a D-Bus service
const char* service_name = "com.voirs.SpeechSynthesis";
if (voirs_linux_register_service(dbus, service_name)) {
    printf("VoiRS service registered on D-Bus\n");
    
    // Export methods
    voirs_linux_export_method(dbus, "/com/voirs/SpeechSynthesis", 
                              "Synthesize", synthesize_method, NULL);
    voirs_linux_export_method(dbus, "/com/voirs/SpeechSynthesis", 
                              "GetVoices", get_voices_method, NULL);
}

// D-Bus method implementations
int synthesize_method(VoirsDBusMessage* message, void* user_data) {
    const char* text = voirs_dbus_get_string_arg(message, 0);
    const char* voice = voirs_dbus_get_string_arg(message, 1);
    
    // Perform synthesis
    VoirsAudioBuffer* audio = voirs_synthesize_with_voice(pipeline, text, voice);
    
    // Return result
    VoirsDBusMessage* reply = voirs_dbus_create_reply(message);
    voirs_dbus_append_string(reply, "synthesis_id_123");
    voirs_dbus_send_reply(dbus, reply);
    
    return 0;
}
```

### System Integration

```c
// Monitor system events
void dbus_signal_callback(VoirsDBusMessage* message, void* user_data) {
    const char* interface = voirs_dbus_get_interface(message);
    const char* member = voirs_dbus_get_member(message);
    
    if (strcmp(interface, "org.freedesktop.login1.Manager") == 0) {
        if (strcmp(member, "PrepareForSleep") == 0) {
            bool sleeping = voirs_dbus_get_boolean_arg(message, 0);
            if (sleeping) {
                printf("System going to sleep, pausing synthesis\n");
                voirs_pipeline_pause(pipeline);
            } else {
                printf("System waking up, resuming synthesis\n");
                voirs_pipeline_resume(pipeline);
            }
        }
    }
}

// Subscribe to system signals
voirs_linux_subscribe_signal(dbus, "org.freedesktop.login1", 
                             "org.freedesktop.login1.Manager", 
                             "PrepareForSleep", dbus_signal_callback, NULL);
```

## SystemD Integration

SystemD provides service management and system integration capabilities.

### Service Management

```c
// Check if SystemD is available
if (voirs_linux_is_systemd_available()) {
    printf("SystemD available\n");
    
    // Create VoiRS service
    VoirsSystemDServiceConfig service_config = {
        .service_name = "voirs-synthesis",
        .description = "VoiRS Speech Synthesis Service",
        .exec_start = "/usr/bin/voirs-daemon",
        .user = "voirs",
        .group = "audio",
        .restart = "always",
        .restart_sec = 5,
        .wanted_by = "multi-user.target"
    };
    
    voirs_linux_create_systemd_service(&service_config);
}

// Service control
voirs_linux_start_service("voirs-synthesis");
voirs_linux_stop_service("voirs-synthesis");
voirs_linux_restart_service("voirs-synthesis");

// Check service status
VoirsSystemDServiceStatus status;
if (voirs_linux_get_service_status("voirs-synthesis", &status)) {
    printf("Service status: %s\n", status.active_state);
    printf("Main PID: %d\n", status.main_pid);
    printf("Memory usage: %zu KB\n", status.memory_current / 1024);
}
```

### Journal Logging

```c
// Log to SystemD journal
voirs_linux_journal_log(VOIRS_LOG_INFO, "VoiRS synthesis started");
voirs_linux_journal_log(VOIRS_LOG_ERROR, "Failed to load voice model: %s", error_msg);

// Structured logging
VoirsJournalField fields[] = {
    {"MESSAGE", "Synthesis completed"},
    {"VOICE_MODEL", "neural_voice_v2"},
    {"DURATION_MS", "1234"},
    {"OUTPUT_FORMAT", "wav"},
    {NULL, NULL}
};

voirs_linux_journal_log_structured(VOIRS_LOG_INFO, fields);
```

## Real-time Scheduling

Linux real-time scheduling capabilities for low-latency audio processing.

### RT Scheduling Setup

```c
// Enable real-time scheduling
if (voirs_linux_enable_rt_scheduling()) {
    printf("Real-time scheduling enabled\n");
    
    // Set RT priority for audio threads
    voirs_linux_set_thread_priority(VOIRS_SCHED_FIFO, 80);
    
    // Set RT scheduling policy
    voirs_linux_set_scheduling_policy(VOIRS_SCHED_RR);  // Round-robin
} else {
    printf("Real-time scheduling not available\n");
    // Check if user has RT privileges
    if (!voirs_linux_check_rt_privileges()) {
        printf("Add user to 'audio' group for RT scheduling\n");
    }
}

// Configure RT limits
VoirsRTLimits rt_limits = {
    .rtprio = 95,           // RT priority limit
    .nice = -20,            // Nice level
    .memlock = 256 * 1024,  // Memory lock limit (KB)
    .rttime = -1            // Unlimited RT CPU time
};

voirs_linux_set_rt_limits(&rt_limits);
```

### Thread Affinity

```c
// Set CPU affinity for audio threads
cpu_set_t cpuset;
CPU_ZERO(&cpuset);
CPU_SET(2, &cpuset);  // Bind to CPU 2
CPU_SET(3, &cpuset);  // Bind to CPU 3

voirs_linux_set_thread_affinity(&cpuset);

// Isolate audio threads from system interrupts
voirs_linux_isolate_audio_cpus(2, 3);  // Isolate CPUs 2 and 3

// Set CPU governor for performance
voirs_linux_set_cpu_governor("performance");
```

### IRQ Optimization

```c
// Optimize interrupt handling for audio
voirs_linux_optimize_audio_irqs();

// Move IRQs away from audio processing CPUs
voirs_linux_move_irqs_away_from_cpus(2, 3);

// Set audio device IRQ affinity
voirs_linux_set_audio_irq_affinity(audio_device_irq, 1);  // CPU 1
```

## NUMA Optimization

Non-Uniform Memory Access optimization for multi-socket systems.

### NUMA Awareness

```c
// Check NUMA topology
int numa_nodes = voirs_linux_get_numa_node_count();
printf("NUMA nodes: %d\n", numa_nodes);

if (numa_nodes > 1) {
    // Get current NUMA node
    int current_node = voirs_linux_get_current_numa_node();
    printf("Current NUMA node: %d\n", current_node);
    
    // Allocate memory on specific NUMA node
    void* memory = voirs_linux_numa_alloc(1024 * 1024, current_node);
    
    // Bind threads to NUMA node
    voirs_linux_bind_to_numa_node(current_node);
    
    // Free NUMA memory
    voirs_linux_numa_free(memory, 1024 * 1024);
}
```

### NUMA-aware Memory Allocation

```c
// Create NUMA-aware memory pool
VoirsLinuxNumaPool* numa_pool = voirs_linux_create_numa_pool();

// Allocate on preferred node
void* buffer = voirs_linux_numa_pool_allocate(numa_pool, 4096);

// Allocate on specific node
void* buffer2 = voirs_linux_numa_pool_allocate_on_node(numa_pool, 4096, 1);

// Free allocations
voirs_linux_numa_pool_free(numa_pool, buffer, 4096);
voirs_linux_numa_pool_free(numa_pool, buffer2, 4096);

voirs_linux_destroy_numa_pool(numa_pool);
```

## Build Configuration

### Package Dependencies

```bash
# Ubuntu/Debian
sudo apt-get install \
    libpulse-dev \
    libasound2-dev \
    libdbus-1-dev \
    libsystemd-dev \
    libnuma-dev

# Fedora/RHEL
sudo dnf install \
    pulseaudio-libs-devel \
    alsa-lib-devel \
    dbus-devel \
    systemd-devel \
    numactl-devel

# Arch Linux
sudo pacman -S \
    libpulse \
    alsa-lib \
    dbus \
    systemd \
    numactl
```

### CMake Configuration

```cmake
# Linux-specific configuration
if(UNIX AND NOT APPLE)
    find_package(PkgConfig REQUIRED)
    
    # PulseAudio
    pkg_check_modules(PULSEAUDIO libpulse)
    if(PULSEAUDIO_FOUND)
        target_compile_definitions(your_app PRIVATE VOIRS_ENABLE_PULSEAUDIO=1)
        target_link_libraries(your_app ${PULSEAUDIO_LIBRARIES})
        target_include_directories(your_app PRIVATE ${PULSEAUDIO_INCLUDE_DIRS})
    endif()
    
    # ALSA
    pkg_check_modules(ALSA alsa)
    if(ALSA_FOUND)
        target_compile_definitions(your_app PRIVATE VOIRS_ENABLE_ALSA=1)
        target_link_libraries(your_app ${ALSA_LIBRARIES})
    endif()
    
    # D-Bus
    pkg_check_modules(DBUS dbus-1)
    if(DBUS_FOUND)
        target_compile_definitions(your_app PRIVATE VOIRS_ENABLE_DBUS=1)
        target_link_libraries(your_app ${DBUS_LIBRARIES})
    endif()
    
    # SystemD
    pkg_check_modules(SYSTEMD libsystemd)
    if(SYSTEMD_FOUND)
        target_compile_definitions(your_app PRIVATE VOIRS_ENABLE_SYSTEMD=1)
        target_link_libraries(your_app ${SYSTEMD_LIBRARIES})
    endif()
    
    # NUMA
    find_library(NUMA_LIBRARY numa)
    if(NUMA_LIBRARY)
        target_compile_definitions(your_app PRIVATE VOIRS_ENABLE_NUMA=1)
        target_link_libraries(your_app ${NUMA_LIBRARY})
    endif()
    
    # Threading
    find_package(Threads REQUIRED)
    target_link_libraries(your_app Threads::Threads)
    
    target_link_libraries(your_app voirs_ffi)
endif()
```

### Compiler Flags

```bash
# Compile with Linux features
gcc -o example example.c \
    -lvoirs_ffi \
    -lpulse \
    -lasound \
    -ldbus-1 \
    -lsystemd \
    -lnuma \
    -lpthread \
    -lrt
```

## Examples

### Complete Linux Integration Example

```c
#include "voirs/voirs_ffi.h"
#include "voirs/platform/linux.h"

int main() {
    // Initialize Linux-specific features
    if (!voirs_linux_initialize()) {
        fprintf(stderr, "Failed to initialize Linux features\n");
        return 1;
    }
    
    // Initialize PulseAudio
    VoirsLinuxPulseAudio* pulse = voirs_linux_init_pulseaudio();
    VoirsLinuxALSA* alsa = NULL;
    
    if (!pulse) {
        printf("PulseAudio not available, falling back to ALSA\n");
        alsa = voirs_linux_init_alsa();
        if (!alsa) {
            fprintf(stderr, "Neither PulseAudio nor ALSA available\n");
            return 1;
        }
    }
    
    // Initialize D-Bus for system integration
    VoirsLinuxDBus* dbus = voirs_linux_init_dbus();
    
    // Configure for optimal Linux performance
    VoirsAudioConfig config = {
        .backend = pulse ? "pulseaudio" : "alsa",
        .sample_rate = 44100,
        .buffer_size = 1024,
        .channels = 2
    };
    
    // Enable real-time scheduling if available
    if (voirs_linux_enable_rt_scheduling()) {
        printf("Real-time scheduling enabled\n");
        config.buffer_size = 256;  // Smaller buffer for lower latency
    }
    
    // Create pipeline with Linux optimizations
    VoirsPipeline* pipeline = voirs_create_pipeline_with_config(&config);
    
    // Enable NUMA optimization if available
    if (voirs_linux_get_numa_node_count() > 1) {
        voirs_linux_enable_numa_optimization(pipeline);
    }
    
    // Synthesize speech
    const char* text = "Hello from VoiRS on Linux!";
    VoirsAudioBuffer* audio = voirs_synthesize(pipeline, text);
    
    if (audio) {
        // Save audio file
        voirs_save_audio(audio, "output.wav", VOIRS_FORMAT_WAV);
        
        // Send desktop notification
        if (dbus) {
            voirs_linux_send_notification(dbus, "VoiRS", 
                                         "Synthesis Complete", 
                                         "Audio saved to output.wav");
        }
        
        voirs_destroy_audio_buffer(audio);
    }
    
    // Clean up
    voirs_destroy_pipeline(pipeline);
    if (pulse) voirs_linux_destroy_pulseaudio(pulse);
    if (alsa) voirs_linux_destroy_alsa(alsa);
    if (dbus) voirs_linux_destroy_dbus(dbus);
    voirs_linux_cleanup();
    
    return 0;
}
```

### Real-time Audio Processing with RT Scheduling

```c
#include <sched.h>
#include <sys/mman.h>

void setup_realtime_linux() {
    // Lock memory to prevent page faults
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
        perror("mlockall failed");
    }
    
    // Set real-time scheduling
    struct sched_param param;
    param.sched_priority = 80;
    
    if (sched_setscheduler(0, SCHED_FIFO, &param) != 0) {
        perror("sched_setscheduler failed");
        // Continue without RT scheduling
    }
    
    // Set CPU affinity
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(2, &cpuset);  // Use CPU 2 for audio processing
    
    if (pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset) != 0) {
        perror("pthread_setaffinity_np failed");
    }
    
    // Create real-time audio thread
    pthread_t audio_thread;
    pthread_attr_t attr;
    struct sched_param thread_param;
    
    pthread_attr_init(&attr);
    pthread_attr_setschedpolicy(&attr, SCHED_FIFO);
    thread_param.sched_priority = 85;  // Higher than main thread
    pthread_attr_setschedparam(&attr, &thread_param);
    
    pthread_create(&audio_thread, &attr, audio_thread_func, NULL);
    pthread_attr_destroy(&attr);
}

void* audio_thread_func(void* arg) {
    // Set thread name for debugging
    pthread_setname_np(pthread_self(), "voirs-audio");
    
    // Initialize audio processing
    VoirsLinuxALSA* alsa = voirs_linux_init_alsa();
    VoirsALSAPCM* pcm = voirs_linux_open_alsa_pcm("hw:0,0", VOIRS_ALSA_STREAM_PLAYBACK);
    
    // Configure for low latency
    VoirsALSAPCMConfig config = {
        .sample_rate = 48000,
        .channels = 2,
        .format = VOIRS_ALSA_FORMAT_S32_LE,
        .period_size = 64,   // Very small for low latency
        .buffer_size = 256,
        .periods = 4
    };
    
    voirs_linux_configure_alsa_pcm(pcm, &config);
    
    // Real-time audio processing loop
    float* buffer = malloc(config.period_size * config.channels * sizeof(float));
    
    while (running) {
        // Generate audio
        voirs_process_audio_realtime(pipeline, buffer, config.period_size);
        
        // Write to ALSA
        int frames_written = voirs_linux_write_alsa_pcm(pcm, buffer, config.period_size);
        if (frames_written != config.period_size) {
            // Handle underrun
            voirs_linux_recover_alsa_pcm(pcm);
        }
    }
    
    free(buffer);
    voirs_linux_close_alsa_pcm(pcm);
    voirs_linux_destroy_alsa(alsa);
    
    return NULL;
}
```

## Troubleshooting

### Common Issues

#### PulseAudio Connection Failed
```c
// Check PulseAudio status
if (!voirs_linux_pulseaudio_is_running(NULL)) {
    printf("PulseAudio not running\n");
    system("pulseaudio --check || pulseaudio --start");
}

// Check user permissions
if (!voirs_linux_check_audio_group()) {
    printf("User not in audio group\n");
    printf("Run: sudo usermod -a -G audio $USER\n");
}
```

#### ALSA Permission Issues
```c
// Check ALSA device permissions
if (!voirs_linux_check_alsa_permissions()) {
    printf("ALSA device permission denied\n");
    printf("Check /dev/snd/ permissions\n");
}

// List available ALSA devices
voirs_linux_list_alsa_devices();
```

#### Real-time Scheduling Failed
```c
// Check RT limits
VoirsRTLimits limits;
if (voirs_linux_get_rt_limits(&limits)) {
    printf("RT priority limit: %d\n", limits.rtprio);
    printf("Memory lock limit: %zu KB\n", limits.memlock);
} else {
    printf("No RT limits configured\n");
    printf("Configure /etc/security/limits.conf:\n");
    printf("@audio - rtprio 95\n");
    printf("@audio - memlock 256000\n");
}
```

#### D-Bus Service Registration Failed
```c
// Check D-Bus daemon
if (!voirs_linux_dbus_daemon_running()) {
    printf("D-Bus daemon not running\n");
    system("systemctl start dbus");
}

// Check service permissions
if (!voirs_linux_check_dbus_policy("com.voirs.SpeechSynthesis")) {
    printf("D-Bus policy not configured\n");
    // Install D-Bus policy file
}
```

### Debug Tools

```c
// Enable Linux-specific debugging
voirs_linux_enable_debug_mode();
voirs_linux_set_debug_level(VOIRS_DEBUG_VERBOSE);

// Audio system debugging
voirs_linux_debug_audio_system();
voirs_linux_log_audio_devices();
voirs_linux_check_audio_configuration();

// Performance monitoring
VoirsLinuxPerformanceMonitor* monitor = voirs_linux_create_performance_monitor();
voirs_linux_start_monitoring(monitor);

// ... perform operations ...

VoirsLinuxPerformanceReport report;
voirs_linux_get_performance_report(monitor, &report);
printf("CPU usage: %.2f%%\n", report.cpu_usage);
printf("Memory usage: %zu MB\n", report.memory_usage_mb);
printf("Audio xruns: %u\n", report.audio_xruns);
printf("RT violations: %u\n", report.rt_violations);

voirs_linux_destroy_performance_monitor(monitor);
```

### Performance Tuning

```c
// System-wide audio optimizations
voirs_linux_optimize_audio_system();

// CPU governor settings
voirs_linux_set_cpu_governor("performance");

// IRQ optimization
voirs_linux_optimize_audio_irqs();

// Memory optimization
voirs_linux_disable_swap_for_audio();
voirs_linux_set_vm_swappiness(1);

// Network optimization (if using network audio)
voirs_linux_optimize_network_for_audio();
```

## Security Considerations

- Real-time scheduling requires appropriate user permissions
- D-Bus service registration may require policy configuration
- SystemD service management requires appropriate privileges
- ALSA direct access may bypass PulseAudio security policies

## Version Compatibility

VoiRS Linux integration supports:
- Ubuntu 18.04 LTS and later
- Fedora 30 and later
- Debian 10 and later
- Arch Linux (rolling release)
- CentOS 8 and later
- Most distributions with PulseAudio 12.0+ or ALSA 1.1.8+

## Related Documentation

- [C API Reference](api_reference.md)
- [Memory Management](memory_management.md)
- [Threading Guide](threading.md)
- [Performance Optimization](performance.md)
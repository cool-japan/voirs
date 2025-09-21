# VoiRS Recognizer C/C++ Examples Makefile

# Compiler settings
CC = gcc
CXX = g++
CFLAGS = -Wall -Wextra -std=c11 -O2
CXXFLAGS = -Wall -Wextra -std=c++17 -O2
LDFLAGS = -L../../target/release
LIBS = -lvoirs_recognizer -lm

# Include paths
INCLUDES = -I../../include

# Source files
C_SOURCES = basic_recognition.c
CXX_SOURCES = streaming_recognition.cpp

# Output binaries
C_TARGETS = $(C_SOURCES:.c=)
CXX_TARGETS = $(CXX_SOURCES:.cpp=)
ALL_TARGETS = $(C_TARGETS) $(CXX_TARGETS)

# Default target
all: rust-lib $(ALL_TARGETS)

# Build the Rust library first
rust-lib:
	@echo "Building Rust library with C API..."
	cd ../.. && cargo build --release --features c-api
	@echo "Rust library built successfully!"

# C targets
$(C_TARGETS): %: %.c
	@echo "Compiling C example: $@"
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $< $(LDFLAGS) $(LIBS)

# C++ targets  
$(CXX_TARGETS): %: %.cpp
	@echo "Compiling C++ example: $@"
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $< $(LDFLAGS) $(LIBS)

# Clean targets
clean:
	@echo "Cleaning C/C++ examples..."
	rm -f $(ALL_TARGETS)

clean-all: clean
	@echo "Cleaning Rust library..."
	cd ../.. && cargo clean

# Install dependencies (for development)
install-deps:
	@echo "Installing development dependencies..."
	@which gcc > /dev/null || (echo "Error: GCC not found. Please install GCC." && exit 1)
	@which g++ > /dev/null || (echo "Error: G++ not found. Please install G++." && exit 1)
	@which cargo > /dev/null || (echo "Error: Cargo not found. Please install Rust." && exit 1)
	@echo "All dependencies are available!"

# Run examples
run-basic: basic_recognition
	@echo "Running basic recognition example..."
	LD_LIBRARY_PATH=../../target/release ./basic_recognition

run-streaming: streaming_recognition
	@echo "Running streaming recognition example..."
	LD_LIBRARY_PATH=../../target/release ./streaming_recognition

run-all: run-basic run-streaming

# Test with sample audio (if available)
test: basic_recognition
	@echo "Testing with sample audio..."
	@if [ -f "sample.wav" ]; then \
		LD_LIBRARY_PATH=../../target/release ./basic_recognition sample.wav; \
	else \
		echo "No sample.wav found, running with generated audio..."; \
		LD_LIBRARY_PATH=../../target/release ./basic_recognition; \
	fi

# Debug builds
debug: CFLAGS += -g -DDEBUG
debug: CXXFLAGS += -g -DDEBUG
debug: LDFLAGS = -L../../target/debug
debug: rust-lib-debug $(ALL_TARGETS)

rust-lib-debug:
	@echo "Building Rust library (debug) with C API..."
	cd ../.. && cargo build --features c-api

# Help target
help:
	@echo "VoiRS Recognizer C/C++ Examples"
	@echo "=============================="
	@echo ""
	@echo "Available targets:"
	@echo "  all          - Build all examples (default)"
	@echo "  rust-lib     - Build the Rust library with C API"
	@echo "  clean        - Clean example binaries"
	@echo "  clean-all    - Clean examples and Rust library"
	@echo "  install-deps - Check for required dependencies"
	@echo "  run-basic    - Run basic recognition example"
	@echo "  run-streaming - Run streaming recognition example"
	@echo "  run-all      - Run all examples"
	@echo "  test         - Test with sample audio file (sample.wav)"
	@echo "  debug        - Build debug versions"
	@echo "  help         - Show this help message"
	@echo ""
	@echo "Example usage:"
	@echo "  make                    # Build all examples"
	@echo "  make run-basic          # Run basic example"
	@echo "  make test               # Test with audio file"
	@echo "  make clean && make debug # Clean rebuild in debug mode"
	@echo ""
	@echo "Note: Examples require the VoiRS library to be built with 'c-api' feature."
	@echo "      Set LD_LIBRARY_PATH or DYLD_LIBRARY_PATH to find the library at runtime."

# Check library
check-lib:
	@echo "Checking for VoiRS library..."
	@if [ -f "../../target/release/libvoirs_recognizer.so" ] || [ -f "../../target/release/libvoirs_recognizer.dylib" ] || [ -f "../../target/release/libvoirs_recognizer.dll" ]; then \
		echo "✓ VoiRS library found"; \
	else \
		echo "✗ VoiRS library not found. Run 'make rust-lib' first."; \
		exit 1; \
	fi

# Phony targets
.PHONY: all rust-lib rust-lib-debug clean clean-all install-deps run-basic run-streaming run-all test debug help check-lib

# Default shell
SHELL := /bin/bash

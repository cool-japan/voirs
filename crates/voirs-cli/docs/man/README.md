# VoiRS CLI Manual Pages

This directory contains comprehensive manual pages for the VoiRS CLI tool. The manual pages provide detailed documentation for all commands, options, and configuration settings.

## Available Manual Pages

### Main Manual
- `voirs-cli.1` - Main manual page with overview and global options

### Command-Specific Manuals
- `voirs-cli-synthesize.1` - Text-to-speech synthesis command
- `voirs-cli-voices.1` - Voice management and listing
- `voirs-cli-models.1` - Acoustic and vocoder model management
- `voirs-cli-batch.1` - Batch processing with parallel synthesis
- `voirs-cli-interactive.1` - Interactive mode for real-time synthesis
- `voirs-cli-server.1` - HTTP server mode with REST API
- `voirs-cli-config.1` - Configuration management and profiles
- `voirs-cli-test.1` - Pipeline testing and validation
- `voirs-cli-guide.1` - Contextual help and tutorials
- `voirs-cli-completion.1` - Shell completion script generation

## Installation

### System-wide Installation
To install the manual pages system-wide:

```bash
# Copy manual pages to system directory
sudo cp *.1 /usr/local/share/man/man1/

# Update manual database
sudo mandb
```

### User Installation
To install manual pages for current user only:

```bash
# Create user manual directory
mkdir -p ~/.local/share/man/man1

# Copy manual pages
cp *.1 ~/.local/share/man/man1/

# Update MANPATH if needed
echo 'export MANPATH="$HOME/.local/share/man:$MANPATH"' >> ~/.bashrc
source ~/.bashrc

# Update manual database
mandb ~/.local/share/man
```

### Package Manager Installation
Manual pages are automatically installed when installing VoiRS CLI through package managers:

- **Homebrew**: `brew install voirs-cli`
- **Debian/Ubuntu**: `sudo apt install voirs-cli`
- **Chocolatey**: `choco install voirs-cli`
- **Scoop**: `scoop install voirs-cli`

## Usage

Once installed, you can access the manual pages using the `man` command:

```bash
# Main manual page
man voirs-cli

# Command-specific manual pages
man voirs-cli-synthesize
man voirs-cli-voices
man voirs-cli-models
man voirs-cli-batch
man voirs-cli-interactive
man voirs-cli-server
man voirs-cli-config
man voirs-cli-test
man voirs-cli-guide
man voirs-cli-completion
```

## Manual Page Format

All manual pages follow standard Unix manual page format (roff/troff) and include:

- **NAME** - Command name and brief description
- **SYNOPSIS** - Command syntax and options
- **DESCRIPTION** - Detailed description of functionality
- **OPTIONS** - Complete list of command-line options
- **EXAMPLES** - Practical usage examples
- **FILES** - Related configuration and data files
- **EXIT STATUS** - Exit code meanings
- **SEE ALSO** - Related commands and documentation

## Navigation

Within manual pages, you can use these navigation keys:

- **Space** - Next page
- **b** - Previous page
- **q** - Quit manual page
- **/** - Search forward
- **?** - Search backward
- **n** - Next search result
- **N** - Previous search result

## Integration with Help System

The manual pages complement the built-in help system:

```bash
# Built-in help (quick reference)
voirs-cli --help
voirs-cli synthesize --help

# Manual pages (comprehensive documentation)
man voirs-cli
man voirs-cli-synthesize
```

## Development

### Generating Manual Pages

Manual pages are written in roff format and can be generated from various sources:

```bash
# View manual page source
cat voirs-cli.1

# Preview manual page
man ./voirs-cli.1

# Convert to other formats
groff -man -Thtml voirs-cli.1 > voirs-cli.html
groff -man -Tpdf voirs-cli.1 > voirs-cli.pdf
```

### Validation

Manual pages can be validated using various tools:

```bash
# Check syntax
lexgrog voirs-cli.1

# Validate format
man --warnings=all ./voirs-cli.1

# Check for common issues
mandb-check voirs-cli.1
```

## Contributing

When contributing to manual pages:

1. Follow standard roff formatting
2. Include comprehensive examples
3. Keep descriptions clear and concise
4. Update related pages when adding new features
5. Test rendering with different terminal widths

## License

Manual pages are part of the VoiRS CLI project and are licensed under the MIT License.
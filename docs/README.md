# VoiRS Documentation

This directory contains comprehensive documentation for the VoiRS speech synthesis framework.

## Structure

```
docs/
├── api/           # API documentation and references
├── guide/         # User and developer guides
├── adr/           # Architecture Decision Records
└── rfc/           # Request for Comments (design proposals)
```

## Documentation Types

### API Documentation (`docs/api/`)
- Auto-generated from Rust doc comments
- Comprehensive function and struct references
- Code examples and usage patterns
- Integration guides for different languages

### User Guides (`docs/guide/`)
- Getting started tutorials
- Installation and setup instructions
- Configuration and deployment guides
- Troubleshooting and FAQ

### Architecture Decision Records (`docs/adr/`)
- Design decisions and rationale
- Technical trade-offs and alternatives
- Implementation history and evolution
- Performance considerations

### Request for Comments (`docs/rfc/`)
- Proposed features and changes
- Community discussion and feedback
- Design specifications and requirements
- Implementation timelines

## Building Documentation

### API Documentation
```bash
# Generate API docs
cargo doc --workspace --no-deps --all-features

# Open in browser
cargo doc --workspace --no-deps --all-features --open
```

### User Guide (mdBook)
```bash
# Install mdbook
cargo install mdbook

# Build and serve guide
cd docs/guide
mdbook serve --open
```

## Contributing

- All public APIs must have comprehensive documentation
- User guides should include practical examples
- ADRs should document significant architectural decisions
- RFCs should be used for major feature proposals

## Links

- [Online Documentation](https://docs.rs/voirs)
- [API Reference](https://docs.rs/voirs/latest/voirs/)
- [Community Discussions](https://github.com/cool-japan/voirs/discussions)
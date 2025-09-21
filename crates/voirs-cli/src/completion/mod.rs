//! Shell completion support for VoiRS CLI
//!
//! Provides shell completion scripts for bash, zsh, fish, and PowerShell.

use clap::CommandFactory;
use clap_complete::{generate, Generator, Shell};
use std::io::{self, Write};
use std::path::Path;

use crate::CliApp;

/// Generate shell completion script for the specified shell
pub fn generate_completion<G: Generator>(
    shell: G,
    mut output: impl Write,
) -> Result<(), io::Error> {
    let mut app = CliApp::command();
    generate(shell, &mut app, "voirs", &mut output);
    Ok(())
}

/// Generate completion script to a file
pub fn generate_completion_to_file<P: AsRef<Path>>(
    shell: Shell,
    output_path: P,
) -> Result<(), io::Error> {
    let file = std::fs::File::create(output_path)?;
    generate_completion(shell, file)
}

/// Generate completion script and write to stdout
pub fn generate_completion_to_stdout(shell: Shell) -> Result<(), io::Error> {
    let stdout = io::stdout();
    generate_completion(shell, stdout.lock())
}

/// Get installation instructions for the generated completion script
pub fn get_installation_instructions(shell: Shell) -> String {
    match shell {
        Shell::Bash => r#"# Installation instructions for Bash completion:

1. Save the completion script:
   voirs generate-completion bash > ~/.local/share/bash-completion/completions/voirs

2. Or for system-wide installation:
   sudo voirs generate-completion bash > /usr/share/bash-completion/completions/voirs

3. Restart your shell or source the completion:
   source ~/.local/share/bash-completion/completions/voirs

Note: Make sure bash-completion is installed on your system."#
            .to_string(),
        Shell::Zsh => r#"# Installation instructions for Zsh completion:

1. Save the completion script to a directory in your $fpath:
   voirs generate-completion zsh > ~/.local/share/zsh/site-functions/_voirs

2. Or add to your .zshrc:
   # Add this line to your .zshrc
   fpath=(~/.local/share/zsh/site-functions $fpath)
   autoload -Uz compinit && compinit

3. Restart your shell or reload completions:
   exec zsh

Alternative: Place in any directory in your $fpath and run 'compinit'."#
            .to_string(),
        Shell::Fish => r#"# Installation instructions for Fish completion:

1. Save the completion script:
   voirs generate-completion fish > ~/.config/fish/completions/voirs.fish

2. Restart fish or reload completions:
   exec fish

Fish will automatically load completions from the completions directory."#
            .to_string(),
        Shell::PowerShell => r#"# Installation instructions for PowerShell completion:

1. Check your PowerShell profile location:
   $PROFILE

2. Create the profile directory if it doesn't exist:
   New-Item -ItemType Directory -Path (Split-Path $PROFILE) -Force

3. Add the completion to your profile:
   voirs generate-completion powershell >> $PROFILE

4. Restart PowerShell or reload your profile:
   . $PROFILE

Note: You may need to adjust PowerShell execution policy to load the profile."#
            .to_string(),
        Shell::Elvish => r#"# Installation instructions for Elvish completion:

1. Save the completion script:
   voirs generate-completion elvish > ~/.elvish/lib/voirs-completion.elv

2. Add to your rc.elv:
   use ./voirs-completion

3. Restart Elvish:
   exec elvish"#
            .to_string(),
        _ => "Completion installation instructions not available for this shell.".to_string(),
    }
}

/// Generate a comprehensive completion installation script
pub fn generate_install_script() -> String {
    r#"#!/bin/bash
# VoiRS CLI Completion Installation Script
# Automatically installs shell completions for VoiRS CLI

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if voirs is available
check_voirs() {
    if ! command -v voirs &> /dev/null; then
        log_error "voirs command not found. Please install VoiRS CLI first."
        exit 1
    fi
    log_info "VoiRS CLI found: $(which voirs)"
}

# Install bash completion
install_bash() {
    log_info "Installing Bash completion..."
    
    # Check for bash-completion package
    if [ -d "/usr/share/bash-completion/completions" ]; then
        # System-wide installation
        sudo voirs generate-completion bash > /tmp/voirs_completion_bash
        sudo mv /tmp/voirs_completion_bash /usr/share/bash-completion/completions/voirs
        log_success "Bash completion installed system-wide"
    elif [ -d "$HOME/.local/share/bash-completion/completions" ]; then
        # User installation
        mkdir -p "$HOME/.local/share/bash-completion/completions"
        voirs generate-completion bash > "$HOME/.local/share/bash-completion/completions/voirs"
        log_success "Bash completion installed for user"
    else
        log_warn "Bash completion directory not found. Creating user directory..."
        mkdir -p "$HOME/.local/share/bash-completion/completions"
        voirs generate-completion bash > "$HOME/.local/share/bash-completion/completions/voirs"
        log_success "Bash completion installed for user"
    fi
}

# Install zsh completion
install_zsh() {
    log_info "Installing Zsh completion..."
    
    # Find zsh fpath directory
    local zsh_dir=""
    if [ -n "$ZSH" ] && [ -d "$ZSH/completions" ]; then
        zsh_dir="$ZSH/completions"
    elif [ -d "/usr/local/share/zsh/site-functions" ]; then
        zsh_dir="/usr/local/share/zsh/site-functions"
    elif [ -d "$HOME/.zsh/completions" ]; then
        zsh_dir="$HOME/.zsh/completions"
    else
        mkdir -p "$HOME/.zsh/completions"
        zsh_dir="$HOME/.zsh/completions"
    fi
    
    voirs generate-completion zsh > "$zsh_dir/_voirs"
    log_success "Zsh completion installed to $zsh_dir"
}

# Install fish completion
install_fish() {
    log_info "Installing Fish completion..."
    
    local fish_dir="$HOME/.config/fish/completions"
    mkdir -p "$fish_dir"
    voirs generate-completion fish > "$fish_dir/voirs.fish"
    log_success "Fish completion installed to $fish_dir"
}

# Install PowerShell completion
install_powershell() {
    log_info "Installing PowerShell completion..."
    
    if command -v pwsh &> /dev/null; then
        local ps_profile=$(pwsh -NoProfile -Command 'Write-Host $PROFILE')
        local ps_dir=$(dirname "$ps_profile")
        mkdir -p "$ps_dir"
        voirs generate-completion powershell > "$ps_dir/voirs_completion.ps1"
        log_success "PowerShell completion installed to $ps_dir"
        log_warn "You may need to update your PowerShell profile to source the completion"
    else
        log_warn "PowerShell not found. Skipping PowerShell completion."
    fi
}

# Install all completions
install_all() {
    log_info "Installing completions for all available shells..."
    
    command -v bash &> /dev/null && install_bash
    command -v zsh &> /dev/null && install_zsh
    command -v fish &> /dev/null && install_fish
    command -v pwsh &> /dev/null && install_powershell
    
    log_success "Installation complete!"
    log_info "You may need to restart your shell or source the completions."
}

# Show usage
show_usage() {
    echo "VoiRS CLI Completion Installation Script"
    echo ""
    echo "Usage: $0 [OPTION]"
    echo ""
    echo "Options:"
    echo "  bash        Install Bash completion"
    echo "  zsh         Install Zsh completion"
    echo "  fish        Install Fish completion"
    echo "  powershell  Install PowerShell completion"
    echo "  all         Install completions for all available shells"
    echo "  help        Show this help message"
    echo ""
    echo "If no option is provided, 'all' is assumed."
}

# Main function
main() {
    check_voirs
    
    case "${1:-all}" in
        bash)
            install_bash
            ;;
        zsh)
            install_zsh
            ;;
        fish)
            install_fish
            ;;
        powershell)
            install_powershell
            ;;
        all)
            install_all
            ;;
        help)
            show_usage
            ;;
        *)
            log_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
}

main "$@"
"#
    .to_string()
}

/// Display completion status and installation info
pub fn display_completion_status() -> String {
    let mut output = String::new();

    output.push_str("VoiRS CLI Shell Completion Status\n");
    output.push_str("==================================\n\n");

    let shells = [
        ("bash", "Bash"),
        ("zsh", "Zsh"),
        ("fish", "Fish"),
        ("pwsh", "PowerShell"),
    ];

    for (cmd, name) in shells.iter() {
        let status = if std::process::Command::new(cmd)
            .arg("--version")
            .output()
            .is_ok()
        {
            "✓ Available"
        } else {
            "✗ Not installed"
        };

        output.push_str(&format!("{:12} {}\n", name, status));
    }

    output.push_str("\nTo generate completion scripts:\n");
    output.push_str("  voirs generate-completion <shell> > completion_file\n\n");
    output.push_str("Supported shells: bash, zsh, fish, powershell, elvish\n\n");
    output.push_str("For installation instructions:\n");
    output.push_str("  voirs generate-completion <shell> --help\n");

    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_generate_bash_completion() {
        let mut output = Cursor::new(Vec::new());
        generate_completion(Shell::Bash, &mut output).unwrap();

        let result = String::from_utf8(output.into_inner()).unwrap();
        assert!(result.contains("voirs"));
        assert!(result.contains("_voirs"));
    }

    #[test]
    fn test_installation_instructions() {
        let bash_instructions = get_installation_instructions(Shell::Bash);
        assert!(bash_instructions.contains("bash-completion"));
        assert!(bash_instructions.contains("~/.local/share"));

        let zsh_instructions = get_installation_instructions(Shell::Zsh);
        assert!(zsh_instructions.contains("$fpath"));
        assert!(zsh_instructions.contains("_voirs"));
    }

    #[test]
    fn test_install_script_generation() {
        let script = generate_install_script();
        assert!(script.contains("#!/bin/bash"));
        assert!(script.contains("voirs generate-completion"));
        assert!(script.contains("install_bash"));
        assert!(script.contains("install_zsh"));
    }

    #[test]
    fn test_completion_status() {
        let status = display_completion_status();
        assert!(status.contains("VoiRS CLI Shell Completion Status"));
        assert!(status.contains("Bash"));
        assert!(status.contains("Zsh"));
        assert!(status.contains("Fish"));
    }
}

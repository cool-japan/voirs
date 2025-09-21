//! Git and version control integration for voirs-dataset
//!
//! This module provides Git integration with LFS support for large dataset files,
//! dataset versioning, change tracking, and collaborative workflows.

use crate::{DatasetError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tokio::process::Command as AsyncCommand;

/// Git configuration for dataset management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GitConfig {
    /// Repository URL
    pub repository_url: String,
    /// Local repository path
    pub local_path: PathBuf,
    /// Git LFS configuration
    pub lfs_config: LFSConfig,
    /// Branch configuration
    pub branch_config: BranchConfig,
    /// Commit configuration
    pub commit_config: CommitConfig,
    /// Remote configuration
    pub remote_config: RemoteConfig,
}

/// Git LFS configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LFSConfig {
    /// Enable Git LFS
    pub enabled: bool,
    /// File patterns to track with LFS
    pub track_patterns: Vec<String>,
    /// Maximum file size before using LFS (in bytes)
    pub max_file_size: u64,
    /// LFS storage endpoint
    pub storage_endpoint: Option<String>,
    /// LFS authentication token
    pub auth_token: Option<String>,
}

/// Branch configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BranchConfig {
    /// Main branch name
    pub main_branch: String,
    /// Development branch name
    pub dev_branch: String,
    /// Feature branch prefix
    pub feature_prefix: String,
    /// Dataset version branch prefix
    pub version_prefix: String,
}

/// Commit configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommitConfig {
    /// Automatic commit message template
    pub commit_template: String,
    /// Include dataset statistics in commit message
    pub include_stats: bool,
    /// Sign commits with GPG
    pub sign_commits: bool,
    /// Commit author name
    pub author_name: String,
    /// Commit author email
    pub author_email: String,
}

/// Remote configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemoteConfig {
    /// Remote name
    pub name: String,
    /// Remote URL
    pub url: String,
    /// Authentication method
    pub auth_method: AuthMethod,
    /// Push/pull strategy
    pub strategy: PushPullStrategy,
}

/// Git authentication methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuthMethod {
    /// SSH key authentication
    SSH {
        /// SSH private key path
        private_key_path: String,
        /// SSH public key path
        public_key_path: String,
        /// SSH key passphrase
        passphrase: Option<String>,
    },
    /// HTTPS token authentication
    Token {
        /// Personal access token
        token: String,
        /// Username (for GitHub/GitLab)
        username: String,
    },
    /// Username and password
    UsernamePassword {
        /// Username
        username: String,
        /// Password
        password: String,
    },
}

/// Push/pull strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PushPullStrategy {
    /// Force push/pull
    Force,
    /// Merge on conflicts
    Merge,
    /// Rebase on conflicts
    Rebase,
    /// Fail on conflicts
    Fail,
}

impl Default for GitConfig {
    fn default() -> Self {
        Self {
            repository_url: "".to_string(),
            local_path: PathBuf::from("./dataset"),
            lfs_config: LFSConfig::default(),
            branch_config: BranchConfig::default(),
            commit_config: CommitConfig::default(),
            remote_config: RemoteConfig::default(),
        }
    }
}

impl Default for LFSConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            track_patterns: vec![
                "*.wav".to_string(),
                "*.flac".to_string(),
                "*.mp3".to_string(),
                "*.ogg".to_string(),
                "*.parquet".to_string(),
                "*.bin".to_string(),
            ],
            max_file_size: 100 * 1024 * 1024, // 100MB
            storage_endpoint: None,
            auth_token: None,
        }
    }
}

impl Default for BranchConfig {
    fn default() -> Self {
        Self {
            main_branch: "main".to_string(),
            dev_branch: "develop".to_string(),
            feature_prefix: "feature/".to_string(),
            version_prefix: "version/".to_string(),
        }
    }
}

impl Default for CommitConfig {
    fn default() -> Self {
        Self {
            commit_template: "Dataset update: {message}".to_string(),
            include_stats: true,
            sign_commits: false,
            author_name: "Dataset Bot".to_string(),
            author_email: "dataset@example.com".to_string(),
        }
    }
}

impl Default for RemoteConfig {
    fn default() -> Self {
        Self {
            name: "origin".to_string(),
            url: "".to_string(),
            auth_method: AuthMethod::SSH {
                private_key_path: "~/.ssh/id_rsa".to_string(),
                public_key_path: "~/.ssh/id_rsa.pub".to_string(),
                passphrase: None,
            },
            strategy: PushPullStrategy::Merge,
        }
    }
}

/// Git repository interface
#[async_trait::async_trait]
pub trait GitRepository: Send + Sync {
    /// Initialize a new Git repository
    async fn init(&self, path: &Path) -> Result<()>;

    /// Clone an existing repository
    async fn clone(&self, url: &str, path: &Path) -> Result<()>;

    /// Add files to the staging area
    async fn add(&self, files: &[&str]) -> Result<()>;

    /// Commit changes
    async fn commit(&self, message: &str) -> Result<String>;

    /// Push changes to remote
    async fn push(&self, remote: &str, branch: &str) -> Result<()>;

    /// Pull changes from remote
    async fn pull(&self, remote: &str, branch: &str) -> Result<()>;

    /// Create a new branch
    async fn create_branch(&self, branch_name: &str) -> Result<()>;

    /// Switch to a branch
    async fn checkout(&self, branch_name: &str) -> Result<()>;

    /// Get current branch
    async fn current_branch(&self) -> Result<String>;

    /// Get repository status
    async fn status(&self) -> Result<GitStatus>;

    /// Get commit history
    async fn log(&self, limit: usize) -> Result<Vec<GitCommit>>;

    /// Get file diff
    async fn diff(&self, file_path: &str) -> Result<String>;

    /// Tag a commit
    async fn tag(&self, tag_name: &str, message: &str) -> Result<()>;

    /// List tags
    async fn list_tags(&self) -> Result<Vec<String>>;

    /// Get remote URLs
    async fn get_remotes(&self) -> Result<HashMap<String, String>>;

    /// Add a remote
    async fn add_remote(&self, name: &str, url: &str) -> Result<()>;
}

/// Git LFS interface
#[async_trait::async_trait]
pub trait GitLFS: Send + Sync {
    /// Initialize Git LFS
    async fn init(&self) -> Result<()>;

    /// Track file patterns with LFS
    async fn track(&self, patterns: &[&str]) -> Result<()>;

    /// Untrack file patterns from LFS
    async fn untrack(&self, patterns: &[&str]) -> Result<()>;

    /// Get LFS file info
    async fn ls_files(&self) -> Result<Vec<LFSFile>>;

    /// Check LFS file integrity
    async fn fsck(&self) -> Result<LFSStatus>;

    /// Prune old LFS files
    async fn prune(&self, days: u32) -> Result<()>;

    /// Get LFS storage info
    async fn storage_info(&self) -> Result<LFSStorageInfo>;
}

/// Git repository status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GitStatus {
    /// Current branch
    pub current_branch: String,
    /// Staged files
    pub staged_files: Vec<String>,
    /// Modified files
    pub modified_files: Vec<String>,
    /// Untracked files
    pub untracked_files: Vec<String>,
    /// Ahead/behind remote
    pub ahead_behind: Option<(usize, usize)>,
    /// Is repository clean
    pub is_clean: bool,
}

/// Git commit information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GitCommit {
    /// Commit hash
    pub hash: String,
    /// Short hash
    pub short_hash: String,
    /// Commit message
    pub message: String,
    /// Author name
    pub author_name: String,
    /// Author email
    pub author_email: String,
    /// Commit timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Files changed
    pub files_changed: Vec<String>,
}

/// LFS file information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LFSFile {
    /// File path
    pub path: String,
    /// File size
    pub size: u64,
    /// LFS pointer hash
    pub pointer_hash: String,
    /// Is file downloaded
    pub is_downloaded: bool,
}

/// LFS status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LFSStatus {
    /// Number of files tracked
    pub files_tracked: usize,
    /// Total size of tracked files
    pub total_size: u64,
    /// Number of corrupt files
    pub corrupt_files: usize,
    /// Missing files
    pub missing_files: Vec<String>,
}

/// LFS storage information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LFSStorageInfo {
    /// Total storage used
    pub total_size: u64,
    /// Available storage
    pub available_size: u64,
    /// Storage endpoint
    pub endpoint: String,
    /// Authentication status
    pub auth_status: String,
}

/// Git repository implementation
pub struct GitRepositoryImpl {
    config: GitConfig,
    repo_path: PathBuf,
}

impl GitRepositoryImpl {
    /// Create a new Git repository instance
    pub fn new(config: GitConfig) -> Result<Self> {
        let repo_path = config.local_path.clone();

        Ok(Self { config, repo_path })
    }

    /// Validate configuration
    pub fn validate_config(&self) -> Result<()> {
        if self.config.repository_url.is_empty() {
            return Err(DatasetError::Configuration(
                "Repository URL cannot be empty".to_string(),
            ));
        }

        if !self.repo_path.exists() {
            return Err(DatasetError::Configuration(format!(
                "Repository path does not exist: {}",
                self.repo_path.display()
            )));
        }

        Ok(())
    }

    /// Execute git command
    async fn execute_git_command(&self, args: &[&str]) -> Result<String> {
        let output = AsyncCommand::new("git")
            .args(args)
            .current_dir(&self.repo_path)
            .output()
            .await
            .map_err(|e| DatasetError::Git(format!("Failed to execute git command: {e}")))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(DatasetError::Git(format!("Git command failed: {stderr}")));
        }

        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    }

    /// Execute git LFS command
    #[allow(dead_code)]
    async fn execute_lfs_command(&self, args: &[&str]) -> Result<String> {
        let output = AsyncCommand::new("git")
            .arg("lfs")
            .args(args)
            .current_dir(&self.repo_path)
            .output()
            .await
            .map_err(|e| DatasetError::Git(format!("Failed to execute git lfs command: {e}")))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(DatasetError::Git(format!(
                "Git LFS command failed: {stderr}"
            )));
        }

        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    }

    /// Setup authentication
    async fn setup_auth(&self) -> Result<()> {
        match &self.config.remote_config.auth_method {
            AuthMethod::SSH { .. } => {
                // SSH authentication is handled by git automatically
                Ok(())
            }
            AuthMethod::Token {
                token: _,
                username: _,
            } => {
                // Setup credential helper for token authentication
                self.execute_git_command(&["config", "credential.helper", "store"])
                    .await?;

                // Store credentials (in a real implementation, this would be more secure)
                Ok(())
            }
            AuthMethod::UsernamePassword { .. } => {
                // Setup credential helper for username/password authentication
                self.execute_git_command(&["config", "credential.helper", "store"])
                    .await?;

                Ok(())
            }
        }
    }

    /// Parse git status output
    fn parse_git_status(&self, output: &str) -> GitStatus {
        let lines: Vec<&str> = output.lines().collect();
        let mut status = GitStatus {
            current_branch: "main".to_string(),
            staged_files: Vec::new(),
            modified_files: Vec::new(),
            untracked_files: Vec::new(),
            ahead_behind: None,
            is_clean: true,
        };

        for line in lines {
            if line.starts_with("On branch ") {
                status.current_branch = line.trim_start_matches("On branch ").to_string();
            } else if let Some(stripped) = line.strip_prefix("A  ") {
                status.staged_files.push(stripped.to_string());
                status.is_clean = false;
            } else if let Some(stripped) = line.strip_prefix(" M ") {
                status.modified_files.push(stripped.to_string());
                status.is_clean = false;
            } else if let Some(stripped) = line.strip_prefix("?? ") {
                status.untracked_files.push(stripped.to_string());
                status.is_clean = false;
            }
        }

        status
    }

    /// Parse git log output
    fn parse_git_log(&self, output: &str) -> Vec<GitCommit> {
        let mut commits = Vec::new();
        let lines: Vec<&str> = output.lines().collect();

        let mut i = 0;
        while i < lines.len() {
            if lines[i].starts_with("commit ") {
                let hash = lines[i].trim_start_matches("commit ").to_string();
                let short_hash = hash.chars().take(7).collect();

                // Skip to author line
                i += 1;
                let author_line = lines.get(i).unwrap_or(&"");
                let (author_name, author_email) = if author_line.starts_with("Author: ") {
                    let author_info = author_line.trim_start_matches("Author: ");
                    if let Some(email_start) = author_info.rfind('<') {
                        let name = author_info[..email_start].trim().to_string();
                        let email = author_info[email_start + 1..]
                            .trim_end_matches('>')
                            .to_string();
                        (name, email)
                    } else {
                        (author_info.to_string(), "".to_string())
                    }
                } else {
                    ("Unknown".to_string(), "".to_string())
                };

                // Skip to date line
                i += 1;
                let date_line = lines.get(i).unwrap_or(&"");
                let timestamp = if date_line.starts_with("Date: ") {
                    chrono::DateTime::parse_from_rfc2822(date_line.trim_start_matches("Date: "))
                        .unwrap_or_else(|_| {
                            chrono::DateTime::parse_from_rfc2822("Mon, 1 Jan 2000 00:00:00 +0000")
                                .unwrap()
                        })
                        .with_timezone(&chrono::Utc)
                } else {
                    chrono::Utc::now()
                };

                // Skip empty line and get commit message
                i += 2;
                let message = lines.get(i).unwrap_or(&"").trim().to_string();

                commits.push(GitCommit {
                    hash,
                    short_hash,
                    message,
                    author_name,
                    author_email,
                    timestamp,
                    files_changed: Vec::new(), // Would need separate command to get this
                });
            }
            i += 1;
        }

        commits
    }
}

#[async_trait::async_trait]
impl GitRepository for GitRepositoryImpl {
    async fn init(&self, path: &Path) -> Result<()> {
        let output = AsyncCommand::new("git")
            .arg("init")
            .arg(path)
            .output()
            .await
            .map_err(|e| DatasetError::Git(format!("Failed to initialize git repository: {e}")))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(DatasetError::Git(format!("Git init failed: {stderr}")));
        }

        Ok(())
    }

    async fn clone(&self, url: &str, path: &Path) -> Result<()> {
        let output = AsyncCommand::new("git")
            .arg("clone")
            .arg(url)
            .arg(path)
            .output()
            .await
            .map_err(|e| DatasetError::Git(format!("Failed to clone repository: {e}")))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(DatasetError::Git(format!("Git clone failed: {stderr}")));
        }

        Ok(())
    }

    async fn add(&self, files: &[&str]) -> Result<()> {
        let mut args = vec!["add"];
        args.extend(files);

        self.execute_git_command(&args).await?;
        Ok(())
    }

    async fn commit(&self, message: &str) -> Result<String> {
        let commit_msg = if self.config.commit_config.include_stats {
            format!("{message}\n\n[Auto-generated commit]")
        } else {
            message.to_string()
        };

        let args = if self.config.commit_config.sign_commits {
            vec!["commit", "-S", "-m", &commit_msg]
        } else {
            vec!["commit", "-m", &commit_msg]
        };

        let output = self.execute_git_command(&args).await?;

        // Extract commit hash from output
        let hash = output
            .lines()
            .find(|line| line.contains("commit"))
            .and_then(|line| line.split_whitespace().nth(1))
            .unwrap_or("unknown")
            .to_string();

        Ok(hash)
    }

    async fn push(&self, remote: &str, branch: &str) -> Result<()> {
        self.setup_auth().await?;

        let args = match self.config.remote_config.strategy {
            PushPullStrategy::Force => vec!["push", "--force", remote, branch],
            _ => vec!["push", remote, branch],
        };

        self.execute_git_command(&args).await?;
        Ok(())
    }

    async fn pull(&self, remote: &str, branch: &str) -> Result<()> {
        self.setup_auth().await?;

        let args = match self.config.remote_config.strategy {
            PushPullStrategy::Rebase => vec!["pull", "--rebase", remote, branch],
            _ => vec!["pull", remote, branch],
        };

        self.execute_git_command(&args).await?;
        Ok(())
    }

    async fn create_branch(&self, branch_name: &str) -> Result<()> {
        self.execute_git_command(&["branch", branch_name]).await?;
        Ok(())
    }

    async fn checkout(&self, branch_name: &str) -> Result<()> {
        self.execute_git_command(&["checkout", branch_name]).await?;
        Ok(())
    }

    async fn current_branch(&self) -> Result<String> {
        let output = self
            .execute_git_command(&["branch", "--show-current"])
            .await?;
        Ok(output.trim().to_string())
    }

    async fn status(&self) -> Result<GitStatus> {
        let output = self.execute_git_command(&["status", "--porcelain"]).await?;
        Ok(self.parse_git_status(&output))
    }

    async fn log(&self, limit: usize) -> Result<Vec<GitCommit>> {
        let limit_str = format!("-{limit}");
        let output = self.execute_git_command(&["log", &limit_str]).await?;
        Ok(self.parse_git_log(&output))
    }

    async fn diff(&self, file_path: &str) -> Result<String> {
        let output = self.execute_git_command(&["diff", file_path]).await?;
        Ok(output)
    }

    async fn tag(&self, tag_name: &str, message: &str) -> Result<()> {
        self.execute_git_command(&["tag", "-a", tag_name, "-m", message])
            .await?;
        Ok(())
    }

    async fn list_tags(&self) -> Result<Vec<String>> {
        let output = self.execute_git_command(&["tag", "-l"]).await?;
        Ok(output.lines().map(str::to_string).collect())
    }

    async fn get_remotes(&self) -> Result<HashMap<String, String>> {
        let output = self.execute_git_command(&["remote", "-v"]).await?;
        let mut remotes = HashMap::new();

        for line in output.lines() {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 2 {
                remotes.insert(parts[0].to_string(), parts[1].to_string());
            }
        }

        Ok(remotes)
    }

    async fn add_remote(&self, name: &str, url: &str) -> Result<()> {
        self.execute_git_command(&["remote", "add", name, url])
            .await?;
        Ok(())
    }
}

/// Git LFS implementation
pub struct GitLFSImpl {
    #[allow(dead_code)]
    config: GitConfig,
    repo_path: PathBuf,
}

impl GitLFSImpl {
    /// Create a new Git LFS instance
    pub fn new(config: GitConfig) -> Result<Self> {
        let repo_path = config.local_path.clone();

        Ok(Self { config, repo_path })
    }

    /// Execute git LFS command
    async fn execute_lfs_command(&self, args: &[&str]) -> Result<String> {
        let output = AsyncCommand::new("git")
            .arg("lfs")
            .args(args)
            .current_dir(&self.repo_path)
            .output()
            .await
            .map_err(|e| DatasetError::Git(format!("Failed to execute git lfs command: {e}")))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(DatasetError::Git(format!(
                "Git LFS command failed: {stderr}"
            )));
        }

        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    }
}

#[async_trait::async_trait]
impl GitLFS for GitLFSImpl {
    async fn init(&self) -> Result<()> {
        self.execute_lfs_command(&["install"]).await?;
        Ok(())
    }

    async fn track(&self, patterns: &[&str]) -> Result<()> {
        for pattern in patterns {
            self.execute_lfs_command(&["track", pattern]).await?;
        }
        Ok(())
    }

    async fn untrack(&self, patterns: &[&str]) -> Result<()> {
        for pattern in patterns {
            self.execute_lfs_command(&["untrack", pattern]).await?;
        }
        Ok(())
    }

    async fn ls_files(&self) -> Result<Vec<LFSFile>> {
        let output = self.execute_lfs_command(&["ls-files", "-l"]).await?;
        let mut files = Vec::new();

        for line in output.lines() {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 3 {
                files.push(LFSFile {
                    path: parts[2].to_string(),
                    size: parts[1].parse().unwrap_or(0),
                    pointer_hash: parts[0].to_string(),
                    is_downloaded: true, // Simplification
                });
            }
        }

        Ok(files)
    }

    async fn fsck(&self) -> Result<LFSStatus> {
        let _output = self.execute_lfs_command(&["fsck"]).await?;

        // Parse fsck output (simplified)
        Ok(LFSStatus {
            files_tracked: 0,
            total_size: 0,
            corrupt_files: 0,
            missing_files: Vec::new(),
        })
    }

    async fn prune(&self, days: u32) -> Result<()> {
        let days_str = format!("--verify-remote --older-than={days}d");
        self.execute_lfs_command(&["prune", &days_str]).await?;
        Ok(())
    }

    async fn storage_info(&self) -> Result<LFSStorageInfo> {
        // In a real implementation, this would query the LFS server
        Ok(LFSStorageInfo {
            total_size: 0,
            available_size: 0,
            endpoint: "".to_string(),
            auth_status: "authenticated".to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_git_config_default() {
        let config = GitConfig::default();
        assert_eq!(config.branch_config.main_branch, "main");
        assert_eq!(config.commit_config.author_name, "Dataset Bot");
        assert!(config.lfs_config.enabled);
    }

    #[test]
    fn test_lfs_config_default() {
        let config = LFSConfig::default();
        assert!(config.enabled);
        assert!(config.track_patterns.contains(&"*.wav".to_string()));
        assert_eq!(config.max_file_size, 100 * 1024 * 1024);
    }

    #[test]
    fn test_git_repository_creation() {
        let temp_dir = TempDir::new().unwrap();
        let config = GitConfig {
            local_path: temp_dir.path().to_path_buf(),
            ..Default::default()
        };

        let repo = GitRepositoryImpl::new(config);
        assert!(repo.is_ok());
    }

    #[test]
    fn test_git_lfs_creation() {
        let temp_dir = TempDir::new().unwrap();
        let config = GitConfig {
            local_path: temp_dir.path().to_path_buf(),
            ..Default::default()
        };

        let lfs = GitLFSImpl::new(config);
        assert!(lfs.is_ok());
    }

    #[test]
    fn test_git_status_parsing() {
        let temp_dir = TempDir::new().unwrap();
        let config = GitConfig {
            local_path: temp_dir.path().to_path_buf(),
            ..Default::default()
        };

        let repo = GitRepositoryImpl::new(config).unwrap();
        let status =
            repo.parse_git_status("On branch main\nA  file1.txt\n M file2.txt\n?? file3.txt\n");

        assert_eq!(status.current_branch, "main");
        assert_eq!(status.staged_files.len(), 1);
        assert_eq!(status.modified_files.len(), 1);
        assert_eq!(status.untracked_files.len(), 1);
        assert!(!status.is_clean);
    }
}

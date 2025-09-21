//! Output formatting utilities.

use console::{Color, Style, Term};
use serde_json;
use std::io::Write;

/// Output formatter for different display modes
pub struct OutputFormatter {
    colored: bool,
    json_mode: bool,
    term: Term,
}

impl OutputFormatter {
    /// Create a new output formatter
    pub fn new(colored: bool, json_mode: bool) -> Self {
        Self {
            colored,
            json_mode,
            term: Term::stdout(),
        }
    }

    /// Print a success message
    pub fn success(&self, msg: &str) {
        if self.json_mode {
            self.print_json("success", msg, None);
        } else if self.colored {
            let style = Style::new().fg(Color::Green).bold();
            println!("{} {}", style.apply_to("✓"), msg);
        } else {
            println!("✓ {}", msg);
        }
    }

    /// Print an error message
    pub fn error(&self, msg: &str) {
        if self.json_mode {
            self.print_json("error", msg, None);
        } else if self.colored {
            let style = Style::new().fg(Color::Red).bold();
            eprintln!("{} {}", style.apply_to("✗"), msg);
        } else {
            eprintln!("✗ {}", msg);
        }
    }

    /// Print a warning message
    pub fn warning(&self, msg: &str) {
        if self.json_mode {
            self.print_json("warning", msg, None);
        } else if self.colored {
            let style = Style::new().fg(Color::Yellow).bold();
            println!("{} {}", style.apply_to("⚠"), msg);
        } else {
            println!("⚠ {}", msg);
        }
    }

    /// Print an info message
    pub fn info(&self, msg: &str) {
        if self.json_mode {
            self.print_json("info", msg, None);
        } else if self.colored {
            let style = Style::new().fg(Color::Blue);
            println!("{} {}", style.apply_to("ℹ"), msg);
        } else {
            println!("ℹ {}", msg);
        }
    }

    /// Print a header
    pub fn header(&self, title: &str) {
        if self.json_mode {
            self.print_json("header", title, None);
        } else if self.colored {
            let style = Style::new().fg(Color::Cyan).bold().underlined();
            println!("{}", style.apply_to(title));
        } else {
            println!("{}", title);
            println!("{}", "=".repeat(title.len()));
        }
    }

    /// Print a list item
    pub fn list_item(&self, item: &str, indent: usize) {
        let spaces = " ".repeat(indent * 2);
        if self.json_mode {
            self.print_json("list_item", item, Some(&format!("indent:{}", indent)));
        } else if self.colored {
            let style = Style::new().fg(Color::White);
            println!("{}• {}", spaces, style.apply_to(item));
        } else {
            println!("{}• {}", spaces, item);
        }
    }

    /// Print key-value pair
    pub fn key_value(&self, key: &str, value: &str) {
        if self.json_mode {
            let data = serde_json::json!({ key: value });
            println!("{}", serde_json::to_string(&data).unwrap_or_default());
        } else if self.colored {
            let key_style = Style::new().fg(Color::Cyan).bold();
            let value_style = Style::new().fg(Color::White);
            println!(
                "{}: {}",
                key_style.apply_to(key),
                value_style.apply_to(value)
            );
        } else {
            println!("{}: {}", key, value);
        }
    }

    /// Print a table header
    pub fn table_header(&self, headers: &[&str]) {
        if self.json_mode {
            self.print_json("table_header", &headers.join(","), None);
        } else if self.colored {
            let style = Style::new().fg(Color::Cyan).bold();
            let header_line = headers
                .iter()
                .map(|h| format!("{:20}", h))
                .collect::<Vec<_>>()
                .join(" ");
            println!("{}", style.apply_to(&header_line));
            println!("{}", style.apply_to(&"-".repeat(header_line.len())));
        } else {
            let header_line = headers
                .iter()
                .map(|h| format!("{:20}", h))
                .collect::<Vec<_>>()
                .join(" ");
            println!("{}", header_line);
            println!("{}", "-".repeat(header_line.len()));
        }
    }

    /// Print a table row
    pub fn table_row(&self, cells: &[&str]) {
        if self.json_mode {
            let data = serde_json::json!(cells);
            println!("{}", serde_json::to_string(&data).unwrap_or_default());
        } else {
            let row = cells
                .iter()
                .map(|c| format!("{:20}", c))
                .collect::<Vec<_>>()
                .join(" ");
            println!("{}", row);
        }
    }

    /// Print command output
    pub fn command_output(&self, cmd: &str, output: &str) {
        if self.json_mode {
            let data = serde_json::json!({
                "command": cmd,
                "output": output
            });
            println!("{}", serde_json::to_string(&data).unwrap_or_default());
        } else if self.colored {
            let cmd_style = Style::new().fg(Color::Green).bold();
            println!("{} {}", cmd_style.apply_to("$"), cmd);
            if !output.is_empty() {
                println!("{}", output);
            }
        } else {
            println!("$ {}", cmd);
            if !output.is_empty() {
                println!("{}", output);
            }
        }
    }

    /// Print structured data as JSON or formatted text
    pub fn structured_data(&self, data: &serde_json::Value) {
        if self.json_mode {
            println!("{}", serde_json::to_string_pretty(data).unwrap_or_default());
        } else {
            self.print_value(data, 0);
        }
    }

    /// Internal method to print JSON messages
    fn print_json(&self, level: &str, message: &str, extra: Option<&str>) {
        let mut data = serde_json::json!({
            "level": level,
            "message": message,
            "timestamp": chrono::Utc::now().to_rfc3339()
        });

        if let Some(extra_data) = extra {
            data["extra"] = serde_json::Value::String(extra_data.to_string());
        }

        println!("{}", serde_json::to_string(&data).unwrap_or_default());
    }

    /// Recursively print JSON values in a readable format
    fn print_value(&self, value: &serde_json::Value, indent: usize) {
        let spaces = "  ".repeat(indent);

        match value {
            serde_json::Value::Object(map) => {
                for (key, val) in map {
                    if self.colored {
                        let key_style = Style::new().fg(Color::Cyan);
                        print!("{}{}: ", spaces, key_style.apply_to(key));
                    } else {
                        print!("{}{}: ", spaces, key);
                    }

                    match val {
                        serde_json::Value::Object(_) | serde_json::Value::Array(_) => {
                            println!();
                            self.print_value(val, indent + 1);
                        }
                        _ => {
                            self.print_value(val, 0);
                        }
                    }
                }
            }
            serde_json::Value::Array(arr) => {
                for (i, val) in arr.iter().enumerate() {
                    print!("{}[{}] ", spaces, i);
                    self.print_value(val, indent);
                }
            }
            serde_json::Value::String(s) => {
                if self.colored {
                    let style = Style::new().fg(Color::Green);
                    println!("{}", style.apply_to(&format!("\"{}\"", s)));
                } else {
                    println!("\"{}\"", s);
                }
            }
            serde_json::Value::Number(n) => {
                if self.colored {
                    let style = Style::new().fg(Color::Yellow);
                    println!("{}", style.apply_to(&n.to_string()));
                } else {
                    println!("{}", n);
                }
            }
            serde_json::Value::Bool(b) => {
                if self.colored {
                    let style = Style::new().fg(Color::Magenta);
                    println!("{}", style.apply_to(&b.to_string()));
                } else {
                    println!("{}", b);
                }
            }
            serde_json::Value::Null => {
                if self.colored {
                    let style = Style::new().fg(Color::Black).italic();
                    println!("{}", style.apply_to("null"));
                } else {
                    println!("null");
                }
            }
        }
    }
}

/// Global formatter instance
static mut FORMATTER: Option<OutputFormatter> = None;

/// Initialize global formatter
pub fn init_formatter(colored: bool, json_mode: bool) {
    unsafe {
        FORMATTER = Some(OutputFormatter::new(colored, json_mode));
    }
}

/// Get global formatter
pub fn get_formatter() -> &'static OutputFormatter {
    unsafe {
        FORMATTER.as_ref().unwrap_or_else(|| {
            FORMATTER = Some(OutputFormatter::new(true, false));
            FORMATTER.as_ref().unwrap()
        })
    }
}

/// Convenience macros for formatted output
#[macro_export]
macro_rules! success {
    ($($arg:tt)*) => {
        crate::output::get_formatter().success(&format!($($arg)*))
    };
}

#[macro_export]
macro_rules! error {
    ($($arg:tt)*) => {
        crate::output::get_formatter().error(&format!($($arg)*))
    };
}

#[macro_export]
macro_rules! warning {
    ($($arg:tt)*) => {
        crate::output::get_formatter().warning(&format!($($arg)*))
    };
}

#[macro_export]
macro_rules! info {
    ($($arg:tt)*) => {
        crate::output::get_formatter().info(&format!($($arg)*))
    };
}

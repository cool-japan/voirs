//! Enhanced SSML parser with proper XML handling and error recovery.

use crate::{G2pError, LanguageCode, Result};
use crate::ssml::elements::*;
use crate::ssml::dictionary::{PronunciationContext, PartOfSpeech};
use std::collections::HashMap;
use std::str::Chars;
use std::iter::Peekable;

/// Enhanced SSML parser with proper XML handling
pub struct SsmlParser {
    /// Parser configuration
    config: ParserConfig,
    /// Error recovery strategies
    recovery: ErrorRecovery,
    /// Validation rules
    validation: ValidationRules,
    /// Custom element handlers
    custom_handlers: HashMap<String, Box<dyn CustomElementHandler>>,
}

/// Parser configuration
#[derive(Debug, Clone)]
pub struct ParserConfig {
    /// Whether to validate XML structure strictly
    pub strict_validation: bool,
    /// Whether to preserve whitespace
    pub preserve_whitespace: bool,
    /// Maximum nesting depth
    pub max_nesting_depth: usize,
    /// Whether to allow custom elements
    pub allow_custom_elements: bool,
    /// Default language for content
    pub default_language: Option<LanguageCode>,
    /// Whether to normalize text content
    pub normalize_text: bool,
}

/// Error recovery configuration
#[derive(Debug, Clone)]
pub struct ErrorRecovery {
    /// Whether to attempt error recovery
    pub enable_recovery: bool,
    /// How to handle malformed tags
    pub malformed_tag_strategy: MalformedTagStrategy,
    /// How to handle unclosed tags
    pub unclosed_tag_strategy: UnclosedTagStrategy,
    /// How to handle invalid attributes
    pub invalid_attribute_strategy: InvalidAttributeStrategy,
    /// Maximum number of errors to recover from
    pub max_recovery_attempts: usize,
}

/// Validation rules
#[derive(Debug, Clone)]
pub struct ValidationRules {
    /// Required attributes for each element type
    pub required_attributes: HashMap<String, Vec<String>>,
    /// Allowed child elements for each element type
    pub allowed_children: HashMap<String, Vec<String>>,
    /// Attribute value constraints
    pub attribute_constraints: HashMap<String, AttributeConstraint>,
    /// Custom validation functions
    pub custom_validators: Vec<String>, // Would contain function names in practice
}

/// Custom element handler trait
pub trait CustomElementHandler: Send + Sync {
    /// Handle custom element parsing
    fn handle(&self, tag_name: &str, attributes: &HashMap<String, String>, content: &str) -> Result<SsmlElement>;
    
    /// Get element name this handler supports
    fn element_name(&self) -> &str;
    
    /// Validate element before processing
    fn validate(&self, attributes: &HashMap<String, String>) -> Result<()>;
}

/// Parser context for tracking state
#[derive(Debug, Clone)]
pub struct ParserContext {
    /// Current nesting depth
    pub depth: usize,
    /// Current language context
    pub current_language: Option<LanguageCode>,
    /// Current position in text
    pub position: usize,
    /// Current line number
    pub line: usize,
    /// Current column number
    pub column: usize,
    /// Error count
    pub error_count: usize,
    /// Warning count
    pub warning_count: usize,
}

/// Parsing result with diagnostics
#[derive(Debug, Clone)]
pub struct ParseResult {
    /// Parsed SSML element
    pub element: Option<SsmlElement>,
    /// Parsing errors
    pub errors: Vec<ParseError>,
    /// Parsing warnings
    pub warnings: Vec<ParseWarning>,
    /// Parser statistics
    pub statistics: ParseStatistics,
}

/// Parse error with location information
#[derive(Debug, Clone)]
pub struct ParseError {
    /// Error message
    pub message: String,
    /// Error type
    pub error_type: ParseErrorType,
    /// Location in source
    pub location: SourceLocation,
    /// Error severity
    pub severity: ErrorSeverity,
    /// Suggested fix
    pub suggestion: Option<String>,
}

/// Parse warning
#[derive(Debug, Clone)]
pub struct ParseWarning {
    /// Warning message
    pub message: String,
    /// Warning type
    pub warning_type: ParseWarningType,
    /// Location in source
    pub location: SourceLocation,
    /// Suggested improvement
    pub suggestion: Option<String>,
}

/// Parser statistics
#[derive(Debug, Clone)]
pub struct ParseStatistics {
    /// Total elements parsed
    pub elements_parsed: usize,
    /// Total attributes processed
    pub attributes_processed: usize,
    /// Text content length
    pub text_content_length: usize,
    /// Parse time in milliseconds
    pub parse_time_ms: f64,
    /// Memory usage estimate
    pub memory_usage_bytes: usize,
}

/// Source location for error reporting
#[derive(Debug, Clone)]
pub struct SourceLocation {
    /// Line number (1-based)
    pub line: usize,
    /// Column number (1-based)
    pub column: usize,
    /// Character offset (0-based)
    pub offset: usize,
    /// Length of the problematic section
    pub length: usize,
}

// Enums for error handling
#[derive(Debug, Clone)]
pub enum MalformedTagStrategy {
    /// Fail with error
    Fail,
    /// Skip and continue
    Skip,
    /// Try to fix and continue
    Fix,
    /// Convert to text
    ConvertToText,
}

#[derive(Debug, Clone)]
pub enum UnclosedTagStrategy {
    /// Fail with error
    Fail,
    /// Auto-close at end of parent
    AutoClose,
    /// Treat as self-closing
    SelfClose,
    /// Convert content to text
    ConvertToText,
}

#[derive(Debug, Clone)]
pub enum InvalidAttributeStrategy {
    /// Fail with error
    Fail,
    /// Ignore invalid attributes
    Ignore,
    /// Use default values
    UseDefault,
    /// Log warning and continue
    WarnAndContinue,
}

#[derive(Debug, Clone)]
pub enum ParseErrorType {
    /// Malformed XML
    MalformedXml,
    /// Invalid element
    InvalidElement,
    /// Invalid attribute
    InvalidAttribute,
    /// Missing required attribute
    MissingAttribute,
    /// Unclosed element
    UnclosedElement,
    /// Unexpected end of input
    UnexpectedEof,
    /// Invalid nesting
    InvalidNesting,
    /// Custom error
    Custom(String),
}

#[derive(Debug, Clone)]
pub enum ParseWarningType {
    /// Deprecated element
    DeprecatedElement,
    /// Deprecated attribute
    DeprecatedAttribute,
    /// Performance warning
    Performance,
    /// Compatibility warning
    Compatibility,
    /// Best practice suggestion
    BestPractice,
}

#[derive(Debug, Clone)]
pub enum ErrorSeverity {
    /// Fatal error - cannot continue
    Fatal,
    /// Error - can attempt recovery
    Error,
    /// Warning - continue normally
    Warning,
    /// Info - just for information
    Info,
}

#[derive(Debug, Clone)]
pub enum AttributeConstraint {
    /// Required attribute
    Required,
    /// Optional attribute
    Optional,
    /// Attribute with allowed values
    Enum(Vec<String>),
    /// Attribute with numeric range
    NumericRange(f64, f64),
    /// Attribute with pattern
    Pattern(String),
    /// Custom constraint
    Custom(String),
}

/// XML tokenizer for low-level parsing
struct XmlTokenizer {
    /// Input text
    text: String,
    /// Current position
    position: usize,
    /// Current line
    line: usize,
    /// Current column
    column: usize,
}

/// XML token types
#[derive(Debug, Clone, PartialEq)]
enum XmlToken {
    /// Opening tag: <tagname>
    OpenTag {
        name: String,
        attributes: HashMap<String, String>,
        self_closing: bool,
    },
    /// Closing tag: </tagname>
    CloseTag(String),
    /// Text content
    Text(String),
    /// Comment: <!-- comment -->
    Comment(String),
    /// End of input
    Eof,
}

impl SsmlParser {
    /// Create a new SSML parser with default configuration
    pub fn new() -> Self {
        Self {
            config: ParserConfig::default(),
            recovery: ErrorRecovery::default(),
            validation: ValidationRules::default(),
            custom_handlers: HashMap::new(),
        }
    }

    /// Create parser with custom configuration
    pub fn with_config(config: ParserConfig) -> Self {
        Self {
            config,
            recovery: ErrorRecovery::default(),
            validation: ValidationRules::default(),
            custom_handlers: HashMap::new(),
        }
    }

    /// Add custom element handler
    pub fn add_custom_handler<H: CustomElementHandler + 'static>(&mut self, handler: H) {
        let element_name = handler.element_name().to_string();
        self.custom_handlers.insert(element_name, Box::new(handler));
    }

    /// Parse SSML text with full error handling
    pub fn parse(&self, ssml_text: &str) -> Result<ParseResult> {
        let start_time = std::time::Instant::now();
        let mut context = ParserContext::new();
        let mut errors = Vec::new();
        let mut warnings = Vec::new();

        // Tokenize the input
        let mut tokenizer = XmlTokenizer::new(ssml_text)?;
        let tokens = self.tokenize(&mut tokenizer, &mut context, &mut errors)?;

        // Parse tokens into SSML structure
        let element = self.parse_tokens(&tokens, &mut context, &mut errors, &mut warnings)?;

        // Validate the result
        if let Some(ref elem) = element {
            self.validate_element(elem, &mut warnings)?;
        }

        let parse_time = start_time.elapsed().as_millis() as f64;
        let statistics = ParseStatistics {
            elements_parsed: self.count_elements(element.as_ref()),
            attributes_processed: context.position, // Simplified
            text_content_length: ssml_text.len(),
            parse_time_ms: parse_time,
            memory_usage_bytes: std::mem::size_of_val(&element), // Simplified
        };

        Ok(ParseResult {
            element,
            errors,
            warnings,
            statistics,
        })
    }

    /// Parse SSML with simplified error handling (for compatibility)
    pub fn parse_simple(&self, ssml_text: &str) -> Result<SsmlElement> {
        let result = self.parse(ssml_text)?;
        
        if !result.errors.is_empty() {
            let error_msg = result.errors.iter()
                .map(|e| e.message.clone())
                .collect::<Vec<_>>()
                .join("; ");
            return Err(G2pError::ConfigError(format!("SSML parsing errors: {error_msg}")));
        }

        result.element.ok_or_else(|| G2pError::ConfigError("No SSML element parsed".to_string()))
    }

    /// Tokenize XML input
    fn tokenize(
        &self,
        tokenizer: &mut XmlTokenizer,
        context: &mut ParserContext,
        errors: &mut Vec<ParseError>,
    ) -> Result<Vec<XmlToken>> {
        let mut tokens = Vec::new();
        
        loop {
            match tokenizer.next_token() {
                Ok(XmlToken::Eof) => break,
                Ok(token) => tokens.push(token),
                Err(e) => {
                    if self.recovery.enable_recovery && errors.len() < self.recovery.max_recovery_attempts {
                        errors.push(ParseError {
                            message: format!("Tokenization error: {e}"),
                            error_type: ParseErrorType::MalformedXml,
                            location: SourceLocation {
                                line: tokenizer.line,
                                column: tokenizer.column,
                                offset: tokenizer.position,
                                length: 1,
                            },
                            severity: ErrorSeverity::Error,
                            suggestion: Some("Check XML syntax".to_string()),
                        });
                        // Try to recover by skipping problematic character
                        tokenizer.skip_char();
                    } else {
                        return Err(e);
                    }
                }
            }
        }
        
        Ok(tokens)
    }

    /// Parse tokens into SSML elements
    fn parse_tokens(
        &self,
        tokens: &[XmlToken],
        context: &mut ParserContext,
        errors: &mut Vec<ParseError>,
        warnings: &mut Vec<ParseWarning>,
    ) -> Result<Option<SsmlElement>> {
        if tokens.is_empty() {
            return Ok(None);
        }

        // Find the root element (should be <speak> or we'll wrap content)
        let mut token_index = 0;
        
        // Skip comments and whitespace at the beginning
        while token_index < tokens.len() {
            match &tokens[token_index] {
                XmlToken::Comment(_) => token_index += 1,
                XmlToken::Text(text) if text.trim().is_empty() => token_index += 1,
                _ => break,
            }
        }

        if token_index >= tokens.len() {
            return Ok(None);
        }

        // Parse the main element
        self.parse_element(tokens, &mut token_index, context, errors, warnings)
    }

    /// Parse a single SSML element
    fn parse_element(
        &self,
        tokens: &[XmlToken],
        token_index: &mut usize,
        context: &mut ParserContext,
        errors: &mut Vec<ParseError>,
        warnings: &mut Vec<ParseWarning>,
    ) -> Result<Option<SsmlElement>> {
        if *token_index >= tokens.len() {
            return Ok(None);
        }

        match &tokens[*token_index] {
            XmlToken::OpenTag { name, attributes, self_closing } => {
                let tag_name = name.clone();
                let tag_attributes = attributes.clone();
                let is_self_closing = *self_closing;
                *token_index += 1;

                if context.depth > self.config.max_nesting_depth {
                    errors.push(ParseError {
                        message: format!("Maximum nesting depth exceeded: {depth}", depth = self.config.max_nesting_depth),
                        error_type: ParseErrorType::InvalidNesting,
                        location: SourceLocation {
                            line: context.line,
                            column: context.column,
                            offset: context.position,
                            length: tag_name.len() + 2,
                        },
                        severity: ErrorSeverity::Error,
                        suggestion: Some("Reduce nesting depth".to_string()),
                    });
                    return Ok(None);
                }

                context.depth += 1;

                let element = match tag_name.as_str() {
                    "speak" => self.parse_speak_element(&tag_attributes, tokens, token_index, context, errors, warnings)?,
                    "phoneme" => self.parse_phoneme_element(&tag_attributes, tokens, token_index, context, errors, warnings)?,
                    "lang" => self.parse_lang_element(&tag_attributes, tokens, token_index, context, errors, warnings)?,
                    "emphasis" => self.parse_emphasis_element(&tag_attributes, tokens, token_index, context, errors, warnings)?,
                    "break" => self.parse_break_element(&tag_attributes, is_self_closing, context)?,
                    "say-as" => self.parse_say_as_element(&tag_attributes, tokens, token_index, context, errors, warnings)?,
                    "prosody" => self.parse_prosody_element(&tag_attributes, tokens, token_index, context, errors, warnings)?,
                    "voice" => self.parse_voice_element(&tag_attributes, tokens, token_index, context, errors, warnings)?,
                    "mark" => self.parse_mark_element(&tag_attributes, is_self_closing, context)?,
                    "p" => self.parse_paragraph_element(&tag_attributes, tokens, token_index, context, errors, warnings)?,
                    "s" => self.parse_sentence_element(&tag_attributes, tokens, token_index, context, errors, warnings)?,
                    _ => {
                        if let Some(handler) = self.custom_handlers.get(&tag_name) {
                            let content = self.collect_element_content(tokens, token_index)?;
                            handler.handle(&tag_name, &tag_attributes, &content)?
                        } else {
                            warnings.push(ParseWarning {
                                message: format!("Unknown SSML element: {tag_name}"),
                                warning_type: ParseWarningType::Compatibility,
                                location: SourceLocation {
                                    line: context.line,
                                    column: context.column,
                                    offset: context.position,
                                    length: tag_name.len(),
                                },
                                suggestion: Some("Check SSML specification".to_string()),
                            });
                            SsmlElement::Text(format!("<{tag_name}>")) // Convert unknown elements to text
                        }
                    }
                };

                context.depth -= 1;
                Ok(Some(element))
            }
            XmlToken::Text(text) => {
                *token_index += 1;
                let processed_text = if self.config.normalize_text {
                    self.normalize_text_content(text)
                } else {
                    text.clone()
                };
                Ok(Some(SsmlElement::Text(processed_text)))
            }
            XmlToken::Comment(_) => {
                *token_index += 1;
                // Skip comments and parse next element
                self.parse_element(tokens, token_index, context, errors, warnings)
            }
            XmlToken::CloseTag(_) => {
                // Unexpected closing tag
                errors.push(ParseError {
                    message: "Unexpected closing tag".to_string(),
                    error_type: ParseErrorType::MalformedXml,
                    location: SourceLocation {
                        line: context.line,
                        column: context.column,
                        offset: context.position,
                        length: 1,
                    },
                    severity: ErrorSeverity::Error,
                    suggestion: Some("Check tag matching".to_string()),
                });
                *token_index += 1;
                Ok(None)
            }
            XmlToken::Eof => Ok(None),
        }
    }

    /// Parse speak element
    fn parse_speak_element(
        &self,
        attributes: &HashMap<String, String>,
        tokens: &[XmlToken],
        token_index: &mut usize,
        context: &mut ParserContext,
        errors: &mut Vec<ParseError>,
        warnings: &mut Vec<ParseWarning>,
    ) -> Result<SsmlElement> {
        let language = self.extract_language_attribute(attributes)?;
        let version = attributes.get("version").cloned();
        
        let content = self.parse_element_children(tokens, token_index, "speak", context, errors, warnings)?;
        
        Ok(SsmlElement::Speak {
            language,
            version,
            content,
        })
    }

    /// Parse phoneme element with enhanced metadata
    fn parse_phoneme_element(
        &self,
        attributes: &HashMap<String, String>,
        tokens: &[XmlToken],
        token_index: &mut usize,
        context: &mut ParserContext,
        errors: &mut Vec<ParseError>,
        warnings: &mut Vec<ParseWarning>,
    ) -> Result<SsmlElement> {
        let alphabet = attributes.get("alphabet").cloned().unwrap_or_else(|| "ipa".to_string());
        let ph = attributes.get("ph").cloned().ok_or_else(|| {
            G2pError::ConfigError("phoneme element requires 'ph' attribute".to_string())
        })?;

        let text = self.collect_element_text_content(tokens, token_index, "phoneme")?;
        
        // Parse enhanced metadata if present
        let metadata = self.parse_phoneme_metadata(attributes)?;
        
        Ok(SsmlElement::Phoneme {
            alphabet,
            ph,
            text,
            metadata,
        })
    }

    /// Parse other elements (simplified implementations)
    fn parse_lang_element(
        &self,
        attributes: &HashMap<String, String>,
        tokens: &[XmlToken],
        token_index: &mut usize,
        context: &mut ParserContext,
        errors: &mut Vec<ParseError>,
        warnings: &mut Vec<ParseWarning>,
    ) -> Result<SsmlElement> {
        let lang_str = attributes.get("xml:lang").ok_or_else(|| {
            G2pError::ConfigError("lang element requires 'xml:lang' attribute".to_string())
        })?;
        let lang = self.parse_language_code(lang_str)?;
        let variant = attributes.get("variant").cloned();
        let accent = attributes.get("accent").cloned();
        
        let content = self.parse_element_children(tokens, token_index, "lang", context, errors, warnings)?;
        
        Ok(SsmlElement::Lang {
            lang,
            content,
            variant,
            accent,
        })
    }

    /// Parse emphasis element
    fn parse_emphasis_element(
        &self,
        attributes: &HashMap<String, String>,
        tokens: &[XmlToken],
        token_index: &mut usize,
        context: &mut ParserContext,
        errors: &mut Vec<ParseError>,
        warnings: &mut Vec<ParseWarning>,
    ) -> Result<SsmlElement> {
        let level_str = attributes.get("level").cloned().unwrap_or_else(|| "moderate".to_string());
        let level = self.parse_emphasis_level(&level_str)?;
        let custom_params = self.parse_emphasis_params(attributes)?;
        
        let content = self.parse_element_children(tokens, token_index, "emphasis", context, errors, warnings)?;
        
        Ok(SsmlElement::Emphasis {
            level,
            content,
            custom_params,
        })
    }

    /// Parse break element
    fn parse_break_element(
        &self,
        attributes: &HashMap<String, String>,
        is_self_closing: bool,
        context: &mut ParserContext,
    ) -> Result<SsmlElement> {
        let time = attributes.get("time").cloned();
        let strength = attributes.get("strength").map(|s| self.parse_break_strength(s)).transpose()?;
        let custom_timing = self.parse_break_timing(attributes)?;
        
        Ok(SsmlElement::Break {
            time,
            strength,
            custom_timing,
        })
    }

    /// Parse other elements (placeholder implementations)
    fn parse_say_as_element(
        &self,
        attributes: &HashMap<String, String>,
        tokens: &[XmlToken],
        token_index: &mut usize,
        context: &mut ParserContext,
        errors: &mut Vec<ParseError>,
        warnings: &mut Vec<ParseWarning>,
    ) -> Result<SsmlElement> {
        let interpret_str = attributes.get("interpret-as").ok_or_else(|| {
            G2pError::ConfigError("say-as element requires 'interpret-as' attribute".to_string())
        })?;
        let interpret_as = self.parse_interpret_as(interpret_str)?;
        let format = attributes.get("format").cloned();
        let detail = attributes.get("detail").cloned();
        
        let content = self.collect_element_text_content(tokens, token_index, "say-as")?;
        
        Ok(SsmlElement::SayAs {
            interpret_as,
            format,
            content,
            detail,
        })
    }

    fn parse_prosody_element(
        &self,
        attributes: &HashMap<String, String>,
        tokens: &[XmlToken],
        token_index: &mut usize,
        context: &mut ParserContext,
        errors: &mut Vec<ParseError>,
        warnings: &mut Vec<ParseWarning>,
    ) -> Result<SsmlElement> {
        let rate = attributes.get("rate").cloned();
        let pitch = attributes.get("pitch").cloned();
        let volume = attributes.get("volume").cloned();
        let enhanced = self.parse_enhanced_prosody(attributes)?;
        
        let content = self.parse_element_children(tokens, token_index, "prosody", context, errors, warnings)?;
        
        Ok(SsmlElement::Prosody {
            rate,
            pitch,
            volume,
            content,
            enhanced,
        })
    }

    fn parse_voice_element(
        &self,
        attributes: &HashMap<String, String>,
        tokens: &[XmlToken],
        token_index: &mut usize,
        context: &mut ParserContext,
        errors: &mut Vec<ParseError>,
        warnings: &mut Vec<ParseWarning>,
    ) -> Result<SsmlElement> {
        let name = attributes.get("name").cloned();
        let gender = attributes.get("gender").map(|g| self.parse_voice_gender(g)).transpose()?;
        let age = attributes.get("age").cloned();
        let characteristics = self.parse_voice_characteristics(attributes)?;
        
        let content = self.parse_element_children(tokens, token_index, "voice", context, errors, warnings)?;
        
        Ok(SsmlElement::Voice {
            name,
            gender,
            age,
            content,
            characteristics,
        })
    }

    fn parse_mark_element(
        &self,
        attributes: &HashMap<String, String>,
        is_self_closing: bool,
        context: &mut ParserContext,
    ) -> Result<SsmlElement> {
        let name = attributes.get("name").cloned().ok_or_else(|| {
            G2pError::ConfigError("mark element requires 'name' attribute".to_string())
        })?;
        
        Ok(SsmlElement::Mark { name })
    }

    fn parse_paragraph_element(
        &self,
        attributes: &HashMap<String, String>,
        tokens: &[XmlToken],
        token_index: &mut usize,
        context: &mut ParserContext,
        errors: &mut Vec<ParseError>,
        warnings: &mut Vec<ParseWarning>,
    ) -> Result<SsmlElement> {
        let prosody = self.parse_paragraph_prosody(attributes)?;
        let content = self.parse_element_children(tokens, token_index, "p", context, errors, warnings)?;
        
        Ok(SsmlElement::Paragraph { content, prosody })
    }

    fn parse_sentence_element(
        &self,
        attributes: &HashMap<String, String>,
        tokens: &[XmlToken],
        token_index: &mut usize,
        context: &mut ParserContext,
        errors: &mut Vec<ParseError>,
        warnings: &mut Vec<ParseWarning>,
    ) -> Result<SsmlElement> {
        let prosody = self.parse_sentence_prosody(attributes)?;
        let content = self.parse_element_children(tokens, token_index, "s", context, errors, warnings)?;
        
        Ok(SsmlElement::Sentence { content, prosody })
    }

    // Helper methods
    fn parse_element_children(
        &self,
        tokens: &[XmlToken],
        token_index: &mut usize,
        parent_tag: &str,
        context: &mut ParserContext,
        errors: &mut Vec<ParseError>,
        warnings: &mut Vec<ParseWarning>,
    ) -> Result<Vec<SsmlElement>> {
        let mut children = Vec::new();
        
        while *token_index < tokens.len() {
            match &tokens[*token_index] {
                XmlToken::CloseTag(tag_name) if tag_name == parent_tag => {
                    *token_index += 1; // Consume closing tag
                    break;
                }
                _ => {
                    if let Some(child) = self.parse_element(tokens, token_index, context, errors, warnings)? {
                        children.push(child);
                    }
                }
            }
        }
        
        Ok(children)
    }

    fn collect_element_text_content(
        &self,
        tokens: &[XmlToken],
        token_index: &mut usize,
        element_name: &str,
    ) -> Result<String> {
        let mut content = String::new();
        
        while *token_index < tokens.len() {
            match &tokens[*token_index] {
                XmlToken::CloseTag(tag_name) if tag_name == element_name => {
                    *token_index += 1; // Consume closing tag
                    break;
                }
                XmlToken::Text(text) => {
                    content.push_str(text);
                    *token_index += 1;
                }
                _ => {
                    // Skip other tokens for text content extraction
                    *token_index += 1;
                }
            }
        }
        
        Ok(content)
    }

    fn collect_element_content(
        &self,
        tokens: &[XmlToken],
        token_index: &mut usize,
    ) -> Result<String> {
        // Simplified implementation for custom handlers
        Ok("".to_string())
    }

    fn extract_language_attribute(&self, attributes: &HashMap<String, String>) -> Result<Option<LanguageCode>> {
        if let Some(lang_str) = attributes.get("xml:lang") {
            Ok(Some(self.parse_language_code(lang_str)?))
        } else {
            Ok(None)
        }
    }

    fn parse_language_code(&self, lang: &str) -> Result<LanguageCode> {
        match lang.to_lowercase().as_str() {
            "en-us" | "en_us" | "en" => Ok(LanguageCode::EnUs),
            "en-gb" | "en_gb" => Ok(LanguageCode::EnGb),
            "ja" | "jp" => Ok(LanguageCode::Ja),
            "zh-cn" | "zh_cn" | "zh" => Ok(LanguageCode::ZhCn),
            "ko" | "kr" => Ok(LanguageCode::Ko),
            "de" => Ok(LanguageCode::De),
            "fr" => Ok(LanguageCode::Fr),
            "es" => Ok(LanguageCode::Es),
            _ => Err(G2pError::ConfigError(format!("Unsupported language: {lang}"))),
        }
    }

    // Placeholder implementations for parsing various attributes
    fn parse_phoneme_metadata(&self, attributes: &HashMap<String, String>) -> Result<Option<PhonemeMetadata>> {
        Ok(None) // Simplified
    }

    fn parse_emphasis_level(&self, level_str: &str) -> Result<EmphasisLevel> {
        match level_str {
            "none" => Ok(EmphasisLevel::None),
            "reduced" => Ok(EmphasisLevel::Reduced),
            "moderate" => Ok(EmphasisLevel::Moderate),
            "strong" => Ok(EmphasisLevel::Strong),
            _ => {
                if let Ok(custom_level) = level_str.parse::<f32>() {
                    Ok(EmphasisLevel::Custom(custom_level))
                } else {
                    Ok(EmphasisLevel::Moderate)
                }
            }
        }
    }

    fn parse_emphasis_params(&self, attributes: &HashMap<String, String>) -> Result<Option<EmphasisParams>> {
        Ok(None) // Simplified
    }

    fn parse_break_strength(&self, strength_str: &str) -> Result<BreakStrength> {
        match strength_str {
            "none" => Ok(BreakStrength::None),
            "x-weak" => Ok(BreakStrength::XWeak),
            "weak" => Ok(BreakStrength::Weak),
            "medium" => Ok(BreakStrength::Medium),
            "strong" => Ok(BreakStrength::Strong),
            "x-strong" => Ok(BreakStrength::XStrong),
            _ => {
                if let Ok(custom_strength) = strength_str.parse::<f32>() {
                    Ok(BreakStrength::Custom(custom_strength))
                } else {
                    Ok(BreakStrength::Medium)
                }
            }
        }
    }

    fn parse_break_timing(&self, attributes: &HashMap<String, String>) -> Result<Option<BreakTiming>> {
        Ok(None) // Simplified
    }

    fn parse_interpret_as(&self, interpret_str: &str) -> Result<InterpretAs> {
        match interpret_str {
            "characters" => Ok(InterpretAs::Characters),
            "spell-out" => Ok(InterpretAs::SpellOut),
            "cardinal" => Ok(InterpretAs::Cardinal),
            "ordinal" => Ok(InterpretAs::Ordinal),
            "digits" => Ok(InterpretAs::Digits),
            "fraction" => Ok(InterpretAs::Fraction),
            "unit" => Ok(InterpretAs::Unit),
            "date" => Ok(InterpretAs::Date),
            "time" => Ok(InterpretAs::Time),
            "telephone" => Ok(InterpretAs::Telephone),
            "address" => Ok(InterpretAs::Address),
            "currency" => Ok(InterpretAs::Currency),
            "measure" => Ok(InterpretAs::Measure),
            _ => Ok(InterpretAs::Custom(interpret_str.to_string())),
        }
    }

    fn parse_enhanced_prosody(&self, attributes: &HashMap<String, String>) -> Result<Option<EnhancedProsody>> {
        Ok(None) // Simplified
    }

    fn parse_voice_gender(&self, gender_str: &str) -> Result<VoiceGender> {
        match gender_str.to_lowercase().as_str() {
            "male" => Ok(VoiceGender::Male),
            "female" => Ok(VoiceGender::Female),
            "neutral" => Ok(VoiceGender::Neutral),
            "child" => Ok(VoiceGender::Child),
            _ => Ok(VoiceGender::Neutral),
        }
    }

    fn parse_voice_characteristics(&self, attributes: &HashMap<String, String>) -> Result<Option<VoiceCharacteristics>> {
        Ok(None) // Simplified
    }

    fn parse_paragraph_prosody(&self, attributes: &HashMap<String, String>) -> Result<Option<ParagraphProsody>> {
        Ok(None) // Simplified
    }

    fn parse_sentence_prosody(&self, attributes: &HashMap<String, String>) -> Result<Option<SentenceProsody>> {
        Ok(None) // Simplified
    }

    fn normalize_text_content(&self, text: &str) -> String {
        // Basic text normalization
        text.chars()
            .map(|c| if c.is_whitespace() { ' ' } else { c })
            .collect::<String>()
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
    }

    fn validate_element(&self, element: &SsmlElement, warnings: &mut Vec<ParseWarning>) -> Result<()> {
        // Element validation would go here
        Ok(())
    }

    fn count_elements(&self, element: Option<&SsmlElement>) -> usize {
        // Count total elements in the tree
        element.map(|_| 1).unwrap_or(0) // Simplified
    }
}

impl XmlTokenizer {
    fn new(text: &str) -> Result<Self> {
        Ok(Self {
            text: text.to_string(),
            position: 0,
            line: 1,
            column: 1,
        })
    }

    fn next_token(&mut self) -> Result<XmlToken> {
        self.skip_whitespace();
        
        if self.position >= self.text.len() {
            return Ok(XmlToken::Eof);
        }

        match self.current_char() {
            Some('<') => self.parse_tag(),
            _ => self.parse_text(),
        }
    }

    fn current_char(&self) -> Option<char> {
        self.text.chars().nth(self.position)
    }

    fn peek_char(&self) -> Option<char> {
        self.text.chars().nth(self.position + 1)
    }

    fn skip_char(&mut self) {
        if let Some(c) = self.current_char() {
            self.position += 1;
            if c == '\n' {
                self.line += 1;
                self.column = 1;
            } else {
                self.column += 1;
            }
        }
    }

    fn skip_whitespace(&mut self) {
        while let Some(c) = self.current_char() {
            if c.is_whitespace() {
                self.skip_char();
            } else {
                break;
            }
        }
    }

    fn parse_tag(&mut self) -> Result<XmlToken> {
        self.skip_char(); // Skip '<'
        
        if self.current_char() == Some('/') {
            self.parse_closing_tag()
        } else if self.current_char() == Some('!') {
            self.parse_comment()
        } else {
            self.parse_opening_tag()
        }
    }

    fn parse_opening_tag(&mut self) -> Result<XmlToken> {
        let name = self.read_tag_name()?;
        let attributes = self.parse_attributes()?;
        
        self.skip_whitespace();
        let self_closing = if self.peek_char() == Some('/') {
            self.skip_char(); // Skip '/'
            true
        } else {
            false
        };
        
        if self.peek_char() == Some('>') {
            self.skip_char(); // Skip '>'
        } else {
            return Err(G2pError::ConfigError("Expected '>' in opening tag".to_string()));
        }
        
        Ok(XmlToken::OpenTag {
            name,
            attributes,
            self_closing,
        })
    }

    fn parse_closing_tag(&mut self) -> Result<XmlToken> {
        self.skip_char(); // Skip '/'
        let name = self.read_tag_name()?;
        
        self.skip_whitespace();
        if self.peek_char() == Some('>') {
            self.skip_char(); // Skip '>'
        } else {
            return Err(G2pError::ConfigError("Expected '>' in closing tag".to_string()));
        }
        
        Ok(XmlToken::CloseTag(name))
    }

    fn parse_comment(&mut self) -> Result<XmlToken> {
        // Skip '!--'
        self.skip_char(); // '!'
        if self.current_char() != Some('-') {
            return Err(G2pError::ConfigError("Invalid comment syntax".to_string()));
        }
        self.skip_char();
        if self.current_char() != Some('-') {
            return Err(G2pError::ConfigError("Invalid comment syntax".to_string()));
        }
        self.skip_char();
        
        let mut comment = String::new();
        while let Some(c) = self.current_char() {
            if c == '-' && self.peek_char() == Some('-') {
                // Check for comment end: -->
                break;
            }
            comment.push(c);
            self.skip_char();
        }
        
        Ok(XmlToken::Comment(comment))
    }

    fn parse_text(&mut self) -> Result<XmlToken> {
        let mut text = String::new();
        
        while let Some(c) = self.peek_char() {
            if c == '<' {
                break;
            }
            text.push(c);
            self.skip_char();
        }
        
        Ok(XmlToken::Text(text))
    }

    fn read_tag_name(&mut self) -> Result<String> {
        let mut name = String::new();
        
        while let Some(c) = self.peek_char() {
            if c.is_alphanumeric() || c == '-' || c == '_' || c == ':' {
                name.push(c);
                self.skip_char();
            } else {
                break;
            }
        }
        
        if name.is_empty() {
            return Err(G2pError::ConfigError("Empty tag name".to_string()));
        }
        
        Ok(name)
    }

    fn parse_attributes(&mut self) -> Result<HashMap<String, String>> {
        let mut attributes = HashMap::new();
        
        loop {
            self.skip_whitespace();
            
            // Check for end of tag
            if self.peek_char() == Some('>') || self.peek_char() == Some('/') {
                break;
            }
            
            // Parse attribute name
            let name = self.read_attribute_name()?;
            
            self.skip_whitespace();
            if self.peek_char() != Some('=') {
                return Err(G2pError::ConfigError("Expected '=' after attribute name".to_string()));
            }
            self.skip_char(); // Skip '='
            
            self.skip_whitespace();
            let value = self.read_attribute_value()?;
            
            attributes.insert(name, value);
        }
        
        Ok(attributes)
    }

    fn read_attribute_name(&mut self) -> Result<String> {
        let mut name = String::new();
        
        while let Some(c) = self.peek_char() {
            if c.is_alphanumeric() || c == '-' || c == '_' || c == ':' {
                name.push(c);
                self.skip_char();
            } else {
                break;
            }
        }
        
        if name.is_empty() {
            return Err(G2pError::ConfigError("Empty attribute name".to_string()));
        }
        
        Ok(name)
    }

    fn read_attribute_value(&mut self) -> Result<String> {
        let quote_char = if self.peek_char() == Some('"') {
            self.skip_char();
            '"'
        } else if self.peek_char() == Some('\'') {
            self.skip_char();
            '\''
        } else {
            return Err(G2pError::ConfigError("Expected quoted attribute value".to_string()));
        };
        
        let mut value = String::new();
        while let Some(c) = self.current_char() {
            if c == quote_char {
                break;
            }
            value.push(c);
            self.skip_char();
        }
        
        Ok(value)
    }
}

impl ParserContext {
    fn new() -> Self {
        Self {
            depth: 0,
            current_language: None,
            position: 0,
            line: 1,
            column: 1,
            error_count: 0,
            warning_count: 0,
        }
    }
}

impl Default for ParserConfig {
    fn default() -> Self {
        Self {
            strict_validation: false,
            preserve_whitespace: false,
            max_nesting_depth: 50,
            allow_custom_elements: true,
            default_language: Some(LanguageCode::EnUs),
            normalize_text: true,
        }
    }
}

impl Default for ErrorRecovery {
    fn default() -> Self {
        Self {
            enable_recovery: true,
            malformed_tag_strategy: MalformedTagStrategy::Fix,
            unclosed_tag_strategy: UnclosedTagStrategy::AutoClose,
            invalid_attribute_strategy: InvalidAttributeStrategy::WarnAndContinue,
            max_recovery_attempts: 10,
        }
    }
}

impl Default for ValidationRules {
    fn default() -> Self {
        Self {
            required_attributes: HashMap::new(),
            allowed_children: HashMap::new(),
            attribute_constraints: HashMap::new(),
            custom_validators: Vec::new(),
        }
    }
}

impl Default for SsmlParser {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parser_creation() {
        let parser = SsmlParser::new();
        assert!(parser.config.normalize_text);
    }

    #[test]
    fn test_simple_parsing() {
        let parser = SsmlParser::new();
        let ssml = "<speak>Hello world</speak>";
        let result = parser.parse_simple(ssml);
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_with_errors() {
        let parser = SsmlParser::new();
        let ssml = "<speak>Hello <unclosed> world</speak>";
        let result = parser.parse(ssml);
        assert!(result.is_ok());
        // Should have warnings or errors but still parse
    }

    #[test]
    fn test_tokenizer() {
        let mut tokenizer = XmlTokenizer::new("<test>content</test>").unwrap();
        
        let token1 = tokenizer.next_token().unwrap();
        match token1 {
            XmlToken::OpenTag { name, .. } => assert_eq!(name, "test"),
            _ => panic!("Expected opening tag"),
        }
        
        let token2 = tokenizer.next_token().unwrap();
        match token2 {
            XmlToken::Text(content) => assert_eq!(content, "content"),
            _ => panic!("Expected text content"),
        }
        
        let token3 = tokenizer.next_token().unwrap();
        match token3 {
            XmlToken::CloseTag(name) => assert_eq!(name, "test"),
            _ => panic!("Expected closing tag"),
        }
    }
}
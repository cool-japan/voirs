//! Internationalization (i18n) Support for VoiRS Feedback System
//!
//! This module provides comprehensive internationalization support including
//! multi-language UI, locale-specific formatting, cultural adaptation,
//! and dynamic language switching.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::RwLock;

/// Internationalization errors
#[derive(Error, Debug)]
pub enum I18nError {
    /// Language not supported
    #[error("Language '{language}' is not supported")]
    /// Description
    LanguageNotSupported {
        /// ISO code or language identifier that is not supported.
        language: String,
    },

    /// Translation key not found
    #[error("Translation key '{key}' not found for language '{language}'")]
    /// Description
    /// Description
    TranslationKeyNotFound {
        /// Lookup key that could not be resolved.
        key: String,
        /// Language in which the translation was requested.
        language: String,
    },

    /// Locale format error
    #[error("Invalid locale format: {locale}")]
    /// Description
    InvalidLocaleFormat {
        /// Locale string that failed validation.
        locale: String,
    },

    /// Resource loading error
    #[error("Failed to load language resources: {message}")]
    /// Description
    ResourceLoadingError {
        /// Details about why resource loading failed.
        message: String,
    },

    /// Pluralization error
    #[error("Pluralization error for language '{language}': {message}")]
    /// Description
    /// Description
    PluralizationError {
        /// Language where pluralization logic failed.
        language: String,
        /// Details about the pluralization error.
        message: String,
    },
}

/// Result type for i18n operations
pub type I18nResult<T> = Result<T, I18nError>;

/// Supported languages
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Language {
    /// English (US)
    English,
    /// Spanish
    Spanish,
    /// French
    French,
    /// German
    German,
    /// Italian
    Italian,
    /// Portuguese
    Portuguese,
    /// Russian
    Russian,
    /// Japanese
    Japanese,
    /// Korean
    Korean,
    /// Chinese Simplified
    ChineseSimplified,
    /// Chinese Traditional
    ChineseTraditional,
    /// Arabic
    Arabic,
    /// Hindi
    Hindi,
    /// Dutch
    Dutch,
    /// Swedish
    Swedish,
    /// Hebrew
    Hebrew,
    /// Persian/Farsi
    Persian,
    /// Urdu
    Urdu,
    /// Custom language
    Custom(String),
}

impl Language {
    /// Get language code (ISO 639-1)
    pub fn code(&self) -> &str {
        match self {
            Language::English => "en",
            Language::Spanish => "es",
            Language::French => "fr",
            Language::German => "de",
            Language::Italian => "it",
            Language::Portuguese => "pt",
            Language::Russian => "ru",
            Language::Japanese => "ja",
            Language::Korean => "ko",
            Language::ChineseSimplified => "zh-CN",
            Language::ChineseTraditional => "zh-TW",
            Language::Arabic => "ar",
            Language::Hindi => "hi",
            Language::Dutch => "nl",
            Language::Swedish => "sv",
            Language::Hebrew => "he",
            Language::Persian => "fa",
            Language::Urdu => "ur",
            Language::Custom(code) => code,
        }
    }

    /// Get language name in English
    pub fn english_name(&self) -> &str {
        match self {
            Language::English => "English",
            Language::Spanish => "Spanish",
            Language::French => "French",
            Language::German => "German",
            Language::Italian => "Italian",
            Language::Portuguese => "Portuguese",
            Language::Russian => "Russian",
            Language::Japanese => "Japanese",
            Language::Korean => "Korean",
            Language::ChineseSimplified => "Chinese (Simplified)",
            Language::ChineseTraditional => "Chinese (Traditional)",
            Language::Arabic => "Arabic",
            Language::Hindi => "Hindi",
            Language::Dutch => "Dutch",
            Language::Swedish => "Swedish",
            Language::Hebrew => "Hebrew",
            Language::Persian => "Persian",
            Language::Urdu => "Urdu",
            Language::Custom(name) => name,
        }
    }

    /// Get language name in native script
    pub fn native_name(&self) -> &str {
        match self {
            Language::English => "English",
            Language::Spanish => "Español",
            Language::French => "Français",
            Language::German => "Deutsch",
            Language::Italian => "Italiano",
            Language::Portuguese => "Português",
            Language::Russian => "Русский",
            Language::Japanese => "日本語",
            Language::Korean => "한국어",
            Language::ChineseSimplified => "简体中文",
            Language::ChineseTraditional => "繁體中文",
            Language::Arabic => "العربية",
            Language::Hindi => "हिन्दी",
            Language::Dutch => "Nederlands",
            Language::Swedish => "Svenska",
            Language::Hebrew => "עברית",
            Language::Persian => "فارسی",
            Language::Urdu => "اردو",
            Language::Custom(name) => name,
        }
    }

    /// Check if language is right-to-left
    pub fn is_rtl(&self) -> bool {
        matches!(
            self,
            Language::Arabic | Language::Hebrew | Language::Persian | Language::Urdu
        )
    }

    /// From language code
    pub fn from_code(code: &str) -> Option<Self> {
        match code.to_lowercase().as_str() {
            "en" | "en-us" => Some(Language::English),
            "es" => Some(Language::Spanish),
            "fr" => Some(Language::French),
            "de" => Some(Language::German),
            "it" => Some(Language::Italian),
            "pt" => Some(Language::Portuguese),
            "ru" => Some(Language::Russian),
            "ja" => Some(Language::Japanese),
            "ko" => Some(Language::Korean),
            "zh-cn" | "zh" => Some(Language::ChineseSimplified),
            "zh-tw" => Some(Language::ChineseTraditional),
            "ar" => Some(Language::Arabic),
            "hi" => Some(Language::Hindi),
            "nl" => Some(Language::Dutch),
            "sv" => Some(Language::Swedish),
            "he" => Some(Language::Hebrew),
            "fa" => Some(Language::Persian),
            "ur" => Some(Language::Urdu),
            _ => None,
        }
    }
}

/// Text direction
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TextDirection {
    /// Left-to-right
    Ltr,
    /// Right-to-left
    Rtl,
}

/// Locale information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Locale {
    /// Language
    pub language: Language,
    /// Country/region code (ISO 3166-1 alpha-2)
    pub country: Option<String>,
    /// Script (ISO 15924)
    pub script: Option<String>,
    /// Text direction
    pub text_direction: TextDirection,
    /// Decimal separator
    pub decimal_separator: String,
    /// Thousands separator
    pub thousands_separator: String,
    /// Currency symbol
    pub currency_symbol: String,
    /// Date format pattern
    pub date_format: String,
    /// Time format pattern
    pub time_format: String,
    /// First day of week (0 = Sunday, 1 = Monday, etc.)
    pub first_day_of_week: u8,
}

impl Default for Locale {
    fn default() -> Self {
        Self::english_us()
    }
}

impl Locale {
    /// English (US) locale
    pub fn english_us() -> Self {
        Self {
            language: Language::English,
            country: Some(String::from("US")),
            script: None,
            text_direction: TextDirection::Ltr,
            decimal_separator: String::from("."),
            thousands_separator: String::from(","),
            currency_symbol: String::from("$"),
            date_format: String::from("MM/dd/yyyy"),
            time_format: String::from("h:mm a"),
            first_day_of_week: 0, // Sunday
        }
    }

    /// French locale
    pub fn french() -> Self {
        Self {
            language: Language::French,
            country: Some(String::from("FR")),
            script: None,
            text_direction: TextDirection::Ltr,
            decimal_separator: String::from(","),
            thousands_separator: String::from(" "),
            currency_symbol: String::from("€"),
            date_format: String::from("dd/MM/yyyy"),
            time_format: String::from("HH:mm"),
            first_day_of_week: 1, // Monday
        }
    }

    /// German locale
    pub fn german() -> Self {
        Self {
            language: Language::German,
            country: Some(String::from("DE")),
            script: None,
            text_direction: TextDirection::Ltr,
            decimal_separator: String::from(","),
            thousands_separator: String::from("."),
            currency_symbol: String::from("€"),
            date_format: String::from("dd.MM.yyyy"),
            time_format: String::from("HH:mm"),
            first_day_of_week: 1, // Monday
        }
    }

    /// Japanese locale
    pub fn japanese() -> Self {
        Self {
            language: Language::Japanese,
            country: Some(String::from("JP")),
            script: None,
            text_direction: TextDirection::Ltr,
            decimal_separator: String::from("."),
            thousands_separator: String::from(","),
            currency_symbol: String::from("¥"),
            date_format: String::from("yyyy/MM/dd"),
            time_format: String::from("H:mm"),
            first_day_of_week: 0, // Sunday
        }
    }

    /// Arabic locale
    pub fn arabic() -> Self {
        Self {
            language: Language::Arabic,
            country: Some(String::from("SA")),
            script: Some(String::from("Arab")),
            text_direction: TextDirection::Rtl,
            decimal_separator: String::from("."),
            thousands_separator: String::from(","),
            currency_symbol: String::from("ر.س"),
            date_format: String::from("dd/MM/yyyy"),
            time_format: String::from("h:mm a"),
            first_day_of_week: 6, // Saturday
        }
    }

    /// Hebrew locale
    pub fn hebrew() -> Self {
        Self {
            language: Language::Hebrew,
            country: Some(String::from("IL")),
            script: Some(String::from("Hebr")),
            text_direction: TextDirection::Rtl,
            decimal_separator: String::from("."),
            thousands_separator: String::from(","),
            currency_symbol: String::from("₪"),
            date_format: String::from("dd/MM/yyyy"),
            time_format: String::from("H:mm"),
            first_day_of_week: 0, // Sunday
        }
    }

    /// Persian locale
    pub fn persian() -> Self {
        Self {
            language: Language::Persian,
            country: Some(String::from("IR")),
            script: Some(String::from("Arab")),
            text_direction: TextDirection::Rtl,
            decimal_separator: String::from("."),
            thousands_separator: String::from(","),
            currency_symbol: String::from("﷼"),
            date_format: String::from("dd/MM/yyyy"),
            time_format: String::from("H:mm"),
            first_day_of_week: 6, // Saturday
        }
    }

    /// Urdu locale
    pub fn urdu() -> Self {
        Self {
            language: Language::Urdu,
            country: Some(String::from("PK")),
            script: Some(String::from("Arab")),
            text_direction: TextDirection::Rtl,
            decimal_separator: String::from("."),
            thousands_separator: String::from(","),
            currency_symbol: String::from("₨"),
            date_format: String::from("dd/MM/yyyy"),
            time_format: String::from("h:mm a"),
            first_day_of_week: 1, // Monday
        }
    }

    /// Get locale identifier string
    pub fn identifier(&self) -> String {
        let mut parts = vec![self.language.code()];

        if let Some(script) = &self.script {
            parts.push(script);
        }

        if let Some(country) = &self.country {
            parts.push(country);
        }

        parts.join("-")
    }
}

/// Translation entry with pluralization support
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Translation {
    /// Singular form
    pub singular: String,
    /// Plural forms (for languages with complex pluralization)
    pub plural: Option<HashMap<String, String>>,
    /// Context information
    pub context: Option<String>,
    /// Description for translators
    pub description: Option<String>,
}

impl Translation {
    /// Create simple translation
    pub fn simple(text: &str) -> Self {
        Self {
            singular: text.to_string(),
            plural: None,
            context: None,
            description: None,
        }
    }

    /// Create translation with plural forms
    pub fn with_plural(singular: &str, plural: &str) -> Self {
        let mut plural_forms = HashMap::new();
        plural_forms.insert(String::from("other"), plural.to_string());

        Self {
            singular: singular.to_string(),
            plural: Some(plural_forms),
            context: None,
            description: None,
        }
    }

    /// Get appropriate form for count
    pub fn get_form(&self, count: i64, language: &Language) -> &str {
        if let Some(plural_forms) = &self.plural {
            let rule = get_plural_rule(language, count);
            plural_forms.get(&rule).unwrap_or(&self.singular)
        } else {
            &self.singular
        }
    }
}

/// Translation parameters for interpolation
pub type TranslationParams = HashMap<String, String>;

/// Language resource bundle
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageBundle {
    /// Language
    pub language: Language,
    /// Locale
    pub locale: Locale,
    /// Translations map
    pub translations: HashMap<String, Translation>,
    /// Metadata
    pub metadata: BundleMetadata,
}

/// Bundle metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BundleMetadata {
    /// Version
    pub version: String,
    /// Last updated
    pub last_updated: DateTime<Utc>,
    /// Translator(s)
    pub translators: Vec<String>,
    /// Completion percentage
    pub completion_percentage: f32,
}

/// Layout properties for UI rendering
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayoutProperties {
    /// Text direction
    pub text_direction: TextDirection,
    /// Text alignment (left/right)
    pub text_align: String,
    /// Reading direction for HTML dir attribute
    pub reading_direction: String,
    /// Whether layout is right-to-left
    pub is_rtl: bool,
    /// Start side for margins/padding (left for LTR, right for RTL)
    pub start_side: String,
    /// End side for margins/padding (right for LTR, left for RTL)
    pub end_side: String,
}

/// Internationalization manager
pub struct I18nManager {
    /// Current language
    current_language: Arc<RwLock<Language>>,
    /// Language bundles
    bundles: Arc<RwLock<HashMap<Language, LanguageBundle>>>,
    /// Fallback language
    fallback_language: Language,
}

impl I18nManager {
    /// Create new i18n manager
    pub fn new(fallback_language: Language) -> Self {
        Self {
            current_language: Arc::new(RwLock::new(fallback_language.clone())),
            bundles: Arc::new(RwLock::new(HashMap::new())),
            fallback_language,
        }
    }

    /// Load language bundle
    pub async fn load_bundle(&self, bundle: LanguageBundle) -> I18nResult<()> {
        let language = bundle.language.clone();
        self.bundles.write().await.insert(language, bundle);
        Ok(())
    }

    /// Set current language
    pub async fn set_language(&self, language: Language) -> I18nResult<()> {
        // Check if language is loaded
        if !self.bundles.read().await.contains_key(&language) {
            return Err(I18nError::LanguageNotSupported {
                language: language.code().to_string(),
            });
        }

        *self.current_language.write().await = language;
        Ok(())
    }

    /// Get current language
    pub async fn get_language(&self) -> Language {
        self.current_language.read().await.clone()
    }

    /// Get current locale
    pub async fn get_locale(&self) -> I18nResult<Locale> {
        let language = self.get_language().await;
        let bundles = self.bundles.read().await;

        if let Some(bundle) = bundles.get(&language) {
            Ok(bundle.locale.clone())
        } else {
            // Fallback to default locale
            Ok(Locale::default())
        }
    }

    /// Translate text
    pub async fn translate(&self, key: &str, params: Option<TranslationParams>) -> String {
        self.translate_with_count(key, None, params).await
    }

    /// Translate text with pluralization
    pub async fn translate_with_count(
        &self,
        key: &str,
        count: Option<i64>,
        params: Option<TranslationParams>,
    ) -> String {
        let language = self.get_language().await;
        let bundles = self.bundles.read().await;

        // Try current language first
        if let Some(bundle) = bundles.get(&language) {
            if let Some(translation) = bundle.translations.get(key) {
                let text = if let Some(count) = count {
                    translation.get_form(count, &language)
                } else {
                    &translation.singular
                };
                return self.interpolate(text, params);
            }
        }

        // Fallback to fallback language
        if language != self.fallback_language {
            if let Some(bundle) = bundles.get(&self.fallback_language) {
                if let Some(translation) = bundle.translations.get(key) {
                    let text = if let Some(count) = count {
                        translation.get_form(count, &self.fallback_language)
                    } else {
                        &translation.singular
                    };
                    return self.interpolate(text, params);
                }
            }
        }

        // Return key if no translation found
        key.to_string()
    }

    /// Interpolate parameters in text
    fn interpolate(&self, text: &str, params: Option<TranslationParams>) -> String {
        if let Some(params) = params {
            let mut result = text.to_string();
            for (key, value) in params {
                result = result.replace(&format!("{{{}}}", key), &value);
            }
            result
        } else {
            text.to_string()
        }
    }

    /// Get available languages
    pub async fn get_available_languages(&self) -> Vec<Language> {
        self.bundles.read().await.keys().cloned().collect()
    }

    /// Format number according to locale
    pub async fn format_number(&self, number: f64) -> I18nResult<String> {
        let locale = self.get_locale().await?;

        // Simple formatting - in production would use proper locale formatting
        let formatted = if number.fract() == 0.0 {
            format!("{}", number as i64)
        } else {
            format!("{:.2}", number)
        };

        // Replace decimal separator
        let formatted = formatted.replace('.', &locale.decimal_separator);

        Ok(formatted)
    }

    /// Format currency according to locale
    pub async fn format_currency(&self, amount: f64) -> I18nResult<String> {
        let locale = self.get_locale().await?;
        let formatted_number = self.format_number(amount).await?;

        Ok(format!("{}{}", locale.currency_symbol, formatted_number))
    }

    /// Format date according to locale
    pub async fn format_date(&self, date: DateTime<Utc>) -> I18nResult<String> {
        let locale = self.get_locale().await?;

        // Simple date formatting - in production would use proper locale formatting
        match locale.date_format.as_str() {
            "MM/dd/yyyy" => Ok(date.format("%m/%d/%Y").to_string()),
            "dd/MM/yyyy" => Ok(date.format("%d/%m/%Y").to_string()),
            "dd.MM.yyyy" => Ok(date.format("%d.%m.%Y").to_string()),
            "yyyy/MM/dd" => Ok(date.format("%Y/%m/%d").to_string()),
            _ => Ok(date.format("%Y-%m-%d").to_string()),
        }
    }

    /// Get text direction for current language
    pub async fn get_text_direction(&self) -> TextDirection {
        match self.get_locale().await {
            Ok(locale) => locale.text_direction,
            Err(_) => TextDirection::Ltr,
        }
    }

    /// Initialize with default bundles
    pub async fn initialize_default_bundles(&self) -> I18nResult<()> {
        // English bundle
        let english_bundle = create_english_bundle();
        self.load_bundle(english_bundle).await?;

        // Spanish bundle
        let spanish_bundle = create_spanish_bundle();
        self.load_bundle(spanish_bundle).await?;

        // French bundle
        let french_bundle = create_french_bundle();
        self.load_bundle(french_bundle).await?;

        // Arabic bundle
        let arabic_bundle = create_arabic_bundle();
        self.load_bundle(arabic_bundle).await?;

        // Hebrew bundle
        let hebrew_bundle = create_hebrew_bundle();
        self.load_bundle(hebrew_bundle).await?;

        Ok(())
    }

    /// Get text alignment for current language
    pub async fn get_text_alignment(&self) -> String {
        match self.get_text_direction().await {
            TextDirection::Rtl => String::from("right"),
            TextDirection::Ltr => String::from("left"),
        }
    }

    /// Get reading direction for current language
    pub async fn get_reading_direction(&self) -> String {
        match self.get_text_direction().await {
            TextDirection::Rtl => String::from("rtl"),
            TextDirection::Ltr => String::from("ltr"),
        }
    }

    /// Check if current language requires RTL layout
    pub async fn requires_rtl_layout(&self) -> bool {
        matches!(self.get_text_direction().await, TextDirection::Rtl)
    }

    /// Apply text direction to CSS class string
    pub async fn apply_direction_class(&self, base_class: &str) -> String {
        if self.requires_rtl_layout().await {
            format!("{} rtl", base_class)
        } else {
            format!("{} ltr", base_class)
        }
    }

    /// Get layout properties for UI rendering
    pub async fn get_layout_properties(&self) -> LayoutProperties {
        let is_rtl = self.requires_rtl_layout().await;
        LayoutProperties {
            text_direction: self.get_text_direction().await,
            text_align: self.get_text_alignment().await,
            reading_direction: self.get_reading_direction().await,
            is_rtl,
            start_side: if is_rtl {
                String::from("right")
            } else {
                String::from("left")
            },
            end_side: if is_rtl {
                String::from("left")
            } else {
                String::from("right")
            },
        }
    }
}

/// Get plural rule for language and count
fn get_plural_rule(language: &Language, count: i64) -> String {
    match language {
        Language::English => {
            if count == 1 {
                "one"
            } else {
                "other"
            }
        }
        Language::Spanish | Language::French | Language::Italian | Language::Portuguese => {
            if count == 1 {
                "one"
            } else {
                "other"
            }
        }
        Language::German | Language::Dutch | Language::Swedish => {
            if count == 1 {
                "one"
            } else {
                "other"
            }
        }
        Language::Russian => {
            let n = count % 100;
            if n % 10 == 1 && n % 100 != 11 {
                "one"
            } else if (2..=4).contains(&(n % 10)) && !(12..=14).contains(&(n % 100)) {
                "few"
            } else {
                "many"
            }
        }
        Language::Japanese
        | Language::Korean
        | Language::ChineseSimplified
        | Language::ChineseTraditional => {
            "other" // No plural forms
        }
        Language::Arabic => match count {
            0 => "zero",
            1 => "one",
            2 => "two",
            n if (3..=10).contains(&(n % 100)) => "few",
            n if (11..=99).contains(&(n % 100)) => "many",
            _ => "other",
        },
        _ => {
            if count == 1 {
                "one"
            } else {
                "other"
            }
        }
    }
    .to_string()
}

/// Create English language bundle
fn create_english_bundle() -> LanguageBundle {
    let mut translations = HashMap::new();

    // Common UI translations
    translations.insert(String::from("welcome"), Translation::simple("Welcome"));
    translations.insert(String::from("feedback"), Translation::simple("Feedback"));
    translations.insert(String::from("progress"), Translation::simple("Progress"));
    translations.insert(String::from("settings"), Translation::simple("Settings"));
    translations.insert(String::from("profile"), Translation::simple("Profile"));
    translations.insert(String::from("logout"), Translation::simple("Logout"));
    translations.insert(String::from("save"), Translation::simple("Save"));
    translations.insert(String::from("cancel"), Translation::simple("Cancel"));
    translations.insert(String::from("delete"), Translation::simple("Delete"));
    translations.insert(String::from("edit"), Translation::simple("Edit"));
    translations.insert(String::from("loading"), Translation::simple("Loading..."));
    translations.insert(String::from("error"), Translation::simple("Error"));
    translations.insert(String::from("success"), Translation::simple("Success"));

    // Feedback specific
    translations.insert(
        String::from("pronunciation_feedback"),
        Translation::simple("Pronunciation Feedback"),
    );
    translations.insert(String::from("score"), Translation::simple("Score"));
    translations.insert(
        String::from("improvement_suggestion"),
        Translation::simple("Improvement Suggestion"),
    );

    // Plurals
    translations.insert(
        String::from("items_count"),
        Translation::with_plural("{count} item", "{count} items"),
    );

    LanguageBundle {
        language: Language::English,
        locale: Locale::english_us(),
        translations,
        metadata: BundleMetadata {
            version: String::from("1.0.0"),
            last_updated: Utc::now(),
            translators: vec![String::from("System")],
            completion_percentage: 100.0,
        },
    }
}

/// Create Spanish language bundle
fn create_spanish_bundle() -> LanguageBundle {
    let mut translations = HashMap::new();

    translations.insert(String::from("welcome"), Translation::simple("Bienvenido"));
    translations.insert(
        String::from("feedback"),
        Translation::simple("Retroalimentación"),
    );
    translations.insert(String::from("progress"), Translation::simple("Progreso"));
    translations.insert(
        String::from("settings"),
        Translation::simple("Configuración"),
    );
    translations.insert(String::from("profile"), Translation::simple("Perfil"));
    translations.insert(String::from("logout"), Translation::simple("Cerrar sesión"));
    translations.insert(String::from("save"), Translation::simple("Guardar"));
    translations.insert(String::from("cancel"), Translation::simple("Cancelar"));
    translations.insert(String::from("delete"), Translation::simple("Eliminar"));
    translations.insert(String::from("edit"), Translation::simple("Editar"));
    translations.insert(String::from("loading"), Translation::simple("Cargando..."));
    translations.insert(String::from("error"), Translation::simple("Error"));
    translations.insert(String::from("success"), Translation::simple("Éxito"));

    translations.insert(
        String::from("pronunciation_feedback"),
        Translation::simple("Retroalimentación de Pronunciación"),
    );
    translations.insert(String::from("score"), Translation::simple("Puntuación"));
    translations.insert(
        String::from("improvement_suggestion"),
        Translation::simple("Sugerencia de Mejora"),
    );

    translations.insert(
        String::from("items_count"),
        Translation::with_plural("{count} elemento", "{count} elementos"),
    );

    LanguageBundle {
        language: Language::Spanish,
        locale: Locale {
            language: Language::Spanish,
            country: Some(String::from("ES")),
            script: None,
            text_direction: TextDirection::Ltr,
            decimal_separator: String::from(","),
            thousands_separator: String::from("."),
            currency_symbol: String::from("€"),
            date_format: String::from("dd/MM/yyyy"),
            time_format: String::from("HH:mm"),
            first_day_of_week: 1,
        },
        translations,
        metadata: BundleMetadata {
            version: String::from("1.0.0"),
            last_updated: Utc::now(),
            translators: vec![String::from("System")],
            completion_percentage: 100.0,
        },
    }
}

/// Create French language bundle
fn create_french_bundle() -> LanguageBundle {
    let mut translations = HashMap::new();

    translations.insert(String::from("welcome"), Translation::simple("Bienvenue"));
    translations.insert(
        String::from("feedback"),
        Translation::simple("Commentaires"),
    );
    translations.insert(String::from("progress"), Translation::simple("Progrès"));
    translations.insert(String::from("settings"), Translation::simple("Paramètres"));
    translations.insert(String::from("profile"), Translation::simple("Profil"));
    translations.insert(String::from("logout"), Translation::simple("Déconnexion"));
    translations.insert(String::from("save"), Translation::simple("Enregistrer"));
    translations.insert(String::from("cancel"), Translation::simple("Annuler"));
    translations.insert(String::from("delete"), Translation::simple("Supprimer"));
    translations.insert(String::from("edit"), Translation::simple("Modifier"));
    translations.insert(
        String::from("loading"),
        Translation::simple("Chargement..."),
    );
    translations.insert(String::from("error"), Translation::simple("Erreur"));
    translations.insert(String::from("success"), Translation::simple("Succès"));

    translations.insert(
        String::from("pronunciation_feedback"),
        Translation::simple("Commentaires de Prononciation"),
    );
    translations.insert(String::from("score"), Translation::simple("Score"));
    translations.insert(
        String::from("improvement_suggestion"),
        Translation::simple("Suggestion d'Amélioration"),
    );

    translations.insert(
        String::from("items_count"),
        Translation::with_plural("{count} élément", "{count} éléments"),
    );

    LanguageBundle {
        language: Language::French,
        locale: Locale::french(),
        translations,
        metadata: BundleMetadata {
            version: String::from("1.0.0"),
            last_updated: Utc::now(),
            translators: vec![String::from("System")],
            completion_percentage: 100.0,
        },
    }
}

/// Create Arabic language bundle
fn create_arabic_bundle() -> LanguageBundle {
    let mut translations = HashMap::new();

    translations.insert(String::from("welcome"), Translation::simple("مرحباً"));
    translations.insert(String::from("feedback"), Translation::simple("تعليقات"));
    translations.insert(String::from("progress"), Translation::simple("التقدم"));
    translations.insert(String::from("settings"), Translation::simple("الإعدادات"));
    translations.insert(String::from("profile"), Translation::simple("الملف الشخصي"));
    translations.insert(String::from("logout"), Translation::simple("تسجيل الخروج"));
    translations.insert(String::from("save"), Translation::simple("حفظ"));
    translations.insert(String::from("cancel"), Translation::simple("إلغاء"));
    translations.insert(String::from("delete"), Translation::simple("حذف"));
    translations.insert(String::from("edit"), Translation::simple("تعديل"));
    translations.insert("loading".to_string(), Translation::simple("جارٍ التحميل..."));
    translations.insert("error".to_string(), Translation::simple("خطأ"));
    translations.insert("success".to_string(), Translation::simple("نجح"));

    translations.insert(
        String::from("pronunciation_feedback"),
        Translation::simple("تعليقات النطق"),
    );
    translations.insert(String::from("score"), Translation::simple("النتيجة"));
    translations.insert(
        String::from("improvement_suggestion"),
        Translation::simple("اقتراح للتحسين"),
    );

    translations.insert(
        String::from("items_count"),
        Translation::with_plural("عنصر {count}", "{count} عناصر"),
    );

    LanguageBundle {
        language: Language::Arabic,
        locale: Locale::arabic(),
        translations,
        metadata: BundleMetadata {
            version: String::from("1.0.0"),
            last_updated: Utc::now(),
            translators: vec![String::from("System")],
            completion_percentage: 100.0,
        },
    }
}

/// Create Hebrew language bundle
fn create_hebrew_bundle() -> LanguageBundle {
    let mut translations = HashMap::new();

    translations.insert(String::from("welcome"), Translation::simple("ברוכים הבאים"));
    translations.insert(String::from("feedback"), Translation::simple("משוב"));
    translations.insert(String::from("progress"), Translation::simple("התקדמות"));
    translations.insert(String::from("settings"), Translation::simple("הגדרות"));
    translations.insert(String::from("profile"), Translation::simple("פרופיל"));
    translations.insert(String::from("logout"), Translation::simple("התנתקות"));
    translations.insert(String::from("save"), Translation::simple("שמור"));
    translations.insert(String::from("cancel"), Translation::simple("ביטול"));
    translations.insert(String::from("delete"), Translation::simple("מחק"));
    translations.insert(String::from("edit"), Translation::simple("ערוך"));
    translations.insert(String::from("loading"), Translation::simple("טוען..."));
    translations.insert(String::from("error"), Translation::simple("שגיאה"));
    translations.insert(String::from("success"), Translation::simple("הצלחה"));

    translations.insert(
        String::from("pronunciation_feedback"),
        Translation::simple("משוב הגייה"),
    );
    translations.insert(String::from("score"), Translation::simple("ציון"));
    translations.insert(
        String::from("improvement_suggestion"),
        Translation::simple("הצעה לשיפור"),
    );

    translations.insert(
        String::from("items_count"),
        Translation::with_plural("פריט {count}", "{count} פריטים"),
    );

    LanguageBundle {
        language: Language::Hebrew,
        locale: Locale::hebrew(),
        translations,
        metadata: BundleMetadata {
            version: String::from("1.0.0"),
            last_updated: Utc::now(),
            translators: vec![String::from("System")],
            completion_percentage: 100.0,
        },
    }
}

/// Convenience macro for translation
#[macro_export]
macro_rules! t {
    ($manager:expr, $key:literal) => {
        $manager.translate($key, None).await
    };
    ($manager:expr, $key:literal, $($param_key:literal => $param_value:expr),+) => {{
        let mut params = std::collections::HashMap::new();
        $(
            params.insert($param_key.to_string(), $param_value.to_string());
        )+
        $manager.translate($key, Some(params)).await
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_i18n_manager_creation() {
        let manager = I18nManager::new(Language::English);
        assert_eq!(manager.get_language().await, Language::English);
    }

    #[tokio::test]
    async fn test_language_bundle_loading() {
        let manager = I18nManager::new(Language::English);
        let bundle = create_english_bundle();

        assert!(manager.load_bundle(bundle).await.is_ok());
        assert!(manager
            .get_available_languages()
            .await
            .contains(&Language::English));
    }

    #[tokio::test]
    async fn test_translation() {
        let manager = I18nManager::new(Language::English);
        manager.initialize_default_bundles().await.unwrap();

        let translation = manager.translate("welcome", None).await;
        assert_eq!(translation, "Welcome");

        // Test non-existent key
        let missing = manager.translate("missing_key", None).await;
        assert_eq!(missing, "missing_key");
    }

    #[tokio::test]
    async fn test_parameter_interpolation() {
        let manager = I18nManager::new(Language::English);
        manager.initialize_default_bundles().await.unwrap();

        let mut params = HashMap::new();
        params.insert(String::from("count"), String::from("5"));

        let translation = manager
            .translate_with_count("items_count", Some(5), Some(params))
            .await;
        assert_eq!(translation, "5 items");
    }

    #[tokio::test]
    async fn test_language_switching() {
        let manager = I18nManager::new(Language::English);
        manager.initialize_default_bundles().await.unwrap();

        // Test English
        let welcome_en = manager.translate("welcome", None).await;
        assert_eq!(welcome_en, "Welcome");

        // Switch to Spanish
        manager.set_language(Language::Spanish).await.unwrap();
        let welcome_es = manager.translate("welcome", None).await;
        assert_eq!(welcome_es, "Bienvenido");

        // Switch to French
        manager.set_language(Language::French).await.unwrap();
        let welcome_fr = manager.translate("welcome", None).await;
        assert_eq!(welcome_fr, "Bienvenue");
    }

    #[tokio::test]
    async fn test_pluralization() {
        let manager = I18nManager::new(Language::English);
        manager.initialize_default_bundles().await.unwrap();

        // Test singular (1 item)
        let singular = manager
            .translate_with_count(
                "items_count",
                Some(1),
                Some({
                    let mut params = HashMap::new();
                    params.insert(String::from("count"), String::from("1"));
                    params
                }),
            )
            .await;
        assert_eq!(singular, "1 item");

        // Test plural (5 items)
        let plural = manager
            .translate_with_count(
                "items_count",
                Some(5),
                Some({
                    let mut params = HashMap::new();
                    params.insert(String::from("count"), String::from("5"));
                    params
                }),
            )
            .await;
        assert_eq!(plural, "5 items");
    }

    #[test]
    fn test_language_properties() {
        assert_eq!(Language::English.code(), "en");
        assert_eq!(Language::Spanish.english_name(), "Spanish");
        assert_eq!(Language::French.native_name(), "Français");
        assert!(Language::Arabic.is_rtl());
        assert!(Language::Hebrew.is_rtl());
        assert!(Language::Persian.is_rtl());
        assert!(Language::Urdu.is_rtl());
        assert!(!Language::English.is_rtl());
    }

    #[test]
    fn test_language_from_code() {
        assert_eq!(Language::from_code("en"), Some(Language::English));
        assert_eq!(Language::from_code("es"), Some(Language::Spanish));
        assert_eq!(Language::from_code("ar"), Some(Language::Arabic));
        assert_eq!(Language::from_code("he"), Some(Language::Hebrew));
        assert_eq!(Language::from_code("fa"), Some(Language::Persian));
        assert_eq!(Language::from_code("ur"), Some(Language::Urdu));
        assert_eq!(Language::from_code("unknown"), None);
    }

    #[test]
    fn test_plural_rules() {
        // English: 1 = one, others = other
        assert_eq!(get_plural_rule(&Language::English, 1), "one");
        assert_eq!(get_plural_rule(&Language::English, 5), "other");

        // Japanese: always other
        assert_eq!(get_plural_rule(&Language::Japanese, 1), "other");
        assert_eq!(get_plural_rule(&Language::Japanese, 5), "other");
    }

    #[tokio::test]
    async fn test_locale_formatting() {
        let manager = I18nManager::new(Language::English);
        manager.initialize_default_bundles().await.unwrap();

        // Test number formatting
        let formatted_number = manager.format_number(1234.56).await.unwrap();
        assert_eq!(formatted_number, "1234.56");

        // Test currency formatting
        let formatted_currency = manager.format_currency(99.99).await.unwrap();
        assert_eq!(formatted_currency, "$99.99");
    }

    #[tokio::test]
    async fn test_rtl_layout_properties() {
        let manager = I18nManager::new(Language::English);
        manager.initialize_default_bundles().await.unwrap();

        // Test LTR layout
        let ltr_properties = manager.get_layout_properties().await;
        assert!(!ltr_properties.is_rtl);
        assert_eq!(ltr_properties.text_align, "left");
        assert_eq!(ltr_properties.reading_direction, "ltr");
        assert_eq!(ltr_properties.start_side, "left");
        assert_eq!(ltr_properties.end_side, "right");

        // Switch to Arabic (RTL)
        manager.set_language(Language::Arabic).await.unwrap();
        let rtl_properties = manager.get_layout_properties().await;
        assert!(rtl_properties.is_rtl);
        assert_eq!(rtl_properties.text_align, "right");
        assert_eq!(rtl_properties.reading_direction, "rtl");
        assert_eq!(rtl_properties.start_side, "right");
        assert_eq!(rtl_properties.end_side, "left");
    }

    #[tokio::test]
    async fn test_rtl_translations() {
        let manager = I18nManager::new(Language::English);
        manager.initialize_default_bundles().await.unwrap();

        // Test Arabic translations
        manager.set_language(Language::Arabic).await.unwrap();
        let welcome_ar = manager.translate("welcome", None).await;
        assert_eq!(welcome_ar, "مرحباً");

        let feedback_ar = manager.translate("feedback", None).await;
        assert_eq!(feedback_ar, "تعليقات");

        // Test Hebrew translations
        manager.set_language(Language::Hebrew).await.unwrap();
        let welcome_he = manager.translate("welcome", None).await;
        assert_eq!(welcome_he, "ברוכים הבאים");

        let feedback_he = manager.translate("feedback", None).await;
        assert_eq!(feedback_he, "משוב");
    }

    #[tokio::test]
    async fn test_css_direction_classes() {
        let manager = I18nManager::new(Language::English);
        manager.initialize_default_bundles().await.unwrap();

        // Test LTR class
        let ltr_class = manager.apply_direction_class("menu").await;
        assert_eq!(ltr_class, "menu ltr");

        // Switch to Arabic
        manager.set_language(Language::Arabic).await.unwrap();
        let rtl_class = manager.apply_direction_class("menu").await;
        assert_eq!(rtl_class, "menu rtl");
    }

    #[test]
    fn test_rtl_locales() {
        let arabic_locale = Locale::arabic();
        assert_eq!(arabic_locale.text_direction, TextDirection::Rtl);
        assert_eq!(arabic_locale.currency_symbol, "ر.س");
        assert_eq!(arabic_locale.first_day_of_week, 6);

        let hebrew_locale = Locale::hebrew();
        assert_eq!(hebrew_locale.text_direction, TextDirection::Rtl);
        assert_eq!(hebrew_locale.currency_symbol, "₪");
        assert_eq!(hebrew_locale.first_day_of_week, 0);

        let persian_locale = Locale::persian();
        assert_eq!(persian_locale.text_direction, TextDirection::Rtl);
        assert_eq!(persian_locale.currency_symbol, "﷼");

        let urdu_locale = Locale::urdu();
        assert_eq!(urdu_locale.text_direction, TextDirection::Rtl);
        assert_eq!(urdu_locale.currency_symbol, "₨");
    }
}

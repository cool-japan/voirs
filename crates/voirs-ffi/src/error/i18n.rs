//! Internationalization and Localization Support
//!
//! This module provides multi-language error messages, locale detection,
//! message formatting, and cultural adaptation for VoiRS FFI.

use once_cell::sync::Lazy;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

use super::structured::{VoirsErrorCategory, VoirsErrorSubcode};

/// Supported locales
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Locale {
    /// English (US)
    EnUs,
    /// English (UK)
    EnGb,
    /// Japanese
    Ja,
    /// Spanish
    Es,
    /// French
    Fr,
    /// German
    De,
    /// Chinese (Simplified)
    ZhCn,
    /// Chinese (Traditional)
    ZhTw,
    /// Korean
    Ko,
    /// Russian
    Ru,
    /// Portuguese
    Pt,
    /// Italian
    It,
    /// Dutch
    Nl,
    /// Arabic
    Ar,
}

impl Locale {
    /// Get locale from language code string
    pub fn from_code(code: &str) -> Option<Self> {
        match code.to_lowercase().as_str() {
            "en-us" | "en_us" | "en" => Some(Self::EnUs),
            "en-gb" | "en_gb" => Some(Self::EnGb),
            "ja" | "ja-jp" | "ja_jp" => Some(Self::Ja),
            "es" | "es-es" | "es_es" => Some(Self::Es),
            "fr" | "fr-fr" | "fr_fr" => Some(Self::Fr),
            "de" | "de-de" | "de_de" => Some(Self::De),
            "zh-cn" | "zh_cn" | "zh" => Some(Self::ZhCn),
            "zh-tw" | "zh_tw" => Some(Self::ZhTw),
            "ko" | "ko-kr" | "ko_kr" => Some(Self::Ko),
            "ru" | "ru-ru" | "ru_ru" => Some(Self::Ru),
            "pt" | "pt-br" | "pt_br" => Some(Self::Pt),
            "it" | "it-it" | "it_it" => Some(Self::It),
            "nl" | "nl-nl" | "nl_nl" => Some(Self::Nl),
            "ar" | "ar-sa" | "ar_sa" => Some(Self::Ar),
            _ => None,
        }
    }

    /// Get language code string
    pub fn to_code(&self) -> &'static str {
        match self {
            Self::EnUs => "en-US",
            Self::EnGb => "en-GB",
            Self::Ja => "ja-JP",
            Self::Es => "es-ES",
            Self::Fr => "fr-FR",
            Self::De => "de-DE",
            Self::ZhCn => "zh-CN",
            Self::ZhTw => "zh-TW",
            Self::Ko => "ko-KR",
            Self::Ru => "ru-RU",
            Self::Pt => "pt-BR",
            Self::It => "it-IT",
            Self::Nl => "nl-NL",
            Self::Ar => "ar-SA",
        }
    }

    /// Get language name in English
    pub fn to_english_name(&self) -> &'static str {
        match self {
            Self::EnUs => "English (US)",
            Self::EnGb => "English (UK)",
            Self::Ja => "Japanese",
            Self::Es => "Spanish",
            Self::Fr => "French",
            Self::De => "German",
            Self::ZhCn => "Chinese (Simplified)",
            Self::ZhTw => "Chinese (Traditional)",
            Self::Ko => "Korean",
            Self::Ru => "Russian",
            Self::Pt => "Portuguese",
            Self::It => "Italian",
            Self::Nl => "Dutch",
            Self::Ar => "Arabic",
        }
    }

    /// Get language name in native script
    pub fn to_native_name(&self) -> &'static str {
        match self {
            Self::EnUs => "English (US)",
            Self::EnGb => "English (UK)",
            Self::Ja => "日本語",
            Self::Es => "Español",
            Self::Fr => "Français",
            Self::De => "Deutsch",
            Self::ZhCn => "简体中文",
            Self::ZhTw => "繁體中文",
            Self::Ko => "한국어",
            Self::Ru => "Русский",
            Self::Pt => "Português",
            Self::It => "Italiano",
            Self::Nl => "Nederlands",
            Self::Ar => "العربية",
        }
    }

    /// Get text direction for the locale
    pub fn text_direction(&self) -> TextDirection {
        match self {
            Self::Ar => TextDirection::RightToLeft,
            _ => TextDirection::LeftToRight,
        }
    }

    /// Get number formatting for the locale
    pub fn number_format(&self) -> NumberFormat {
        match self {
            Self::EnUs => NumberFormat {
                decimal_separator: ".".to_string(),
                thousands_separator: ",".to_string(),
            },
            Self::EnGb => NumberFormat {
                decimal_separator: ".".to_string(),
                thousands_separator: ",".to_string(),
            },
            Self::Es | Self::Fr | Self::It | Self::Pt => NumberFormat {
                decimal_separator: ",".to_string(),
                thousands_separator: ".".to_string(),
            },
            Self::De | Self::Nl => NumberFormat {
                decimal_separator: ",".to_string(),
                thousands_separator: ".".to_string(),
            },
            _ => NumberFormat {
                decimal_separator: ".".to_string(),
                thousands_separator: ",".to_string(),
            },
        }
    }
}

/// Text direction for locales
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextDirection {
    LeftToRight,
    RightToLeft,
}

/// Number formatting for locales
#[derive(Debug, Clone)]
pub struct NumberFormat {
    pub decimal_separator: String,
    pub thousands_separator: String,
}

/// Message template with placeholders
#[derive(Debug, Clone)]
pub struct MessageTemplate {
    pub template: String,
    pub placeholders: Vec<String>,
}

impl MessageTemplate {
    pub fn new(template: &str) -> Self {
        let placeholders = Self::extract_placeholders(template);
        Self {
            template: template.to_string(),
            placeholders,
        }
    }

    fn extract_placeholders(template: &str) -> Vec<String> {
        let mut placeholders = Vec::new();
        let mut in_placeholder = false;
        let mut current_placeholder = String::new();

        for ch in template.chars() {
            match ch {
                '{' => {
                    in_placeholder = true;
                    current_placeholder.clear();
                }
                '}' => {
                    if in_placeholder {
                        placeholders.push(current_placeholder.clone());
                        in_placeholder = false;
                    }
                }
                _ => {
                    if in_placeholder {
                        current_placeholder.push(ch);
                    }
                }
            }
        }

        placeholders
    }

    pub fn format(&self, values: &HashMap<String, String>) -> String {
        let mut result = self.template.clone();

        for (key, value) in values {
            let placeholder = format!("{{{}}}", key);
            result = result.replace(&placeholder, value);
        }

        result
    }
}

/// Internationalization manager
pub struct I18nManager {
    current_locale: RwLock<Locale>,
    messages: RwLock<HashMap<(VoirsErrorCategory, VoirsErrorSubcode, Locale), MessageTemplate>>,
    fallback_locale: Locale,
}

impl I18nManager {
    pub fn new() -> Self {
        let mut manager = Self {
            current_locale: RwLock::new(Self::detect_system_locale()),
            messages: RwLock::new(HashMap::new()),
            fallback_locale: Locale::EnUs,
        };

        manager.load_default_messages();
        manager
    }

    /// Detect system locale
    fn detect_system_locale() -> Locale {
        // Try environment variables
        for env_var in &["LANG", "LANGUAGE", "LC_ALL", "LC_MESSAGES"] {
            if let Ok(lang) = std::env::var(env_var) {
                if let Some(locale) = Locale::from_code(&lang) {
                    return locale;
                }
            }
        }

        // Platform-specific detection
        #[cfg(target_os = "windows")]
        {
            // Use Windows API to get user locale
            if let Some(locale) = Self::detect_windows_locale() {
                return locale;
            }
        }

        #[cfg(target_os = "macos")]
        {
            // Use macOS APIs to get user locale
            if let Some(locale) = Self::detect_macos_locale() {
                return locale;
            }
        }

        // Default fallback
        Locale::EnUs
    }

    #[cfg(target_os = "windows")]
    fn detect_windows_locale() -> Option<Locale> {
        // This would use Windows API calls to get the user's locale
        // For now, return None to use the fallback
        None
    }

    #[cfg(target_os = "macos")]
    fn detect_macos_locale() -> Option<Locale> {
        // This would use Core Foundation to get the user's locale
        // For now, return None to use the fallback
        None
    }

    /// Load default error messages for all locales
    fn load_default_messages(&mut self) {
        self.add_message_templates();
    }

    fn add_message_templates(&mut self) {
        let mut messages = self.messages.write();

        // Network timeout messages
        messages.insert(
            (
                VoirsErrorCategory::Network,
                VoirsErrorSubcode::NetworkTimeout,
                Locale::EnUs,
            ),
            MessageTemplate::new(
                "Network timeout after {duration} seconds. Please check your connection.",
            ),
        );
        messages.insert(
            (
                VoirsErrorCategory::Network,
                VoirsErrorSubcode::NetworkTimeout,
                Locale::Ja,
            ),
            MessageTemplate::new(
                "{duration}秒後にネットワークタイムアウトが発生しました。接続を確認してください。",
            ),
        );
        messages.insert(
            (VoirsErrorCategory::Network, VoirsErrorSubcode::NetworkTimeout, Locale::Es),
            MessageTemplate::new("Tiempo de espera de red agotado después de {duration} segundos. Verifique su conexión."),
        );
        messages.insert(
            (VoirsErrorCategory::Network, VoirsErrorSubcode::NetworkTimeout, Locale::Fr),
            MessageTemplate::new("Délai d'attente réseau dépassé après {duration} secondes. Vérifiez votre connexion."),
        );
        messages.insert(
            (
                VoirsErrorCategory::Network,
                VoirsErrorSubcode::NetworkTimeout,
                Locale::De,
            ),
            MessageTemplate::new(
                "Netzwerk-Timeout nach {duration} Sekunden. Bitte überprüfen Sie Ihre Verbindung.",
            ),
        );

        // Out of memory messages
        messages.insert(
            (
                VoirsErrorCategory::Resource,
                VoirsErrorSubcode::OutOfMemory,
                Locale::EnUs,
            ),
            MessageTemplate::new(
                "Insufficient memory available. Required: {required}MB, Available: {available}MB.",
            ),
        );
        messages.insert(
            (
                VoirsErrorCategory::Resource,
                VoirsErrorSubcode::OutOfMemory,
                Locale::Ja,
            ),
            MessageTemplate::new(
                "メモリが不足しています。必要: {required}MB、利用可能: {available}MB。",
            ),
        );
        messages.insert(
            (VoirsErrorCategory::Resource, VoirsErrorSubcode::OutOfMemory, Locale::Es),
            MessageTemplate::new("Memoria insuficiente disponible. Requerida: {required}MB, Disponible: {available}MB."),
        );
        messages.insert(
            (
                VoirsErrorCategory::Resource,
                VoirsErrorSubcode::OutOfMemory,
                Locale::Fr,
            ),
            MessageTemplate::new(
                "Mémoire insuffisante disponible. Requis: {required}MB, Disponible: {available}MB.",
            ),
        );
        messages.insert(
            (VoirsErrorCategory::Resource, VoirsErrorSubcode::OutOfMemory, Locale::De),
            MessageTemplate::new("Unzureichender Speicher verfügbar. Erforderlich: {required}MB, Verfügbar: {available}MB."),
        );

        // File not found messages
        messages.insert(
            (
                VoirsErrorCategory::Resource,
                VoirsErrorSubcode::FileNotFound,
                Locale::EnUs,
            ),
            MessageTemplate::new("File not found: {filename}. Please check the file path."),
        );
        messages.insert(
            (
                VoirsErrorCategory::Resource,
                VoirsErrorSubcode::FileNotFound,
                Locale::Ja,
            ),
            MessageTemplate::new(
                "ファイルが見つかりません: {filename}。ファイルパスを確認してください。",
            ),
        );
        messages.insert(
            (
                VoirsErrorCategory::Resource,
                VoirsErrorSubcode::FileNotFound,
                Locale::Es,
            ),
            MessageTemplate::new(
                "Archivo no encontrado: {filename}. Verifique la ruta del archivo.",
            ),
        );
        messages.insert(
            (
                VoirsErrorCategory::Resource,
                VoirsErrorSubcode::FileNotFound,
                Locale::Fr,
            ),
            MessageTemplate::new("Fichier introuvable: {filename}. Vérifiez le chemin du fichier."),
        );
        messages.insert(
            (
                VoirsErrorCategory::Resource,
                VoirsErrorSubcode::FileNotFound,
                Locale::De,
            ),
            MessageTemplate::new("Datei nicht gefunden: {filename}. Überprüfen Sie den Dateipfad."),
        );

        // Authentication failed messages
        messages.insert(
            (
                VoirsErrorCategory::Security,
                VoirsErrorSubcode::AuthenticationFailed,
                Locale::EnUs,
            ),
            MessageTemplate::new(
                "Authentication failed for user '{username}'. Please check your credentials.",
            ),
        );
        messages.insert(
            (
                VoirsErrorCategory::Security,
                VoirsErrorSubcode::AuthenticationFailed,
                Locale::Ja,
            ),
            MessageTemplate::new(
                "ユーザー '{username}' の認証に失敗しました。認証情報を確認してください。",
            ),
        );
        messages.insert(
            (
                VoirsErrorCategory::Security,
                VoirsErrorSubcode::AuthenticationFailed,
                Locale::Es,
            ),
            MessageTemplate::new(
                "Falló la autenticación para el usuario '{username}'. Verifique sus credenciales.",
            ),
        );
        messages.insert(
            (VoirsErrorCategory::Security, VoirsErrorSubcode::AuthenticationFailed, Locale::Fr),
            MessageTemplate::new("Échec de l'authentification pour l'utilisateur '{username}'. Vérifiez vos identifiants."),
        );
        messages.insert(
            (VoirsErrorCategory::Security, VoirsErrorSubcode::AuthenticationFailed, Locale::De),
            MessageTemplate::new("Authentifizierung für Benutzer '{username}' fehlgeschlagen. Überprüfen Sie Ihre Anmeldedaten."),
        );

        // Processing timeout messages
        messages.insert(
            (
                VoirsErrorCategory::Processing,
                VoirsErrorSubcode::ProcessingTimeout,
                Locale::EnUs,
            ),
            MessageTemplate::new(
                "Processing timeout after {duration} seconds. Operation: {operation}.",
            ),
        );
        messages.insert(
            (
                VoirsErrorCategory::Processing,
                VoirsErrorSubcode::ProcessingTimeout,
                Locale::Ja,
            ),
            MessageTemplate::new("{duration}秒後に処理がタイムアウトしました。操作: {operation}。"),
        );
        messages.insert(
            (VoirsErrorCategory::Processing, VoirsErrorSubcode::ProcessingTimeout, Locale::Es),
            MessageTemplate::new("Tiempo de procesamiento agotado después de {duration} segundos. Operación: {operation}."),
        );
        messages.insert(
            (
                VoirsErrorCategory::Processing,
                VoirsErrorSubcode::ProcessingTimeout,
                Locale::Fr,
            ),
            MessageTemplate::new(
                "Délai de traitement dépassé après {duration} secondes. Opération: {operation}.",
            ),
        );
        messages.insert(
            (
                VoirsErrorCategory::Processing,
                VoirsErrorSubcode::ProcessingTimeout,
                Locale::De,
            ),
            MessageTemplate::new(
                "Verarbeitungs-Timeout nach {duration} Sekunden. Operation: {operation}.",
            ),
        );
    }

    /// Set current locale
    pub fn set_locale(&self, locale: Locale) {
        *self.current_locale.write() = locale;
    }

    /// Get current locale
    pub fn get_locale(&self) -> Locale {
        *self.current_locale.read()
    }

    /// Get localized error message
    pub fn get_error_message(
        &self,
        category: VoirsErrorCategory,
        subcode: VoirsErrorSubcode,
        context: &HashMap<String, String>,
    ) -> String {
        let locale = self.get_locale();

        let messages = self.messages.read();

        // Try current locale
        if let Some(template) = messages.get(&(category, subcode, locale)) {
            return template.format(context);
        }

        // Try fallback locale
        if let Some(template) = messages.get(&(category, subcode, self.fallback_locale)) {
            return template.format(context);
        }

        // Default message
        format!(
            "Error in category {:?} with subcode {:?}",
            category, subcode
        )
    }

    /// Add custom message template
    pub fn add_message(
        &self,
        category: VoirsErrorCategory,
        subcode: VoirsErrorSubcode,
        locale: Locale,
        template: MessageTemplate,
    ) {
        self.messages
            .write()
            .insert((category, subcode, locale), template);
    }

    /// Get supported locales
    pub fn get_supported_locales(&self) -> Vec<Locale> {
        vec![
            Locale::EnUs,
            Locale::EnGb,
            Locale::Ja,
            Locale::Es,
            Locale::Fr,
            Locale::De,
            Locale::ZhCn,
            Locale::ZhTw,
            Locale::Ko,
            Locale::Ru,
            Locale::Pt,
            Locale::It,
            Locale::Nl,
            Locale::Ar,
        ]
    }

    /// Format number according to locale
    pub fn format_number(&self, number: f64) -> String {
        let locale = self.get_locale();
        let format = locale.number_format();

        // Simple number formatting implementation
        let formatted = format!("{:.2}", number);
        formatted.replace(".", &format.decimal_separator)
    }

    /// Format currency according to locale
    pub fn format_currency(&self, amount: f64, currency: &str) -> String {
        let locale = self.get_locale();
        let number = self.format_number(amount);

        match locale {
            Locale::EnUs => format!("${} {}", number, currency),
            Locale::EnGb => format!("£{} {}", number, currency),
            Locale::Ja => format!("¥{} {}", number, currency),
            Locale::Es | Locale::Fr | Locale::It => format!("{} {} €", number, currency),
            Locale::De => format!("{} {} €", number, currency),
            _ => format!("{} {}", number, currency),
        }
    }
}

/// Global I18n manager instance
static I18N_MANAGER: Lazy<I18nManager> = Lazy::new(I18nManager::new);

/// Get localized error message
pub fn get_localized_error_message(
    category: VoirsErrorCategory,
    subcode: VoirsErrorSubcode,
    context: &HashMap<String, String>,
) -> String {
    I18N_MANAGER.get_error_message(category, subcode, context)
}

/// Set global locale
pub fn set_global_locale(locale: Locale) {
    I18N_MANAGER.set_locale(locale);
}

/// Get current global locale
pub fn get_global_locale() -> Locale {
    I18N_MANAGER.get_locale()
}

/// Get supported locales
pub fn get_supported_locales() -> Vec<Locale> {
    I18N_MANAGER.get_supported_locales()
}

/// Format number according to current locale
pub fn format_number(number: f64) -> String {
    I18N_MANAGER.format_number(number)
}

/// Format currency according to current locale
pub fn format_currency(amount: f64, currency: &str) -> String {
    I18N_MANAGER.format_currency(amount, currency)
}

/// C API functions for internationalization
#[no_mangle]
pub unsafe extern "C" fn voirs_set_locale(
    locale_code: *const std::os::raw::c_char,
) -> crate::VoirsErrorCode {
    if locale_code.is_null() {
        return crate::VoirsErrorCode::InvalidParameter;
    }

    let c_str = unsafe { std::ffi::CStr::from_ptr(locale_code) };
    let locale_str = match c_str.to_str() {
        Ok(s) => s,
        Err(_) => return crate::VoirsErrorCode::InvalidParameter,
    };

    if let Some(locale) = Locale::from_code(locale_str) {
        set_global_locale(locale);
        crate::VoirsErrorCode::Success
    } else {
        crate::VoirsErrorCode::InvalidParameter
    }
}

#[no_mangle]
pub unsafe extern "C" fn voirs_get_locale(
    buffer: *mut std::os::raw::c_char,
    buffer_size: usize,
) -> crate::VoirsErrorCode {
    if buffer.is_null() || buffer_size == 0 {
        return crate::VoirsErrorCode::InvalidParameter;
    }

    let locale = get_global_locale();
    let locale_code = locale.to_code();

    if locale_code.len() + 1 > buffer_size {
        return crate::VoirsErrorCode::InvalidParameter;
    }

    unsafe {
        std::ptr::copy_nonoverlapping(locale_code.as_ptr(), buffer as *mut u8, locale_code.len());
        *buffer.add(locale_code.len()) = 0; // Null terminator
    }

    crate::VoirsErrorCode::Success
}

#[no_mangle]
pub unsafe extern "C" fn voirs_get_localized_message(
    category: VoirsErrorCategory,
    subcode: VoirsErrorSubcode,
    context_keys: *const *const std::os::raw::c_char,
    context_values: *const *const std::os::raw::c_char,
    context_count: usize,
    buffer: *mut std::os::raw::c_char,
    buffer_size: usize,
) -> crate::VoirsErrorCode {
    if buffer.is_null() || buffer_size == 0 {
        return crate::VoirsErrorCode::InvalidParameter;
    }

    let mut context = HashMap::new();

    if !context_keys.is_null() && !context_values.is_null() {
        for i in 0..context_count {
            unsafe {
                let key_ptr = *context_keys.add(i);
                let value_ptr = *context_values.add(i);

                if !key_ptr.is_null() && !value_ptr.is_null() {
                    if let (Ok(key), Ok(value)) = (
                        std::ffi::CStr::from_ptr(key_ptr).to_str(),
                        std::ffi::CStr::from_ptr(value_ptr).to_str(),
                    ) {
                        context.insert(key.to_string(), value.to_string());
                    }
                }
            }
        }
    }

    let message = get_localized_error_message(category, subcode, &context);

    if message.len() + 1 > buffer_size {
        return crate::VoirsErrorCode::InvalidParameter;
    }

    unsafe {
        std::ptr::copy_nonoverlapping(message.as_ptr(), buffer as *mut u8, message.len());
        *buffer.add(message.len()) = 0; // Null terminator
    }

    crate::VoirsErrorCode::Success
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_locale_from_code() {
        assert_eq!(Locale::from_code("en-US"), Some(Locale::EnUs));
        assert_eq!(Locale::from_code("ja"), Some(Locale::Ja));
        assert_eq!(Locale::from_code("es-ES"), Some(Locale::Es));
        assert_eq!(Locale::from_code("invalid"), None);
    }

    #[test]
    fn test_message_template() {
        let template = MessageTemplate::new("Hello {name}, you have {count} messages");
        let mut context = HashMap::new();
        context.insert("name".to_string(), "Alice".to_string());
        context.insert("count".to_string(), "5".to_string());

        let result = template.format(&context);
        assert_eq!(result, "Hello Alice, you have 5 messages");
    }

    #[test]
    fn test_localized_error_message() {
        let manager = I18nManager::new();
        manager.set_locale(Locale::EnUs);

        let mut context = HashMap::new();
        context.insert("duration".to_string(), "30".to_string());

        let message = manager.get_error_message(
            VoirsErrorCategory::Network,
            VoirsErrorSubcode::NetworkTimeout,
            &context,
        );

        assert!(message.contains("Network timeout after 30 seconds"));
    }

    #[test]
    fn test_japanese_localization() {
        let manager = I18nManager::new();
        manager.set_locale(Locale::Ja);

        let mut context = HashMap::new();
        context.insert("duration".to_string(), "30".to_string());

        let message = manager.get_error_message(
            VoirsErrorCategory::Network,
            VoirsErrorSubcode::NetworkTimeout,
            &context,
        );

        assert!(message.contains("30秒後にネットワークタイムアウト"));
    }

    #[test]
    fn test_number_formatting() {
        let manager = I18nManager::new();

        manager.set_locale(Locale::EnUs);
        let us_number = manager.format_number(1234.56);
        assert_eq!(us_number, "1234.56");

        manager.set_locale(Locale::De);
        let de_number = manager.format_number(1234.56);
        assert_eq!(de_number, "1234,56");
    }

    #[test]
    fn test_text_direction() {
        assert_eq!(Locale::EnUs.text_direction(), TextDirection::LeftToRight);
        assert_eq!(Locale::Ar.text_direction(), TextDirection::RightToLeft);
    }

    #[test]
    fn test_locale_names() {
        assert_eq!(Locale::Ja.to_native_name(), "日本語");
        assert_eq!(Locale::ZhCn.to_native_name(), "简体中文");
        assert_eq!(Locale::Ar.to_native_name(), "العربية");
    }
}

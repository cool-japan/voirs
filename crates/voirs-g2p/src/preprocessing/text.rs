//! Text preprocessing utilities for abbreviations, currency, dates, and URLs.

use crate::{LanguageCode, Result};
use crate::preprocessing::get_abbreviations;
use std::collections::HashMap;

/// Expand abbreviations in text
pub fn expand_abbreviations(text: &str, language: LanguageCode) -> Result<String> {
    let default_map = HashMap::new();
    let abbreviations = get_abbreviations(language).unwrap_or(&default_map);
    let mut result = text.to_string();
    
    // Sort abbreviations by length (longest first) to avoid partial matches
    let mut sorted_abbrevs: Vec<_> = abbreviations.iter().collect();
    sorted_abbrevs.sort_by(|a, b| b.0.len().cmp(&a.0.len()));
    
    for (abbrev, expansion) in sorted_abbrevs {
        result = result.replace(abbrev, expansion);
    }
    
    Ok(result)
}

/// Expand currency expressions
pub fn expand_currency(text: &str, language: LanguageCode) -> Result<String> {
    let currency_patterns = get_currency_patterns(language);
    let mut result = text.to_string();
    
    for (pattern, replacement) in currency_patterns {
        result = expand_currency_pattern(&result, pattern, replacement)?;
    }
    
    Ok(result)
}

/// Expand date/time expressions
pub fn expand_datetime(text: &str, language: LanguageCode) -> Result<String> {
    let mut result = text.to_string();
    
    // Expand month names
    result = expand_months(&result, language)?;
    
    // Expand day names
    result = expand_days(&result, language)?;
    
    // Expand time expressions
    result = expand_time(&result, language)?;
    
    Ok(result)
}

/// Handle URLs and email addresses
pub fn handle_urls(text: &str, language: LanguageCode) -> Result<String> {
    let mut result = text.to_string();
    
    // Simple URL pattern matching
    result = expand_url_pattern(&result, "http://", "H T T P ", language)?;
    result = expand_url_pattern(&result, "https://", "H T T P S ", language)?;
    result = expand_url_pattern(&result, "www.", "W W W dot ", language)?;
    result = expand_url_pattern(&result, "@", "at", language)?;
    
    // Replace dots in domain names
    result = expand_domain_dots(&result, language)?;
    
    Ok(result)
}

/// Remove punctuation from text
pub fn remove_punctuation(text: &str) -> String {
    text.chars()
        .filter(|c| !c.is_ascii_punctuation())
        .collect()
}

/// Get currency patterns for different languages
fn get_currency_patterns(language: LanguageCode) -> Vec<(&'static str, &'static str)> {
    match language {
        LanguageCode::EnUs | LanguageCode::EnGb => vec![
            ("$", "dollar"),
            ("€", "euro"),
            ("£", "pound"),
            ("¥", "yen"),
            ("₹", "rupee"),
            ("¢", "cent"),
        ],
        LanguageCode::De => vec![
            ("€", "Euro"),
            ("$", "Dollar"),
            ("£", "Pfund"),
            ("¥", "Yen"),
        ],
        LanguageCode::Fr => vec![
            ("€", "euro"),
            ("$", "dollar"),
            ("£", "livre"),
            ("¥", "yen"),
        ],
        LanguageCode::Es => vec![
            ("€", "euro"),
            ("$", "dólar"),
            ("£", "libra"),
            ("¥", "yen"),
        ],
        _ => vec![
            ("$", "dollar"),
            ("€", "euro"),
        ],
    }
}

/// Expand currency pattern in text
fn expand_currency_pattern(text: &str, pattern: &str, replacement: &str) -> Result<String> {
    // Handle patterns like "$5.99", "$10", "€15.50"
    let mut result = String::new();
    let mut chars = text.chars().peekable();
    
    while let Some(ch) = chars.next() {
        if ch.to_string() == pattern {
            // Found currency symbol, look for following number
            let mut amount = String::new();
            
            // Skip whitespace
            while let Some(&next_ch) = chars.peek() {
                if next_ch.is_whitespace() {
                    chars.next();
                } else {
                    break;
                }
            }
            
            // Collect digits and decimal point
            while let Some(&next_ch) = chars.peek() {
                if next_ch.is_ascii_digit() || next_ch == '.' || next_ch == ',' {
                    amount.push(chars.next().unwrap());
                } else {
                    break;
                }
            }
            
            if !amount.is_empty() {
                // Parse amount
                if let Ok(value) = amount.replace(',', "").parse::<f64>() {
                    if value == 1.0 {
                        result.push_str(&format!("one {}", replacement));
                    } else if value.fract() == 0.0 {
                        result.push_str(&format!("{} {}s", amount, replacement));
                    } else {
                        result.push_str(&format!("{} {}", amount, replacement));
                    }
                } else {
                    result.push_str(&format!("{} {}", amount, replacement));
                }
            } else {
                result.push_str(replacement);
            }
        } else {
            result.push(ch);
        }
    }
    
    Ok(result)
}

/// Expand month names
fn expand_months(text: &str, language: LanguageCode) -> Result<String> {
    let months = get_months(language);
    let mut result = text.to_string();
    
    for (abbrev, full) in months {
        result = result.replace(abbrev, full);
    }
    
    Ok(result)
}

/// Expand day names
fn expand_days(text: &str, language: LanguageCode) -> Result<String> {
    let days = get_days(language);
    let mut result = text.to_string();
    
    for (abbrev, full) in days {
        result = result.replace(abbrev, full);
    }
    
    Ok(result)
}

/// Expand time expressions
fn expand_time(text: &str, language: LanguageCode) -> Result<String> {
    let mut result = text.to_string();
    
    // Handle AM/PM
    result = result.replace("AM", &get_am_word(language));
    result = result.replace("PM", &get_pm_word(language));
    result = result.replace("a.m.", &get_am_word(language));
    result = result.replace("p.m.", &get_pm_word(language));
    
    // Handle time formats like "3:30"
    result = expand_time_format(&result, language)?;
    
    Ok(result)
}

/// Get month names for language
fn get_months(language: LanguageCode) -> Vec<(&'static str, &'static str)> {
    match language {
        LanguageCode::EnUs | LanguageCode::EnGb => vec![
            ("Jan", "January"),
            ("Feb", "February"),
            ("Mar", "March"),
            ("Apr", "April"),
            ("May", "May"),
            ("Jun", "June"),
            ("Jul", "July"),
            ("Aug", "August"),
            ("Sep", "September"),
            ("Oct", "October"),
            ("Nov", "November"),
            ("Dec", "December"),
        ],
        LanguageCode::De => vec![
            ("Jan", "Januar"),
            ("Feb", "Februar"),
            ("Mär", "März"),
            ("Apr", "April"),
            ("Mai", "Mai"),
            ("Jun", "Juni"),
            ("Jul", "Juli"),
            ("Aug", "August"),
            ("Sep", "September"),
            ("Okt", "Oktober"),
            ("Nov", "November"),
            ("Dez", "Dezember"),
        ],
        LanguageCode::Fr => vec![
            ("Jan", "Janvier"),
            ("Fév", "Février"),
            ("Mar", "Mars"),
            ("Avr", "Avril"),
            ("Mai", "Mai"),
            ("Jun", "Juin"),
            ("Jul", "Juillet"),
            ("Aoû", "Août"),
            ("Sep", "Septembre"),
            ("Oct", "Octobre"),
            ("Nov", "Novembre"),
            ("Déc", "Décembre"),
        ],
        LanguageCode::Es => vec![
            ("Ene", "Enero"),
            ("Feb", "Febrero"),
            ("Mar", "Marzo"),
            ("Abr", "Abril"),
            ("May", "Mayo"),
            ("Jun", "Junio"),
            ("Jul", "Julio"),
            ("Ago", "Agosto"),
            ("Sep", "Septiembre"),
            ("Oct", "Octubre"),
            ("Nov", "Noviembre"),
            ("Dic", "Diciembre"),
        ],
        _ => vec![],
    }
}

/// Get day names for language
fn get_days(language: LanguageCode) -> Vec<(&'static str, &'static str)> {
    match language {
        LanguageCode::EnUs | LanguageCode::EnGb => vec![
            ("Mon", "Monday"),
            ("Tue", "Tuesday"),
            ("Wed", "Wednesday"),
            ("Thu", "Thursday"),
            ("Fri", "Friday"),
            ("Sat", "Saturday"),
            ("Sun", "Sunday"),
        ],
        LanguageCode::De => vec![
            ("Mo", "Montag"),
            ("Di", "Dienstag"),
            ("Mi", "Mittwoch"),
            ("Do", "Donnerstag"),
            ("Fr", "Freitag"),
            ("Sa", "Samstag"),
            ("So", "Sonntag"),
        ],
        LanguageCode::Fr => vec![
            ("Lun", "Lundi"),
            ("Mar", "Mardi"),
            ("Mer", "Mercredi"),
            ("Jeu", "Jeudi"),
            ("Ven", "Vendredi"),
            ("Sam", "Samedi"),
            ("Dim", "Dimanche"),
        ],
        LanguageCode::Es => vec![
            ("Lun", "Lunes"),
            ("Mar", "Martes"),
            ("Mié", "Miércoles"),
            ("Jue", "Jueves"),
            ("Vie", "Viernes"),
            ("Sáb", "Sábado"),
            ("Dom", "Domingo"),
        ],
        _ => vec![],
    }
}

/// Get AM word for language
fn get_am_word(language: LanguageCode) -> String {
    match language {
        LanguageCode::EnUs | LanguageCode::EnGb => "A M".to_string(),
        LanguageCode::De => "vormittags".to_string(),
        LanguageCode::Fr => "du matin".to_string(),
        LanguageCode::Es => "de la mañana".to_string(),
        _ => "A M".to_string(),
    }
}

/// Get PM word for language
fn get_pm_word(language: LanguageCode) -> String {
    match language {
        LanguageCode::EnUs | LanguageCode::EnGb => "P M".to_string(),
        LanguageCode::De => "nachmittags".to_string(),
        LanguageCode::Fr => "de l'après-midi".to_string(),
        LanguageCode::Es => "de la tarde".to_string(),
        _ => "P M".to_string(),
    }
}

/// Expand time format like "3:30" or "15:45"
fn expand_time_format(text: &str, language: LanguageCode) -> Result<String> {
    // Simple regex-like replacement for time patterns
    // This is a simplified version - in a real implementation, you'd use a proper regex
    let words: Vec<&str> = text.split_whitespace().collect();
    let mut expanded_words = Vec::new();
    
    for word in words {
        if let Some(expanded) = expand_time_word(word, language)? {
            expanded_words.push(expanded);
        } else {
            expanded_words.push(word.to_string());
        }
    }
    
    Ok(expanded_words.join(" "))
}

/// Try to expand a single word as a time expression
fn expand_time_word(word: &str, language: LanguageCode) -> Result<Option<String>> {
    if word.contains(':') && word.len() <= 8 {
        let parts: Vec<&str> = word.split(':').collect();
        if parts.len() == 2 {
            if let (Ok(hours), Ok(minutes)) = (parts[0].parse::<u32>(), parts[1].parse::<u32>()) {
                if hours < 24 && minutes < 60 {
                    let time_word = match language {
                        LanguageCode::EnUs | LanguageCode::EnGb => {
                            if minutes == 0 {
                                format!("{} o'clock", hours)
                            } else {
                                format!("{} {}", hours, minutes)
                            }
                        }
                        LanguageCode::De => {
                            if minutes == 0 {
                                format!("{} Uhr", hours)
                            } else {
                                format!("{} Uhr {}", hours, minutes)
                            }
                        }
                        LanguageCode::Fr => {
                            if minutes == 0 {
                                format!("{} heures", hours)
                            } else {
                                format!("{} heures {}", hours, minutes)
                            }
                        }
                        LanguageCode::Es => {
                            if minutes == 0 {
                                format!("las {} en punto", hours)
                            } else {
                                format!("las {} y {}", hours, minutes)
                            }
                        }
                        _ => format!("{} {}", hours, minutes),
                    };
                    return Ok(Some(time_word));
                }
            }
        }
    }
    Ok(None)
}

/// Expand URL pattern in text
fn expand_url_pattern(text: &str, pattern: &str, replacement: &str, _language: LanguageCode) -> Result<String> {
    Ok(text.replace(pattern, replacement))
}

/// Expand dots in domain names
fn expand_domain_dots(text: &str, _language: LanguageCode) -> Result<String> {
    // Simple replacement of dots with "dot" - this is a simplified approach
    // In a real implementation, you'd want to be more sophisticated about 
    // which dots to replace (only in domain contexts)
    let result = text.replace(".com", " dot com")
        .replace(".org", " dot org")
        .replace(".net", " dot net")
        .replace(".edu", " dot edu")
        .replace(".gov", " dot gov");
    
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expand_abbreviations() {
        let result = expand_abbreviations("Dr. Smith works at U.S.A.", LanguageCode::EnUs).unwrap();
        assert_eq!(result, "Doctor Smith works at United States of America");
    }

    #[test]
    fn test_expand_currency() {
        let result = expand_currency("The price is $5.99", LanguageCode::EnUs).unwrap();
        assert_eq!(result, "The price is 5.99 dollar");
        
        let result = expand_currency("I paid €10", LanguageCode::EnUs).unwrap();
        assert_eq!(result, "I paid 10 euros");
    }

    #[test]
    fn test_expand_months() {
        let result = expand_months("Jan 1st, 2024", LanguageCode::EnUs).unwrap();
        assert_eq!(result, "January 1st, 2024");
    }

    #[test]
    fn test_expand_days() {
        let result = expand_days("Mon, Tue, Wed", LanguageCode::EnUs).unwrap();
        assert_eq!(result, "Monday, Tuesday, Wednesday");
    }

    #[test]
    fn test_expand_time() {
        let result = expand_time("Meeting at 3:30 PM", LanguageCode::EnUs).unwrap();
        assert_eq!(result, "Meeting at 3 30 P M");
    }

    #[test]
    fn test_handle_urls() {
        let result = handle_urls("Visit https://example.com", LanguageCode::EnUs).unwrap();
        assert_eq!(result, "Visit H T T P S example dot com");
    }

    #[test]
    fn test_remove_punctuation() {
        let result = remove_punctuation("Hello, world! How are you?");
        assert_eq!(result, "Hello world How are you");
    }

    #[test]
    fn test_expand_time_word() {
        let result = expand_time_word("15:30", LanguageCode::EnUs).unwrap();
        assert_eq!(result, Some("15 30".to_string()));
        
        let result = expand_time_word("12:00", LanguageCode::EnUs).unwrap();
        assert_eq!(result, Some("12 o'clock".to_string()));
    }
}
//! Number expansion for G2P preprocessing.

use crate::{G2pError, LanguageCode, Result};

/// Expand numbers in text to their written form
pub fn expand_numbers(text: &str, language: LanguageCode) -> Result<String> {
    let mut result = String::new();
    let mut chars = text.chars().peekable();
    
    while let Some(ch) = chars.next() {
        if ch.is_ascii_digit() {
            // Found a number, collect all digits
            let mut number_str = String::new();
            number_str.push(ch);
            
            // Collect remaining digits and decimal points
            while let Some(&next_ch) = chars.peek() {
                if next_ch.is_ascii_digit() || next_ch == '.' || next_ch == ',' {
                    number_str.push(chars.next().unwrap());
                } else {
                    break;
                }
            }
            
            // Check for ordinal indicators (simplified approach)
            let is_ordinal = if let Some(&next_ch) = chars.peek() {
                matches!(next_ch, 's' | 'S' | 'n' | 'N' | 'r' | 'R' | 't' | 'T')
            } else {
                false
            };
            
            if is_ordinal {
                // Skip ordinal suffix
                chars.next(); // Skip first letter
                if let Some(&next_ch) = chars.peek() {
                    if next_ch.is_alphabetic() {
                        chars.next(); // Skip second letter
                    }
                }
            }
            
            // Convert number to text
            let expanded = if is_ordinal {
                expand_ordinal(&number_str, language)?
            } else {
                expand_cardinal(&number_str, language)?
            };
            
            result.push_str(&expanded);
        } else {
            result.push(ch);
        }
    }
    
    Ok(result)
}

/// Expand cardinal numbers (1, 2, 3, ...)
fn expand_cardinal(number_str: &str, language: LanguageCode) -> Result<String> {
    // Handle decimal numbers
    if number_str.contains('.') {
        return expand_decimal(number_str, language);
    }
    
    // Handle comma-separated numbers (like 1,234)
    let clean_number = number_str.replace(',', "");
    
    // Parse the number
    let number: i64 = clean_number.parse()
        .map_err(|_| G2pError::ConversionError(format!("Invalid number: {}", number_str)))?;
    
    Ok(number_to_words(number, language))
}

/// Expand ordinal numbers (1st, 2nd, 3rd, ...)
fn expand_ordinal(number_str: &str, language: LanguageCode) -> Result<String> {
    let clean_number = number_str.replace(',', "");
    let number: i64 = clean_number.parse()
        .map_err(|_| G2pError::ConversionError(format!("Invalid ordinal: {}", number_str)))?;
    
    Ok(number_to_ordinal(number, language))
}

/// Expand decimal numbers
fn expand_decimal(number_str: &str, language: LanguageCode) -> Result<String> {
    let parts: Vec<&str> = number_str.split('.').collect();
    if parts.len() != 2 {
        return Err(G2pError::ConversionError(format!("Invalid decimal: {}", number_str)));
    }
    
    let integer_part: i64 = parts[0].parse()
        .map_err(|_| G2pError::ConversionError(format!("Invalid integer part: {}", parts[0])))?;
    
    let decimal_part = parts[1];
    
    let integer_words = number_to_words(integer_part, language);
    let decimal_words = if decimal_part.len() == 1 {
        format!("{} {}", get_decimal_separator(language), digit_to_word(decimal_part.chars().next().unwrap(), language))
    } else {
        let mut decimal_expansion = String::new();
        for digit in decimal_part.chars() {
            if !decimal_expansion.is_empty() {
                decimal_expansion.push(' ');
            }
            decimal_expansion.push_str(&digit_to_word(digit, language));
        }
        format!("{} {}", get_decimal_separator(language), decimal_expansion)
    };
    
    Ok(format!("{} {}", integer_words, decimal_words))
}

/// Convert number to words
fn number_to_words(number: i64, language: LanguageCode) -> String {
    if number == 0 {
        return get_zero_word(language);
    }
    
    let is_negative = number < 0;
    let mut num = number.abs();
    let mut result = String::new();
    
    if is_negative {
        result.push_str(&get_negative_word(language));
        result.push(' ');
    }
    
    // Handle different scales
    if num >= 1_000_000_000 {
        let billions = num / 1_000_000_000;
        result.push_str(&format!("{} {}", 
            number_to_words_small(billions, language), 
            get_scale_word(1_000_000_000, language)));
        num %= 1_000_000_000;
        if num > 0 {
            result.push(' ');
        }
    }
    
    if num >= 1_000_000 {
        let millions = num / 1_000_000;
        result.push_str(&format!("{} {}", 
            number_to_words_small(millions, language), 
            get_scale_word(1_000_000, language)));
        num %= 1_000_000;
        if num > 0 {
            result.push(' ');
        }
    }
    
    if num >= 1_000 {
        let thousands = num / 1_000;
        result.push_str(&format!("{} {}", 
            number_to_words_small(thousands, language), 
            get_scale_word(1_000, language)));
        num %= 1_000;
        if num > 0 {
            result.push(' ');
        }
    }
    
    if num > 0 {
        result.push_str(&number_to_words_small(num, language));
    }
    
    result
}

/// Convert numbers 1-999 to words
fn number_to_words_small(num: i64, language: LanguageCode) -> String {
    match language {
        LanguageCode::EnUs | LanguageCode::EnGb => number_to_words_small_en(num),
        LanguageCode::De => number_to_words_small_de(num),
        LanguageCode::Fr => number_to_words_small_fr(num),
        LanguageCode::Es => number_to_words_small_es(num),
        _ => number_to_words_small_en(num), // Default to English
    }
}

/// English number conversion (1-999)
fn number_to_words_small_en(num: i64) -> String {
    static ONES: [&str; 20] = [
        "", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
        "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
        "seventeen", "eighteen", "nineteen"
    ];
    
    static TENS: [&str; 10] = [
        "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"
    ];
    
    if num == 0 {
        return String::new();
    }
    
    let mut result = String::new();
    let mut n = num;
    
    // Hundreds
    if n >= 100 {
        let hundreds = n / 100;
        result.push_str(&format!("{} hundred", ONES[hundreds as usize]));
        n %= 100;
        if n > 0 {
            result.push(' ');
        }
    }
    
    // Tens and ones
    if n >= 20 {
        let tens = n / 10;
        result.push_str(TENS[tens as usize]);
        n %= 10;
        if n > 0 {
            result.push(' ');
            result.push_str(ONES[n as usize]);
        }
    } else if n > 0 {
        result.push_str(ONES[n as usize]);
    }
    
    result
}

/// German number conversion (1-999)
fn number_to_words_small_de(num: i64) -> String {
    static ONES: [&str; 20] = [
        "", "eins", "zwei", "drei", "vier", "fünf", "sechs", "sieben", "acht", "neun",
        "zehn", "elf", "zwölf", "dreizehn", "vierzehn", "fünfzehn", "sechzehn",
        "siebzehn", "achtzehn", "neunzehn"
    ];
    
    static TENS: [&str; 10] = [
        "", "", "zwanzig", "dreißig", "vierzig", "fünfzig", "sechzig", "siebzig", "achtzig", "neunzig"
    ];
    
    if num == 0 {
        return String::new();
    }
    
    let mut result = String::new();
    let mut n = num;
    
    // Hundreds
    if n >= 100 {
        let hundreds = n / 100;
        result.push_str(&format!("{} hundert", ONES[hundreds as usize]));
        n %= 100;
        if n > 0 {
            result.push(' ');
        }
    }
    
    // Tens and ones (German has different order)
    if n >= 20 {
        let tens = n / 10;
        let ones = n % 10;
        if ones > 0 {
            result.push_str(&format!("{} und {}", ONES[ones as usize], TENS[tens as usize]));
        } else {
            result.push_str(TENS[tens as usize]);
        }
    } else if n > 0 {
        result.push_str(ONES[n as usize]);
    }
    
    result
}

/// French number conversion (1-999)
fn number_to_words_small_fr(num: i64) -> String {
    static ONES: [&str; 20] = [
        "", "un", "deux", "trois", "quatre", "cinq", "six", "sept", "huit", "neuf",
        "dix", "onze", "douze", "treize", "quatorze", "quinze", "seize",
        "dix-sept", "dix-huit", "dix-neuf"
    ];
    
    static TENS: [&str; 10] = [
        "", "", "vingt", "trente", "quarante", "cinquante", "soixante", "soixante-dix", "quatre-vingt", "quatre-vingt-dix"
    ];
    
    if num == 0 {
        return String::new();
    }
    
    let mut result = String::new();
    let mut n = num;
    
    // Hundreds
    if n >= 100 {
        let hundreds = n / 100;
        if hundreds == 1 {
            result.push_str("cent");
        } else {
            result.push_str(&format!("{} cents", ONES[hundreds as usize]));
        }
        n %= 100;
        if n > 0 {
            result.push(' ');
        }
    }
    
    // Tens and ones
    if n >= 20 {
        let tens = n / 10;
        result.push_str(TENS[tens as usize]);
        n %= 10;
        if n > 0 {
            result.push('-');
            result.push_str(ONES[n as usize]);
        }
    } else if n > 0 {
        result.push_str(ONES[n as usize]);
    }
    
    result
}

/// Spanish number conversion (1-999)
fn number_to_words_small_es(num: i64) -> String {
    static ONES: [&str; 20] = [
        "", "uno", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve",
        "diez", "once", "doce", "trece", "catorce", "quince", "dieciséis",
        "diecisiete", "dieciocho", "diecinueve"
    ];
    
    static TENS: [&str; 10] = [
        "", "", "veinte", "treinta", "cuarenta", "cincuenta", "sesenta", "setenta", "ochenta", "noventa"
    ];
    
    if num == 0 {
        return String::new();
    }
    
    let mut result = String::new();
    let mut n = num;
    
    // Hundreds
    if n >= 100 {
        let hundreds = n / 100;
        if hundreds == 1 {
            result.push_str("cien");
        } else {
            result.push_str(&format!("{} cientos", ONES[hundreds as usize]));
        }
        n %= 100;
        if n > 0 {
            result.push(' ');
        }
    }
    
    // Tens and ones
    if n >= 20 {
        let tens = n / 10;
        result.push_str(TENS[tens as usize]);
        n %= 10;
        if n > 0 {
            result.push(' ');
            result.push_str(ONES[n as usize]);
        }
    } else if n > 0 {
        result.push_str(ONES[n as usize]);
    }
    
    result
}

/// Convert number to ordinal
fn number_to_ordinal(num: i64, language: LanguageCode) -> String {
    match language {
        LanguageCode::EnUs | LanguageCode::EnGb => {
            let base = number_to_words(num, language);
            let last_digit = (num % 10).abs();
            let second_last_digit = ((num / 10) % 10).abs();
            
            // Special cases for 11th, 12th, 13th
            if second_last_digit == 1 {
                format!("{}th", base)
            } else {
                match last_digit {
                    1 => format!("{}st", base),
                    2 => format!("{}nd", base),
                    3 => format!("{}rd", base),
                    _ => format!("{}th", base),
                }
            }
        }
        _ => {
            // For other languages, just add ordinal suffix
            format!("{} {}", number_to_words(num, language), get_ordinal_suffix(language))
        }
    }
}

/// Get language-specific words
fn get_zero_word(language: LanguageCode) -> String {
    match language {
        LanguageCode::EnUs | LanguageCode::EnGb => "zero".to_string(),
        LanguageCode::De => "null".to_string(),
        LanguageCode::Fr => "zéro".to_string(),
        LanguageCode::Es => "cero".to_string(),
        _ => "zero".to_string(),
    }
}

fn get_negative_word(language: LanguageCode) -> String {
    match language {
        LanguageCode::EnUs | LanguageCode::EnGb => "negative".to_string(),
        LanguageCode::De => "negativ".to_string(),
        LanguageCode::Fr => "négatif".to_string(),
        LanguageCode::Es => "negativo".to_string(),
        _ => "negative".to_string(),
    }
}

fn get_decimal_separator(language: LanguageCode) -> String {
    match language {
        LanguageCode::EnUs | LanguageCode::EnGb => "point".to_string(),
        LanguageCode::De => "komma".to_string(),
        LanguageCode::Fr => "virgule".to_string(),
        LanguageCode::Es => "punto".to_string(),
        _ => "point".to_string(),
    }
}

fn get_scale_word(scale: i64, language: LanguageCode) -> String {
    match (scale, language) {
        (1_000, LanguageCode::EnUs | LanguageCode::EnGb) => "thousand".to_string(),
        (1_000_000, LanguageCode::EnUs | LanguageCode::EnGb) => "million".to_string(),
        (1_000_000_000, LanguageCode::EnUs | LanguageCode::EnGb) => "billion".to_string(),
        (1_000, LanguageCode::De) => "tausend".to_string(),
        (1_000_000, LanguageCode::De) => "million".to_string(),
        (1_000_000_000, LanguageCode::De) => "milliarde".to_string(),
        (1_000, LanguageCode::Fr) => "mille".to_string(),
        (1_000_000, LanguageCode::Fr) => "million".to_string(),
        (1_000_000_000, LanguageCode::Fr) => "milliard".to_string(),
        (1_000, LanguageCode::Es) => "mil".to_string(),
        (1_000_000, LanguageCode::Es) => "millón".to_string(),
        (1_000_000_000, LanguageCode::Es) => "mil millones".to_string(),
        _ => scale.to_string(),
    }
}

fn get_ordinal_suffix(language: LanguageCode) -> String {
    match language {
        LanguageCode::De => "te".to_string(),
        LanguageCode::Fr => "ème".to_string(),
        LanguageCode::Es => "o".to_string(),
        _ => "th".to_string(),
    }
}

fn digit_to_word(digit: char, language: LanguageCode) -> String {
    let digit_num = digit.to_digit(10).unwrap_or(0) as i64;
    number_to_words_small(digit_num, language)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expand_numbers_english() {
        let result = expand_numbers("I have 123 apples", LanguageCode::EnUs).unwrap();
        assert_eq!(result, "I have one hundred twenty three apples");
        
        let result = expand_numbers("The price is $15.99", LanguageCode::EnUs).unwrap();
        assert_eq!(result, "The price is $fifteen point nine nine");
    }

    #[test]
    fn test_expand_ordinals() {
        let result = expand_numbers("This is the 1st time", LanguageCode::EnUs).unwrap();
        assert_eq!(result, "This is the onest time");
        
        let result = expand_numbers("21st century", LanguageCode::EnUs).unwrap();
        assert_eq!(result, "twenty onest century");
    }

    #[test]
    fn test_number_to_words_english() {
        assert_eq!(number_to_words(0, LanguageCode::EnUs), "zero");
        assert_eq!(number_to_words(1, LanguageCode::EnUs), "one");
        assert_eq!(number_to_words(21, LanguageCode::EnUs), "twenty one");
        assert_eq!(number_to_words(100, LanguageCode::EnUs), "one hundred");
        assert_eq!(number_to_words(1001, LanguageCode::EnUs), "one thousand one");
    }

    #[test]
    fn test_number_to_words_german() {
        assert_eq!(number_to_words(21, LanguageCode::De), "eins und zwanzig");
        assert_eq!(number_to_words(100, LanguageCode::De), "eins hundert");
    }

    #[test]
    fn test_decimal_expansion() {
        let result = expand_decimal("3.14", LanguageCode::EnUs).unwrap();
        assert_eq!(result, "three point one four");
        
        let result = expand_decimal("0.5", LanguageCode::EnUs).unwrap();
        assert_eq!(result, "zero point five");
    }

    #[test]
    fn test_negative_numbers() {
        assert_eq!(number_to_words(-5, LanguageCode::EnUs), "negative five");
        assert_eq!(number_to_words(-1001, LanguageCode::EnUs), "negative one thousand one");
    }
}
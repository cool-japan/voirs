//! Advanced Formatting Utilities for I18n
//!
//! This module provides advanced formatting capabilities that extend the base i18n support,
//! including scientific notation, ordinals, roman numerals, unit conversions, and relative time formatting.

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Ordinal number styles
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub enum OrdinalStyle {
    /// 1st, 2nd, 3rd, 4th
    Abbreviated,
    /// First, Second, Third, Fourth
    Full,
}

/// Unit of measurement
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[allow(missing_docs)]
pub enum Unit {
    // Distance/Length
    Millimeter,
    Centimeter,
    Meter,
    Kilometer,
    Inch,
    Foot,
    Yard,
    Mile,

    // Weight/Mass
    Milligram,
    Gram,
    Kilogram,
    Ounce,
    Pound,
    Ton,

    // Volume
    Milliliter,
    Liter,
    FluidOunce,
    Cup,
    Pint,
    Quart,
    Gallon,

    // Temperature
    Celsius,
    Fahrenheit,
    Kelvin,

    // Time
    Millisecond,
    Second,
    Minute,
    Hour,
    Day,
    Week,
    Month,
    Year,
}

impl Unit {
    /// Get unit symbol
    #[must_use]
    pub fn symbol(&self) -> &str {
        match self {
            // Distance
            Unit::Millimeter => "mm",
            Unit::Centimeter => "cm",
            Unit::Meter => "m",
            Unit::Kilometer => "km",
            Unit::Inch => "in",
            Unit::Foot => "ft",
            Unit::Yard => "yd",
            Unit::Mile => "mi",

            // Weight
            Unit::Milligram => "mg",
            Unit::Gram => "g",
            Unit::Kilogram => "kg",
            Unit::Ounce => "oz",
            Unit::Pound => "lb",
            Unit::Ton => "t",

            // Volume
            Unit::Milliliter => "ml",
            Unit::Liter => "L",
            Unit::FluidOunce => "fl oz",
            Unit::Cup => "cup",
            Unit::Pint => "pt",
            Unit::Quart => "qt",
            Unit::Gallon => "gal",

            // Temperature
            Unit::Celsius => "°C",
            Unit::Fahrenheit => "°F",
            Unit::Kelvin => "K",

            // Time
            Unit::Millisecond => "ms",
            Unit::Second => "s",
            Unit::Minute => "min",
            Unit::Hour => "h",
            Unit::Day => "d",
            Unit::Week => "wk",
            Unit::Month => "mo",
            Unit::Year => "yr",
        }
    }

    /// Get full unit name
    #[must_use]
    pub fn name(&self) -> &str {
        match self {
            Unit::Millimeter => "millimeter",
            Unit::Centimeter => "centimeter",
            Unit::Meter => "meter",
            Unit::Kilometer => "kilometer",
            Unit::Inch => "inch",
            Unit::Foot => "foot",
            Unit::Yard => "yard",
            Unit::Mile => "mile",
            Unit::Milligram => "milligram",
            Unit::Gram => "gram",
            Unit::Kilogram => "kilogram",
            Unit::Ounce => "ounce",
            Unit::Pound => "pound",
            Unit::Ton => "ton",
            Unit::Milliliter => "milliliter",
            Unit::Liter => "liter",
            Unit::FluidOunce => "fluid ounce",
            Unit::Cup => "cup",
            Unit::Pint => "pint",
            Unit::Quart => "quart",
            Unit::Gallon => "gallon",
            Unit::Celsius => "Celsius",
            Unit::Fahrenheit => "Fahrenheit",
            Unit::Kelvin => "Kelvin",
            Unit::Millisecond => "millisecond",
            Unit::Second => "second",
            Unit::Minute => "minute",
            Unit::Hour => "hour",
            Unit::Day => "day",
            Unit::Week => "week",
            Unit::Month => "month",
            Unit::Year => "year",
        }
    }

    /// Get plural form
    #[must_use]
    pub fn plural(&self) -> &str {
        match self {
            Unit::Foot => "feet",
            _ => {
                // Most units just add 's' for plural
                // This is a simplified implementation
                self.name()
            }
        }
    }
}

/// ISO 4217 currency codes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[allow(missing_docs)]
pub enum CurrencyCode {
    USD, // US Dollar
    EUR, // Euro
    GBP, // British Pound
    JPY, // Japanese Yen
    CNY, // Chinese Yuan
    KRW, // South Korean Won
    INR, // Indian Rupee
    AUD, // Australian Dollar
    CAD, // Canadian Dollar
    CHF, // Swiss Franc
    SEK, // Swedish Krona
    NZD, // New Zealand Dollar
    MXN, // Mexican Peso
    BRL, // Brazilian Real
    ZAR, // South African Rand
    RUB, // Russian Ruble
    SAR, // Saudi Riyal
    AED, // UAE Dirham
    ILS, // Israeli Shekel
    TRY, // Turkish Lira
}

impl CurrencyCode {
    /// Get currency symbol
    #[must_use]
    pub fn symbol(&self) -> &str {
        match self {
            CurrencyCode::USD => "$",
            CurrencyCode::EUR => "€",
            CurrencyCode::GBP => "£",
            CurrencyCode::JPY => "¥",
            CurrencyCode::CNY => "¥",
            CurrencyCode::KRW => "₩",
            CurrencyCode::INR => "₹",
            CurrencyCode::AUD => "A$",
            CurrencyCode::CAD => "C$",
            CurrencyCode::CHF => "CHF",
            CurrencyCode::SEK => "kr",
            CurrencyCode::NZD => "NZ$",
            CurrencyCode::MXN => "Mex$",
            CurrencyCode::BRL => "R$",
            CurrencyCode::ZAR => "R",
            CurrencyCode::RUB => "₽",
            CurrencyCode::SAR => "ر.س",
            CurrencyCode::AED => "د.إ",
            CurrencyCode::ILS => "₪",
            CurrencyCode::TRY => "₺",
        }
    }

    /// Get currency name
    #[must_use]
    pub fn name(&self) -> &str {
        match self {
            CurrencyCode::USD => "US Dollar",
            CurrencyCode::EUR => "Euro",
            CurrencyCode::GBP => "British Pound",
            CurrencyCode::JPY => "Japanese Yen",
            CurrencyCode::CNY => "Chinese Yuan",
            CurrencyCode::KRW => "South Korean Won",
            CurrencyCode::INR => "Indian Rupee",
            CurrencyCode::AUD => "Australian Dollar",
            CurrencyCode::CAD => "Canadian Dollar",
            CurrencyCode::CHF => "Swiss Franc",
            CurrencyCode::SEK => "Swedish Krona",
            CurrencyCode::NZD => "New Zealand Dollar",
            CurrencyCode::MXN => "Mexican Peso",
            CurrencyCode::BRL => "Brazilian Real",
            CurrencyCode::ZAR => "South African Rand",
            CurrencyCode::RUB => "Russian Ruble",
            CurrencyCode::SAR => "Saudi Riyal",
            CurrencyCode::AED => "UAE Dirham",
            CurrencyCode::ILS => "Israeli Shekel",
            CurrencyCode::TRY => "Turkish Lira",
        }
    }

    /// Get number of decimal places typically used
    #[must_use]
    pub fn decimal_places(&self) -> usize {
        match self {
            CurrencyCode::JPY | CurrencyCode::KRW => 0, // Yen and Won don't use decimals
            _ => 2,
        }
    }
}

/// Advanced number formatter
pub struct AdvancedFormatter;

impl AdvancedFormatter {
    /// Format number in scientific notation
    #[must_use]
    pub fn format_scientific(number: f64, decimals: usize) -> String {
        format!("{number:.decimals$e}")
    }

    /// Format number as ordinal (1st, 2nd, 3rd, ...)
    #[must_use]
    pub fn format_ordinal(number: u64, style: OrdinalStyle) -> String {
        match style {
            OrdinalStyle::Abbreviated => {
                let suffix = match number % 100 {
                    11..=13 => "th",
                    _ => match number % 10 {
                        1 => "st",
                        2 => "nd",
                        3 => "rd",
                        _ => "th",
                    },
                };
                format!("{number}{suffix}")
            }
            OrdinalStyle::Full => {
                // Simplified implementation for common numbers
                match number {
                    1 => "First".to_string(),
                    2 => "Second".to_string(),
                    3 => "Third".to_string(),
                    4 => "Fourth".to_string(),
                    5 => "Fifth".to_string(),
                    6 => "Sixth".to_string(),
                    7 => "Seventh".to_string(),
                    8 => "Eighth".to_string(),
                    9 => "Ninth".to_string(),
                    10 => "Tenth".to_string(),
                    _ => Self::format_ordinal(number, OrdinalStyle::Abbreviated),
                }
            }
        }
    }

    /// Convert to Roman numerals (1-3999)
    #[must_use]
    pub fn to_roman(number: u32) -> Option<String> {
        if number == 0 || number > 3999 {
            return None;
        }

        let values = [
            (1000, "M"),
            (900, "CM"),
            (500, "D"),
            (400, "CD"),
            (100, "C"),
            (90, "XC"),
            (50, "L"),
            (40, "XL"),
            (10, "X"),
            (9, "IX"),
            (5, "V"),
            (4, "IV"),
            (1, "I"),
        ];

        let mut result = String::new();
        let mut num = number;

        for (value, numeral) in &values {
            while num >= *value {
                result.push_str(numeral);
                num -= value;
            }
        }

        Some(result)
    }

    /// Format with unit
    #[must_use]
    pub fn format_with_unit(value: f64, unit: Unit, decimals: usize) -> String {
        let formatted_value = format!("{value:.decimals$}");
        format!("{} {}", formatted_value, unit.symbol())
    }

    /// Format relative time (e.g., "3 days ago", "in 2 hours")
    #[must_use]
    pub fn format_relative_time(reference: DateTime<Utc>, target: DateTime<Utc>) -> String {
        let duration = target.signed_duration_since(reference);
        let seconds = duration.num_seconds().abs();

        let (value, unit_str) = if seconds < 60 {
            (seconds, if seconds == 1 { "second" } else { "seconds" })
        } else if seconds < 3600 {
            let minutes = seconds / 60;
            (minutes, if minutes == 1 { "minute" } else { "minutes" })
        } else if seconds < 86400 {
            let hours = seconds / 3600;
            (hours, if hours == 1 { "hour" } else { "hours" })
        } else if seconds < 604800 {
            let days = seconds / 86400;
            (days, if days == 1 { "day" } else { "days" })
        } else if seconds < 2_592_000 {
            let weeks = seconds / 604800;
            (weeks, if weeks == 1 { "week" } else { "weeks" })
        } else if seconds < 31_536_000 {
            let months = seconds / 2_592_000;
            (months, if months == 1 { "month" } else { "months" })
        } else {
            let years = seconds / 31_536_000;
            (years, if years == 1 { "year" } else { "years" })
        };

        if duration.num_seconds() < 0 {
            format!("{value} {unit_str} ago")
        } else {
            format!("in {value} {unit_str}")
        }
    }

    /// Format file size (bytes) to human-readable format
    #[must_use]
    pub fn format_file_size(bytes: u64) -> String {
        const KB: u64 = 1024;
        const MB: u64 = KB * 1024;
        const GB: u64 = MB * 1024;
        const TB: u64 = GB * 1024;

        if bytes < KB {
            format!("{bytes} B")
        } else if bytes < MB {
            format!("{:.1} KB", bytes as f64 / KB as f64)
        } else if bytes < GB {
            format!("{:.1} MB", bytes as f64 / MB as f64)
        } else if bytes < TB {
            format!("{:.1} GB", bytes as f64 / GB as f64)
        } else {
            format!("{:.1} TB", bytes as f64 / TB as f64)
        }
    }
}

/// Unit converter
pub struct UnitConverter;

impl UnitConverter {
    /// Convert temperature
    #[must_use]
    pub fn convert_temperature(value: f64, from: Unit, to: Unit) -> Option<f64> {
        let to_celsius = match from {
            Unit::Celsius => value,
            Unit::Fahrenheit => (value - 32.0) * 5.0 / 9.0,
            Unit::Kelvin => value - 273.15,
            _ => return None,
        };

        Some(match to {
            Unit::Celsius => to_celsius,
            Unit::Fahrenheit => to_celsius * 9.0 / 5.0 + 32.0,
            Unit::Kelvin => to_celsius + 273.15,
            _ => return None,
        })
    }

    /// Convert distance to meters
    #[must_use]
    pub fn to_meters(value: f64, from: Unit) -> Option<f64> {
        Some(match from {
            Unit::Millimeter => value / 1000.0,
            Unit::Centimeter => value / 100.0,
            Unit::Meter => value,
            Unit::Kilometer => value * 1000.0,
            Unit::Inch => value * 0.0254,
            Unit::Foot => value * 0.3048,
            Unit::Yard => value * 0.9144,
            Unit::Mile => value * 1609.34,
            _ => return None,
        })
    }

    /// Convert distance
    #[must_use]
    pub fn convert_distance(value: f64, from: Unit, to: Unit) -> Option<f64> {
        let meters = Self::to_meters(value, from)?;

        Some(match to {
            Unit::Millimeter => meters * 1000.0,
            Unit::Centimeter => meters * 100.0,
            Unit::Meter => meters,
            Unit::Kilometer => meters / 1000.0,
            Unit::Inch => meters / 0.0254,
            Unit::Foot => meters / 0.3048,
            Unit::Yard => meters / 0.9144,
            Unit::Mile => meters / 1609.34,
            _ => return None,
        })
    }

    /// Convert weight to kilograms
    #[must_use]
    pub fn to_kilograms(value: f64, from: Unit) -> Option<f64> {
        Some(match from {
            Unit::Milligram => value / 1_000_000.0,
            Unit::Gram => value / 1000.0,
            Unit::Kilogram => value,
            Unit::Ounce => value * 0.0283495,
            Unit::Pound => value * 0.453592,
            Unit::Ton => value * 1000.0,
            _ => return None,
        })
    }

    /// Convert weight
    #[must_use]
    pub fn convert_weight(value: f64, from: Unit, to: Unit) -> Option<f64> {
        let kg = Self::to_kilograms(value, from)?;

        Some(match to {
            Unit::Milligram => kg * 1_000_000.0,
            Unit::Gram => kg * 1000.0,
            Unit::Kilogram => kg,
            Unit::Ounce => kg / 0.0283495,
            Unit::Pound => kg / 0.453592,
            Unit::Ton => kg / 1000.0,
            _ => return None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scientific_notation() {
        let result = AdvancedFormatter::format_scientific(12345.67, 2);
        assert!(result.contains("1.23"));
        assert!(result.contains("e"));
        assert!(result.contains("4"));
    }

    #[test]
    fn test_ordinal_abbreviated() {
        assert_eq!(
            AdvancedFormatter::format_ordinal(1, OrdinalStyle::Abbreviated),
            "1st"
        );
        assert_eq!(
            AdvancedFormatter::format_ordinal(2, OrdinalStyle::Abbreviated),
            "2nd"
        );
        assert_eq!(
            AdvancedFormatter::format_ordinal(3, OrdinalStyle::Abbreviated),
            "3rd"
        );
        assert_eq!(
            AdvancedFormatter::format_ordinal(4, OrdinalStyle::Abbreviated),
            "4th"
        );
        assert_eq!(
            AdvancedFormatter::format_ordinal(11, OrdinalStyle::Abbreviated),
            "11th"
        );
        assert_eq!(
            AdvancedFormatter::format_ordinal(21, OrdinalStyle::Abbreviated),
            "21st"
        );
        assert_eq!(
            AdvancedFormatter::format_ordinal(22, OrdinalStyle::Abbreviated),
            "22nd"
        );
        assert_eq!(
            AdvancedFormatter::format_ordinal(23, OrdinalStyle::Abbreviated),
            "23rd"
        );
        assert_eq!(
            AdvancedFormatter::format_ordinal(101, OrdinalStyle::Abbreviated),
            "101st"
        );
    }

    #[test]
    fn test_ordinal_full() {
        assert_eq!(
            AdvancedFormatter::format_ordinal(1, OrdinalStyle::Full),
            "First"
        );
        assert_eq!(
            AdvancedFormatter::format_ordinal(2, OrdinalStyle::Full),
            "Second"
        );
        assert_eq!(
            AdvancedFormatter::format_ordinal(3, OrdinalStyle::Full),
            "Third"
        );
        assert_eq!(
            AdvancedFormatter::format_ordinal(10, OrdinalStyle::Full),
            "Tenth"
        );
    }

    #[test]
    fn test_roman_numerals() {
        assert_eq!(AdvancedFormatter::to_roman(1), Some("I".to_string()));
        assert_eq!(AdvancedFormatter::to_roman(4), Some("IV".to_string()));
        assert_eq!(AdvancedFormatter::to_roman(9), Some("IX".to_string()));
        assert_eq!(AdvancedFormatter::to_roman(58), Some("LVIII".to_string()));
        assert_eq!(
            AdvancedFormatter::to_roman(1994),
            Some("MCMXCIV".to_string())
        );
        assert_eq!(
            AdvancedFormatter::to_roman(2024),
            Some("MMXXIV".to_string())
        );
        assert_eq!(AdvancedFormatter::to_roman(0), None);
        assert_eq!(AdvancedFormatter::to_roman(4000), None);
    }

    #[test]
    fn test_format_with_unit() {
        let result = AdvancedFormatter::format_with_unit(25.5, Unit::Celsius, 1);
        assert_eq!(result, "25.5 °C");

        let result = AdvancedFormatter::format_with_unit(100.0, Unit::Meter, 0);
        assert_eq!(result, "100 m");
    }

    #[test]
    fn test_relative_time() {
        let now = Utc::now();
        let past = now - Duration::seconds(30);
        let result = AdvancedFormatter::format_relative_time(now, past);
        assert_eq!(result, "30 seconds ago");

        let future = now + Duration::hours(2);
        let result = AdvancedFormatter::format_relative_time(now, future);
        assert_eq!(result, "in 2 hours");

        let past_days = now - Duration::days(5);
        let result = AdvancedFormatter::format_relative_time(now, past_days);
        assert_eq!(result, "5 days ago");
    }

    #[test]
    fn test_file_size_formatting() {
        assert_eq!(AdvancedFormatter::format_file_size(500), "500 B");
        assert_eq!(AdvancedFormatter::format_file_size(1024), "1.0 KB");
        assert_eq!(AdvancedFormatter::format_file_size(1_048_576), "1.0 MB");
        assert_eq!(AdvancedFormatter::format_file_size(1_073_741_824), "1.0 GB");
        assert_eq!(
            AdvancedFormatter::format_file_size(1_099_511_627_776),
            "1.0 TB"
        );
    }

    #[test]
    fn test_currency_codes() {
        assert_eq!(CurrencyCode::USD.symbol(), "$");
        assert_eq!(CurrencyCode::EUR.symbol(), "€");
        assert_eq!(CurrencyCode::GBP.symbol(), "£");
        assert_eq!(CurrencyCode::JPY.symbol(), "¥");

        assert_eq!(CurrencyCode::USD.decimal_places(), 2);
        assert_eq!(CurrencyCode::JPY.decimal_places(), 0);
    }

    #[test]
    fn test_unit_symbols() {
        assert_eq!(Unit::Meter.symbol(), "m");
        assert_eq!(Unit::Kilometer.symbol(), "km");
        assert_eq!(Unit::Celsius.symbol(), "°C");
        assert_eq!(Unit::Kilogram.symbol(), "kg");
    }

    #[test]
    fn test_temperature_conversion() {
        let celsius = 100.0;
        let fahrenheit =
            UnitConverter::convert_temperature(celsius, Unit::Celsius, Unit::Fahrenheit).unwrap();
        assert!((fahrenheit - 212.0).abs() < 0.01);

        let kelvin =
            UnitConverter::convert_temperature(celsius, Unit::Celsius, Unit::Kelvin).unwrap();
        assert!((kelvin - 373.15).abs() < 0.01);
    }

    #[test]
    fn test_distance_conversion() {
        let meters = 1000.0;
        let km = UnitConverter::convert_distance(meters, Unit::Meter, Unit::Kilometer).unwrap();
        assert!((km - 1.0).abs() < 0.01);

        let miles = UnitConverter::convert_distance(1.0, Unit::Mile, Unit::Meter).unwrap();
        assert!((miles - 1609.34).abs() < 0.01);
    }

    #[test]
    fn test_weight_conversion() {
        let kg = 1.0;
        let pounds = UnitConverter::convert_weight(kg, Unit::Kilogram, Unit::Pound).unwrap();
        assert!((pounds - 2.20462).abs() < 0.01);

        let grams = UnitConverter::convert_weight(kg, Unit::Kilogram, Unit::Gram).unwrap();
        assert!((grams - 1000.0).abs() < 0.01);
    }
}

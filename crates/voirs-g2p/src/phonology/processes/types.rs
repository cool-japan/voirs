//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::{
    find_similar_phoneme, get_features, has_feature, Deserialize, PhonologicalFeature,
    PhonologicalProcess, ProcessType, Serialize,
};
use crate::{G2pError, LanguageCode, Phoneme, Result};

/// Nasalization process - vowels become nasalized before nasal consonants
///
/// This process is particularly prominent in:
/// - French: vowels before nasal consonants
/// - Portuguese: extensive vowel nasalization
/// - Polish: nasalized vowels
/// - English: slight nasalization in casual speech
pub struct NasalizationProcess {
    pub(super) language: LanguageCode,
}
impl NasalizationProcess {
    pub fn new(language: LanguageCode) -> Self {
        Self { language }
    }
    /// Check if nasalization should apply for the language
    pub(super) fn applies_for_language(&self) -> bool {
        matches!(self.language, LanguageCode::Fr | LanguageCode::Pt)
    }
    /// Check if a phoneme is a nasal consonant
    pub(super) fn is_nasal(&self, phoneme: &Phoneme) -> bool {
        matches!(
            phoneme.symbol.as_str(),
            "m" | "n" | "ŋ" | "ɲ" | "ɱ" | "ɳ" | "ɴ"
        )
    }
    /// Nasalize a vowel (add ~ diacritic)
    pub(super) fn nasalize_vowel(&self, phoneme: &Phoneme) -> Phoneme {
        let mut nasalized = phoneme.clone();
        if !nasalized.symbol.contains('̃') {
            nasalized.symbol = format!("{}̃", nasalized.symbol);
        }
        nasalized
    }
    /// Check if a phoneme is a vowel
    pub(super) fn is_vowel(&self, phoneme: &Phoneme) -> bool {
        has_feature(&phoneme.symbol, PhonologicalFeature::Vowel)
    }
}
/// Main phonological processor
pub struct PhonologicalProcessor {
    config: ProcessConfig,
    pub(super) processes: Vec<Box<dyn PhonologicalProcess>>,
}
impl PhonologicalProcessor {
    /// Create a new phonological processor
    pub fn new(language: LanguageCode) -> Self {
        let config = ProcessConfig {
            language,
            ..Default::default()
        };
        Self::with_config(config)
    }
    /// Create a processor with custom configuration
    pub fn with_config(config: ProcessConfig) -> Self {
        let mut processes: Vec<Box<dyn PhonologicalProcess>> = Vec::new();
        if config.enable_place_assimilation {
            processes.push(Box::new(AssimilationProcess::new(config.language)));
        }
        if config.enable_voicing_assimilation {
            processes.push(Box::new(VoicingAssimilationProcess::new(config.language)));
        }
        if config.enable_vowel_reduction {
            processes.push(Box::new(VowelReductionProcess::new(config.language)));
        }
        if config.enable_elision {
            processes.push(Box::new(ElisionProcess::new(config.language)));
        }
        if config.enable_liaison {
            processes.push(Box::new(LiaisonProcess::new(config.language)));
        }
        if config.enable_nasalization {
            processes.push(Box::new(NasalizationProcess::new(config.language)));
        }
        if config.enable_palatalization {
            processes.push(Box::new(PalatalizationProcess::new(config.language)));
        }
        if config.enable_lenition {
            processes.push(Box::new(LenitionProcess::new(config.language)));
        }
        if config.enable_fortition {
            processes.push(Box::new(FortitionProcess::new(config.language)));
        }
        if config.enable_r_dropping {
            processes.push(Box::new(RDroppingProcess::new(config.language)));
        }
        if config.enable_t_flapping {
            processes.push(Box::new(TFlappingProcess::new(config.language)));
        }
        if config.enable_final_devoicing {
            processes.push(Box::new(FinalDevoicingProcess::new(config.language)));
        }
        if config.enable_vowel_devoicing {
            processes.push(Box::new(VowelDevoicingProcess::new(config.language)));
        }
        if config.enable_h_dropping {
            processes.push(Box::new(HDroppingProcess::new(config.language)));
        }
        if config.enable_th_fronting {
            processes.push(Box::new(ThFrontingProcess::new(config.language)));
        }
        if config.enable_l_vocalization {
            processes.push(Box::new(LVocalizationProcess::new(config.language)));
        }
        if config.enable_g_dropping {
            processes.push(Box::new(GDroppingProcess::new(config.language)));
        }
        if config.enable_t_glottaling {
            processes.push(Box::new(TGlottalingProcess::new(config.language)));
        }
        if config.enable_yod_coalescence {
            processes.push(Box::new(YodCoalescenceProcess::new(config.language)));
        }
        Self { config, processes }
    }
    /// Apply all configured phonological processes
    pub fn apply_all_processes(&self, phonemes: &[Phoneme]) -> Result<Vec<Phoneme>> {
        let mut result = phonemes.to_vec();
        for process in &self.processes {
            result = process.apply(&result, &self.config)?;
        }
        Ok(result)
    }
    /// Apply a specific process type
    pub fn apply_process(
        &self,
        phonemes: &[Phoneme],
        process_type: ProcessType,
    ) -> Result<Vec<Phoneme>> {
        match process_type {
            ProcessType::PlaceAssimilation => {
                let process = AssimilationProcess::new(self.config.language);
                process.apply(phonemes, &self.config)
            }
            ProcessType::VoicingAssimilation => {
                let process = VoicingAssimilationProcess::new(self.config.language);
                process.apply(phonemes, &self.config)
            }
            ProcessType::VowelReduction => {
                let process = VowelReductionProcess::new(self.config.language);
                process.apply(phonemes, &self.config)
            }
            ProcessType::Elision => {
                let process = ElisionProcess::new(self.config.language);
                process.apply(phonemes, &self.config)
            }
            ProcessType::Liaison => {
                let process = LiaisonProcess::new(self.config.language);
                process.apply(phonemes, &self.config)
            }
            ProcessType::Nasalization => {
                let process = NasalizationProcess::new(self.config.language);
                process.apply(phonemes, &self.config)
            }
            ProcessType::Palatalization => {
                let process = PalatalizationProcess::new(self.config.language);
                process.apply(phonemes, &self.config)
            }
            ProcessType::Lenition => {
                let process = LenitionProcess::new(self.config.language);
                process.apply(phonemes, &self.config)
            }
            ProcessType::Fortition => {
                let process = FortitionProcess::new(self.config.language);
                process.apply(phonemes, &self.config)
            }
            ProcessType::RDropping => {
                let process = RDroppingProcess::new(self.config.language);
                process.apply(phonemes, &self.config)
            }
            ProcessType::TFlapping => {
                let process = TFlappingProcess::new(self.config.language);
                process.apply(phonemes, &self.config)
            }
            ProcessType::FinalDevoicing => {
                let process = FinalDevoicingProcess::new(self.config.language);
                process.apply(phonemes, &self.config)
            }
            ProcessType::VowelDevoicing => {
                let process = VowelDevoicingProcess::new(self.config.language);
                process.apply(phonemes, &self.config)
            }
            ProcessType::HDropping => {
                let process = HDroppingProcess::new(self.config.language);
                process.apply(phonemes, &self.config)
            }
            ProcessType::ThFronting => {
                let process = ThFrontingProcess::new(self.config.language);
                process.apply(phonemes, &self.config)
            }
            ProcessType::LVocalization => {
                let process = LVocalizationProcess::new(self.config.language);
                process.apply(phonemes, &self.config)
            }
            ProcessType::GDropping => {
                let process = GDroppingProcess::new(self.config.language);
                process.apply(phonemes, &self.config)
            }
            ProcessType::TGlottaling => {
                let process = TGlottalingProcess::new(self.config.language);
                process.apply(phonemes, &self.config)
            }
            ProcessType::YodCoalescence => {
                let process = YodCoalescenceProcess::new(self.config.language);
                process.apply(phonemes, &self.config)
            }
            _ => Err(G2pError::ConversionError(format!(
                "Process type {:?} not implemented",
                process_type
            ))),
        }
    }
}
/// Palatalization process - consonants become palatalized before front vowels or /j/
///
/// This process is particularly prominent in:
/// - Russian: extensive palatalization
/// - Polish: palatalization before front vowels
/// - Japanese: palatalization of /t, d/ before /i/
/// - English: palatalization in "tune" /tjuːn/ → /tʃuːn/
pub struct PalatalizationProcess {
    pub(super) language: LanguageCode,
}
impl PalatalizationProcess {
    pub fn new(language: LanguageCode) -> Self {
        Self { language }
    }
    /// Check if palatalization should apply for the language
    pub(super) fn applies_for_language(&self) -> bool {
        matches!(
            self.language,
            LanguageCode::Ja | LanguageCode::EnUs | LanguageCode::EnGb
        )
    }
    /// Check if a phoneme is a front vowel or palatal approximant
    pub(super) fn is_palatalizing_context(&self, phoneme: &Phoneme) -> bool {
        matches!(
            phoneme.symbol.as_str(),
            "i" | "iː" | "ɪ" | "e" | "eː" | "ɛ" | "j" | "y" | "ʏ" | "ø" | "œ"
        )
    }
    /// Palatalize a consonant
    pub(super) fn palatalize(&self, phoneme: &Phoneme) -> Option<Phoneme> {
        let mut palatalized = phoneme.clone();
        palatalized.symbol = match phoneme.symbol.as_str() {
            "t" => "tʲ".to_string(),
            "d" => "dʲ".to_string(),
            "k" => "kʲ".to_string(),
            "g" => "gʲ".to_string(),
            "s" => "ʃ".to_string(),
            "z" => "ʒ".to_string(),
            "n" => "ɲ".to_string(),
            "l" => "ʎ".to_string(),
            _ if phoneme.symbol.contains('ʲ') => return None,
            _ => return None,
        };
        Some(palatalized)
    }
}
/// Configuration for phonological processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessConfig {
    /// Enable place assimilation
    pub enable_place_assimilation: bool,
    /// Enable voicing assimilation
    pub enable_voicing_assimilation: bool,
    /// Enable vowel reduction
    pub enable_vowel_reduction: bool,
    /// Enable elision
    pub enable_elision: bool,
    /// Enable liaison
    pub enable_liaison: bool,
    /// Enable nasalization
    pub enable_nasalization: bool,
    /// Enable palatalization
    pub enable_palatalization: bool,
    /// Enable lenition
    pub enable_lenition: bool,
    /// Enable fortition
    pub enable_fortition: bool,
    /// Enable R-dropping (non-rhotic English dialects)
    pub enable_r_dropping: bool,
    /// Enable T-flapping (American English)
    pub enable_t_flapping: bool,
    /// Enable final devoicing (German)
    pub enable_final_devoicing: bool,
    /// Enable vowel devoicing (Japanese)
    pub enable_vowel_devoicing: bool,
    /// Enable H-dropping (British English dialects)
    pub enable_h_dropping: bool,
    /// Enable TH-fronting (London/Southern US English)
    pub enable_th_fronting: bool,
    /// Enable L-vocalization (London English, Polish)
    pub enable_l_vocalization: bool,
    /// Enable G-dropping (casual English -ing endings)
    pub enable_g_dropping: bool,
    /// Enable T-glottaling (British English)
    pub enable_t_glottaling: bool,
    /// Enable Yod-coalescence (American English)
    pub enable_yod_coalescence: bool,
    /// Language code
    pub language: LanguageCode,
    /// Aggressiveness of processes (0.0 = conservative, 1.0 = aggressive)
    pub aggressiveness: f32,
}
/// Vowel devoicing process - high vowels devoice between voiceless consonants
///
/// This process is particularly prominent in:
/// - Japanese: /i/ and /u/ devoice between voiceless consonants or after voiceless + before pause
/// - Very systematic and predictable in Japanese
/// - Examples: "sukiyaki" /sɯkijaki/ → [sɯ̥kijaki], "desu" /desɯ/ → [desɯ̥]
pub struct VowelDevoicingProcess {
    pub(super) language: LanguageCode,
}
impl VowelDevoicingProcess {
    pub fn new(language: LanguageCode) -> Self {
        Self { language }
    }
    /// Check if vowel devoicing should apply for the language
    pub(super) fn applies_for_language(&self) -> bool {
        matches!(self.language, LanguageCode::Ja)
    }
    /// Check if a phoneme is a high vowel (i or u)
    pub(super) fn is_high_vowel(&self, phoneme: &Phoneme) -> bool {
        matches!(phoneme.symbol.as_str(), "i" | "u" | "ɯ")
    }
    /// Check if a phoneme is voiceless
    pub(super) fn is_voiceless(&self, phoneme: &Phoneme) -> bool {
        matches!(
            phoneme.symbol.as_str(),
            "p" | "t" | "k" | "s" | "ʃ" | "h" | "f" | "θ" | "tʃ" | "ts"
        )
    }
    /// Devoice a vowel (add ring diacritic)
    pub(super) fn devoice_vowel(&self, phoneme: &Phoneme) -> Phoneme {
        let mut devoiced = phoneme.clone();
        if !devoiced.symbol.contains('̥') {
            devoiced.symbol = format!("{}̥", devoiced.symbol);
        }
        devoiced
    }
}
/// Fortition process - consonants strengthen in certain positions
///
/// This process is particularly prominent in:
/// - Italian: gemination (consonant doubling) for emphasis
/// - Finnish: gemination as phonemic feature
/// - Japanese: gemination (促音 sokuon)
/// - English: fortis consonants in stressed positions
pub struct FortitionProcess {
    pub(super) language: LanguageCode,
}
impl FortitionProcess {
    pub fn new(language: LanguageCode) -> Self {
        Self { language }
    }
    /// Check if fortition should apply for the language
    pub(super) fn applies_for_language(&self) -> bool {
        matches!(self.language, LanguageCode::It | LanguageCode::Ja)
    }
    /// Check if a phoneme is stressed
    pub(super) fn is_stressed(&self, phoneme: &Phoneme) -> bool {
        phoneme.stress > 0
    }
    /// Strengthen (fortify) a consonant through gemination
    pub(super) fn fortify(&self, phoneme: &Phoneme) -> Option<Phoneme> {
        let mut fortified = phoneme.clone();
        match phoneme.symbol.as_str() {
            "p" | "t" | "k" | "b" | "d" | "g" | "f" | "v" | "s" | "z" | "ʃ" | "ʒ" | "m" | "n"
            | "l" | "r" => {
                fortified.symbol = format!("{}{}", phoneme.symbol, phoneme.symbol);
                Some(fortified)
            }
            _ if phoneme.symbol.len() > 2 && {
                let chars: Vec<char> = phoneme.symbol.chars().collect();
                chars.len() >= 2 && chars[0] == chars[1]
            } =>
            {
                None
            }
            _ => None,
        }
    }
    /// Check if a phoneme is a vowel
    pub(super) fn is_vowel(&self, phoneme: &Phoneme) -> bool {
        has_feature(&phoneme.symbol, PhonologicalFeature::Vowel)
    }
}
/// Vowel reduction process
pub struct VowelReductionProcess {
    pub(super) language: LanguageCode,
}
impl VowelReductionProcess {
    pub fn new(language: LanguageCode) -> Self {
        Self { language }
    }
    /// Check if a vowel should be reduced
    pub(super) fn should_reduce(&self, phoneme: &Phoneme) -> bool {
        phoneme.stress == 0
            && has_feature(&phoneme.symbol, PhonologicalFeature::Vowel)
            && phoneme.symbol != "ə"
    }
    /// Get the reduced form of a vowel
    pub(super) fn reduce_vowel(&self, phoneme: &str) -> String {
        match self.language {
            LanguageCode::EnUs | LanguageCode::EnGb => "ə".to_string(),
            _ => phoneme.to_string(),
        }
    }
}
/// Place assimilation process (e.g., "input" /ɪnpʊt/ -> /ɪmpʊt/)
pub struct AssimilationProcess {
    pub(super) language: LanguageCode,
}
impl AssimilationProcess {
    pub fn new(language: LanguageCode) -> Self {
        Self { language }
    }
    /// Check if assimilation should apply
    pub(super) fn should_assimilate(&self, current: &str, next: &str) -> bool {
        use PhonologicalFeature::*;
        if has_feature(current, Nasal) && has_feature(next, Stop) {
            let current_features = get_features(current);
            let next_features = get_features(next);
            if current_features.contains(&Alveolar)
                && (next_features.contains(&Bilabial) || next_features.contains(&Labiodental))
            {
                return true;
            }
            if current_features.contains(&Alveolar) && next_features.contains(&Velar) {
                return true;
            }
        }
        false
    }
    /// Find the assimilated phoneme
    pub(super) fn get_assimilated_phoneme(&self, current: &str, next: &str) -> Option<String> {
        let next_features = get_features(next);
        if next_features.contains(&PhonologicalFeature::Bilabial) {
            find_similar_phoneme(
                current,
                PhonologicalFeature::Alveolar,
                PhonologicalFeature::Bilabial,
            )
        } else if next_features.contains(&PhonologicalFeature::Velar) {
            find_similar_phoneme(
                current,
                PhonologicalFeature::Alveolar,
                PhonologicalFeature::Velar,
            )
        } else {
            None
        }
    }
}
/// Lenition process - consonants weaken in certain positions
///
/// This process is particularly prominent in:
/// - Spanish: /b, d, g/ → [β, ð, ɣ] between vowels
/// - Irish: extensive lenition system
/// - Portuguese: weakening of intervocalic stops
/// - English: flapping of /t, d/ in American English
pub struct LenitionProcess {
    pub(super) language: LanguageCode,
}
impl LenitionProcess {
    pub fn new(language: LanguageCode) -> Self {
        Self { language }
    }
    /// Check if lenition should apply for the language
    pub(super) fn applies_for_language(&self) -> bool {
        matches!(self.language, LanguageCode::Es | LanguageCode::Pt)
    }
    /// Check if a phoneme is a vowel
    pub(super) fn is_vowel(&self, phoneme: &Phoneme) -> bool {
        has_feature(&phoneme.symbol, PhonologicalFeature::Vowel)
    }
    /// Weaken (lenite) a consonant
    pub(super) fn lenite(&self, phoneme: &Phoneme) -> Option<Phoneme> {
        let mut lenited = phoneme.clone();
        lenited.symbol = match phoneme.symbol.as_str() {
            "b" => "β".to_string(),
            "d" => "ð".to_string(),
            "g" => "ɣ".to_string(),
            "p" => "ɸ".to_string(),
            "t" => "θ".to_string(),
            "k" => "x".to_string(),
            _ if matches!(phoneme.symbol.as_str(), "β" | "ð" | "ɣ" | "ɸ" | "θ" | "x") => {
                return None
            }
            _ => return None,
        };
        Some(lenited)
    }
}
/// R-dropping process - R is not pronounced in non-rhotic dialects
///
/// This process is particularly prominent in:
/// - British RP (Received Pronunciation): R dropped unless followed by vowel
/// - New England American English: Similar pattern to British English
/// - New York City (some speakers): R-dropping in certain contexts
/// - Southern Hemisphere English (Australia, NZ): Non-rhotic in some varieties
pub struct RDroppingProcess {
    pub(super) language: LanguageCode,
}
impl RDroppingProcess {
    pub fn new(language: LanguageCode) -> Self {
        Self { language }
    }
    /// Check if R-dropping should apply for the language
    pub(super) fn applies_for_language(&self) -> bool {
        matches!(self.language, LanguageCode::EnGb)
    }
    /// Check if a phoneme is R or R-colored
    pub(super) fn is_r_phoneme(&self, phoneme: &Phoneme) -> bool {
        matches!(phoneme.symbol.as_str(), "r" | "ɹ" | "ɚ" | "ɝ")
    }
    /// Check if a phoneme is a vowel
    pub(super) fn is_vowel(&self, phoneme: &Phoneme) -> bool {
        has_feature(&phoneme.symbol, PhonologicalFeature::Vowel)
    }
    /// Should drop R in this context
    pub(super) fn should_drop_r(&self, phonemes: &[Phoneme], index: usize) -> bool {
        if index + 1 < phonemes.len() && self.is_vowel(&phonemes[index + 1]) {
            return false;
        }
        true
    }
}
/// Final devoicing process - voiced obstruents become voiceless at word end
///
/// This process is particularly prominent in:
/// - German (Auslautverhärtung): Systematic final devoicing
/// - Dutch: Similar to German
/// - Russian: Final devoicing before pauses
/// - Polish: Final devoicing
/// - Examples: German "Hund" \[hunt\] (not \[hund\]), "Tag" \[tak\] (not \[tag\])
pub struct FinalDevoicingProcess {
    pub(super) language: LanguageCode,
}
impl FinalDevoicingProcess {
    pub fn new(language: LanguageCode) -> Self {
        Self { language }
    }
    /// Check if final devoicing should apply for the language
    pub(super) fn applies_for_language(&self) -> bool {
        matches!(self.language, LanguageCode::De)
    }
    /// Devoice a voiced consonant
    pub(super) fn devoice(&self, phoneme: &Phoneme) -> Option<Phoneme> {
        let mut devoiced = phoneme.clone();
        devoiced.symbol = match phoneme.symbol.as_str() {
            "b" => "p".to_string(),
            "d" => "t".to_string(),
            "g" => "k".to_string(),
            "v" => "f".to_string(),
            "z" => "s".to_string(),
            "ʒ" => "ʃ".to_string(),
            "ʤ" | "dʒ" => "tʃ".to_string(),
            _ if matches!(
                phoneme.symbol.as_str(),
                "p" | "t" | "k" | "f" | "s" | "ʃ" | "tʃ"
            ) =>
            {
                return None
            }
            _ => return None,
        };
        Some(devoiced)
    }
    /// Check if this is a voiced obstruent
    pub(super) fn is_voiced_obstruent(&self, phoneme: &Phoneme) -> bool {
        matches!(
            phoneme.symbol.as_str(),
            "b" | "d" | "g" | "v" | "z" | "ʒ" | "ʤ" | "dʒ"
        )
    }
}
/// Liaison (linking) process
pub struct LiaisonProcess {
    pub(super) language: LanguageCode,
}
impl LiaisonProcess {
    pub fn new(language: LanguageCode) -> Self {
        Self { language }
    }
}
/// Elision (sound deletion) process
pub struct ElisionProcess {
    pub(super) language: LanguageCode,
}
impl ElisionProcess {
    pub fn new(language: LanguageCode) -> Self {
        Self { language }
    }
    /// Check if a phoneme can be elided
    pub(super) fn can_elide(
        &self,
        phoneme: &Phoneme,
        position: usize,
        phonemes: &[Phoneme],
    ) -> bool {
        if position == 0 || position == phonemes.len() - 1 {
            return false;
        }
        if phoneme.stress > 0 {
            return false;
        }
        match self.language {
            LanguageCode::EnUs | LanguageCode::EnGb => {
                if phoneme.symbol == "ə" {
                    let prev = &phonemes[position - 1].symbol;
                    let next = &phonemes[position + 1].symbol;
                    !has_feature(prev, PhonologicalFeature::Vowel)
                        && !has_feature(next, PhonologicalFeature::Vowel)
                } else {
                    false
                }
            }
            LanguageCode::Fr => phoneme.symbol == "ə" && position > 0,
            _ => false,
        }
    }
}
/// T-flapping process - /t/ becomes flap \[ɾ\] in American English
///
/// This process is particularly prominent in:
/// - American English: /t/ → \[ɾ\] between vowels
/// - Canadian English: Similar to American
/// - Conditions: intervocalic /t/ with following unstressed syllable
/// - Examples: "better" /bɛtɚ/ → \[bɛɾɚ\], "water" /wɔtɚ/ → \[wɔɾɚ\]
pub struct TFlappingProcess {
    pub(super) language: LanguageCode,
}
impl TFlappingProcess {
    pub fn new(language: LanguageCode) -> Self {
        Self { language }
    }
    /// Check if T-flapping should apply for the language
    pub(super) fn applies_for_language(&self) -> bool {
        matches!(self.language, LanguageCode::EnUs)
    }
    /// Check if a phoneme is /t/ or /d/
    pub(super) fn is_flappable(&self, phoneme: &Phoneme) -> bool {
        matches!(phoneme.symbol.as_str(), "t" | "d")
    }
    /// Check if a phoneme is a vowel (including R-colored vowels)
    pub(super) fn is_vowel(&self, phoneme: &Phoneme) -> bool {
        has_feature(&phoneme.symbol, PhonologicalFeature::Vowel)
            || matches!(phoneme.symbol.as_str(), "ɚ" | "ɝ")
    }
    /// Convert /t/ or /d/ to flap [ɾ]
    pub(super) fn to_flap(&self, _phoneme: &Phoneme) -> Phoneme {
        Phoneme::new("ɾ".to_string())
    }
}
/// Voicing assimilation process
pub struct VoicingAssimilationProcess {
    pub(super) language: LanguageCode,
}
impl VoicingAssimilationProcess {
    pub fn new(language: LanguageCode) -> Self {
        Self { language }
    }
    /// Check if voicing assimilation should apply
    pub(super) fn should_voice_assimilate(&self, current: &str, next: &str) -> bool {
        use PhonologicalFeature::*;
        if !has_feature(current, Stop) && !has_feature(current, Fricative) {
            return false;
        }
        has_feature(next, Voiced) && !has_feature(current, Voiced)
    }
    /// Get the voiced counterpart of a voiceless obstruent
    pub(super) fn get_voiced_counterpart(&self, phoneme: &str) -> Option<String> {
        match phoneme {
            "p" => Some("b".to_string()),
            "t" => Some("d".to_string()),
            "k" => Some("g".to_string()),
            "f" => Some("v".to_string()),
            "θ" => Some("ð".to_string()),
            "s" => Some("z".to_string()),
            "ʃ" => Some("ʒ".to_string()),
            _ => None,
        }
    }
}

/// H-dropping process - /h/ deletion in unstressed syllables
///
/// This process is particularly prominent in:
/// - Cockney English: H-dropping at the beginning of words and syllables
/// - Yorkshire English: Similar to Cockney
/// - Australian English (some varieties): H-dropping in casual speech
/// - New York City (some speakers): H-dropping in certain contexts
/// - Examples: "house" /haʊs/ → /aʊs/, "behind" /bɪhaɪnd/ → /bɪaɪnd/
pub struct HDroppingProcess {
    pub(super) language: LanguageCode,
}

impl HDroppingProcess {
    pub fn new(language: LanguageCode) -> Self {
        Self { language }
    }

    /// Check if H-dropping should apply for the language
    pub(super) fn applies_for_language(&self) -> bool {
        // H-dropping is characteristic of certain British English dialects
        matches!(self.language, LanguageCode::EnGb)
    }

    /// Check if a phoneme is /h/
    pub(super) fn is_h_phoneme(&self, phoneme: &Phoneme) -> bool {
        matches!(phoneme.symbol.as_str(), "h")
    }

    /// Check if a phoneme is stressed (H-dropping typically doesn't occur in stressed syllables)
    pub(super) fn is_stressed(&self, phoneme: &Phoneme) -> bool {
        phoneme.stress > 0
    }

    /// Should drop H in this context
    pub(super) fn should_drop_h(
        &self,
        phoneme: &Phoneme,
        _index: usize,
        _phonemes: &[Phoneme],
    ) -> bool {
        // H-dropping typically occurs in unstressed syllables
        self.is_h_phoneme(phoneme) && !self.is_stressed(phoneme)
    }
}

/// TH-fronting process - /θ/ → /f/, /ð/ → /v/
///
/// This process is particularly prominent in:
/// - London English (Cockney, Multicultural London English): Systematic TH-fronting
/// - Southern US English (some varieties): Variable TH-fronting
/// - New York City (some speakers): TH-fronting in casual speech
/// - Examples: "think" /θɪŋk/ → /fɪŋk/, "this" /ðɪs/ → /vɪs/
pub struct ThFrontingProcess {
    pub(super) language: LanguageCode,
}

impl ThFrontingProcess {
    pub fn new(language: LanguageCode) -> Self {
        Self { language }
    }

    /// Check if TH-fronting should apply for the language
    pub(super) fn applies_for_language(&self) -> bool {
        // TH-fronting is characteristic of London English and some US varieties
        matches!(self.language, LanguageCode::EnGb | LanguageCode::EnUs)
    }

    /// Check if a phoneme is dental fricative (/θ/ or /ð/)
    pub(super) fn is_dental_fricative(&self, phoneme: &Phoneme) -> bool {
        matches!(phoneme.symbol.as_str(), "θ" | "ð")
    }

    /// Convert dental fricative to labiodental fricative
    pub(super) fn front_dental(&self, phoneme: &Phoneme) -> Phoneme {
        let mut fronted = phoneme.clone();
        fronted.symbol = match phoneme.symbol.as_str() {
            "θ" => "f".to_string(),
            "ð" => "v".to_string(),
            _ => phoneme.symbol.clone(),
        };
        fronted
    }
}

/// L-vocalization process - syllable-final /l/ → /w/ or /ʊ/
///
/// This process is particularly prominent in:
/// - London English (Estuary English, Cockney): Systematic L-vocalization
/// - Brazilian Portuguese: L-vocalization in coda position
/// - American English (some Southern varieties): L-vocalization
/// - Examples: "milk" /mɪlk/ → /mɪwk/, "feel" /fiːl/ → /fiːw/
pub struct LVocalizationProcess {
    pub(super) language: LanguageCode,
}

impl LVocalizationProcess {
    pub fn new(language: LanguageCode) -> Self {
        Self { language }
    }

    /// Check if L-vocalization should apply for the language
    pub(super) fn applies_for_language(&self) -> bool {
        // L-vocalization is characteristic of London English and Portuguese
        matches!(self.language, LanguageCode::EnGb | LanguageCode::Pt)
    }

    /// Check if a phoneme is /l/
    pub(super) fn is_l_phoneme(&self, phoneme: &Phoneme) -> bool {
        matches!(phoneme.symbol.as_str(), "l" | "ɫ")
    }

    /// Check if /l/ is in coda position (before consonant or word-finally)
    pub(super) fn is_coda_position(&self, index: usize, phonemes: &[Phoneme]) -> bool {
        // Check if followed by consonant or at end of word
        if index + 1 >= phonemes.len() {
            return true; // Word-final
        }

        let next = &phonemes[index + 1];
        !has_feature(&next.symbol, PhonologicalFeature::Vowel)
    }

    /// Vocalize /l/ to /w/
    pub(super) fn vocalize_l(&self, _phoneme: &Phoneme) -> Phoneme {
        Phoneme::new("w".to_string())
    }
}

/// G-dropping process - /ŋ/ → /n/ in unstressed syllables
///
/// This process is particularly prominent in:
/// - Casual English (all varieties): G-dropping in -ing endings
/// - Southern US English: More frequent G-dropping
/// - Informal speech across English dialects
/// - Examples: "running" /rʌnɪŋ/ → /rʌnɪn/, "walking" /wɔːkɪŋ/ → /wɔːkɪn/
pub struct GDroppingProcess {
    pub(super) language: LanguageCode,
}

impl GDroppingProcess {
    pub fn new(language: LanguageCode) -> Self {
        Self { language }
    }

    /// Check if G-dropping should apply for the language
    pub(super) fn applies_for_language(&self) -> bool {
        // G-dropping occurs in all English varieties in casual speech
        matches!(self.language, LanguageCode::EnUs | LanguageCode::EnGb)
    }

    /// Check if a phoneme is /ŋ/
    pub(super) fn is_velar_nasal(&self, phoneme: &Phoneme) -> bool {
        matches!(phoneme.symbol.as_str(), "ŋ")
    }

    /// Check if this is in an unstressed syllable (typical for -ing endings)
    pub(super) fn is_unstressed(&self, phoneme: &Phoneme) -> bool {
        phoneme.stress == 0
    }

    /// Convert /ŋ/ to /n/
    pub(super) fn drop_g(&self, _phoneme: &Phoneme) -> Phoneme {
        Phoneme::new("n".to_string())
    }
}

/// T-glottaling process - /t/ → /ʔ/ in coda position
///
/// This process is particularly prominent in:
/// - British English (widespread): T-glottaling in coda position
/// - London English: Very frequent glottalization
/// - Scottish English: T-glottaling common
/// - Estuary English: Characteristic feature
/// - Examples: "butter" /bʌtə/ → /bʌʔə/, "bottle" /bɒtl̩/ → /bɒʔl̩/
pub struct TGlottalingProcess {
    pub(super) language: LanguageCode,
}

impl TGlottalingProcess {
    pub fn new(language: LanguageCode) -> Self {
        Self { language }
    }

    /// Check if T-glottaling should apply for the language
    pub(super) fn applies_for_language(&self) -> bool {
        // T-glottaling is very characteristic of British English
        matches!(self.language, LanguageCode::EnGb)
    }

    /// Check if a phoneme is /t/
    pub(super) fn is_t_phoneme(&self, phoneme: &Phoneme) -> bool {
        matches!(phoneme.symbol.as_str(), "t")
    }

    /// Check if /t/ is in coda position (before consonant or intervocalic before unstressed vowel)
    pub(super) fn is_coda_or_intervocalic_unstressed(
        &self,
        index: usize,
        phonemes: &[Phoneme],
    ) -> bool {
        // Word-final
        if index + 1 >= phonemes.len() {
            return true;
        }

        let next = &phonemes[index + 1];

        // Before consonant
        if !has_feature(&next.symbol, PhonologicalFeature::Vowel) {
            return true;
        }

        // Intervocalic before unstressed vowel
        if has_feature(&next.symbol, PhonologicalFeature::Vowel) && next.stress == 0 {
            return true;
        }

        false
    }

    /// Convert /t/ to glottal stop /ʔ/
    pub(super) fn glottalize_t(&self, _phoneme: &Phoneme) -> Phoneme {
        Phoneme::new("ʔ".to_string())
    }
}

/// Yod-coalescence process - /tj/ → /tʃ/, /dj/ → /dʒ/
///
/// This process is particularly prominent in:
/// - American English: Frequent yod-coalescence (tune /tjuːn/ → /tʃuːn/)
/// - Some British English varieties: Variable yod-coalescence
/// - Examples: "tune" /tjuːn/ → /tʃuːn/, "dune" /djuːn/ → /dʒuːn/
pub struct YodCoalescenceProcess {
    pub(super) language: LanguageCode,
}

impl YodCoalescenceProcess {
    pub fn new(language: LanguageCode) -> Self {
        Self { language }
    }

    /// Check if yod-coalescence should apply for the language
    pub(super) fn applies_for_language(&self) -> bool {
        // Yod-coalescence is more common in American English
        matches!(self.language, LanguageCode::EnUs)
    }

    /// Check if a phoneme is /t/ or /d/
    pub(super) fn is_alveolar_stop(&self, phoneme: &Phoneme) -> bool {
        matches!(phoneme.symbol.as_str(), "t" | "d")
    }

    /// Check if a phoneme is /j/ (yod)
    pub(super) fn is_yod(&self, phoneme: &Phoneme) -> bool {
        matches!(phoneme.symbol.as_str(), "j" | "y")
    }

    /// Convert /tj/ to /tʃ/ or /dj/ to /dʒ/
    pub(super) fn coalesce(&self, stop: &Phoneme, _yod: &Phoneme) -> Phoneme {
        let coalesced = match stop.symbol.as_str() {
            "t" => "tʃ",
            "d" => "dʒ",
            _ => return stop.clone(),
        };
        Phoneme::new(coalesced.to_string())
    }
}

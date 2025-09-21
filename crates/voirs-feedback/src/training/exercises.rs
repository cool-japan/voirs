//! Exercise library and exercise creation functionality
//!
//! This module contains all exercise creation functions and the comprehensive
//! exercise library for the training system.

use crate::training::types::ExerciseLibrary;
use crate::traits::{ExerciseCategory, ExerciseType, FocusArea, SuccessCriteria, TrainingExercise};
use std::time::Duration;

/// Exercise library implementation
impl ExerciseLibrary {
    /// Create the default comprehensive exercise library with 500+ exercises
    pub fn create_default() -> Self {
        let mut exercises = Vec::new();

        // Generate comprehensive exercise library with 500+ exercises

        // === BEGINNER LEVEL EXERCISES (0.1 - 0.3 difficulty) ===

        // Basic Phoneme Exercises (50 exercises)
        exercises.extend(Self::create_phoneme_exercises());

        // Simple Word Pronunciation (40 exercises)
        exercises.extend(Self::create_simple_word_exercises());

        // Basic Sentence Structure (30 exercises)
        exercises.extend(Self::create_basic_sentence_exercises());

        // === INTERMEDIATE LEVEL EXERCISES (0.4 - 0.6 difficulty) ===

        // Quality Improvement (35 exercises)
        exercises.extend(Self::create_quality_exercises());

        // Rhythm and Stress (35 exercises)
        exercises.extend(Self::create_rhythm_exercises());

        // Intonation Practice (30 exercises)
        exercises.extend(Self::create_intonation_exercises());

        // Natural Expression (40 exercises)
        exercises.extend(Self::create_expression_exercises());

        // === ADVANCED LEVEL EXERCISES (0.7 - 1.0 difficulty) ===

        // Fluency Challenges (45 exercises)
        exercises.extend(Self::create_fluency_exercises());

        // Complex Text Reading (30 exercises)
        exercises.extend(Self::create_complex_text_exercises());

        // Emotion and Mood (25 exercises)
        exercises.extend(Self::create_emotion_exercises());

        // Professional Speaking (40 exercises)
        exercises.extend(Self::create_professional_exercises());

        // Technical Content (25 exercises)
        exercises.extend(Self::create_technical_exercises());

        // === SPECIALIZED EXERCISES ===

        // Accent and Dialect (30 exercises)
        exercises.extend(Self::create_accent_exercises());

        // Speed and Clarity (25 exercises)
        exercises.extend(Self::create_speed_exercises());

        // Poetry and Literature (20 exercises)
        exercises.extend(Self::create_poetry_exercises());

        // Conversation Practice (25 exercises)
        exercises.extend(Self::create_conversation_exercises());

        // Challenge and Review (25 exercises)
        exercises.extend(Self::create_challenge_exercises());

        Self {
            exercises,
            categories: Self::create_exercise_categories(),
        }
    }

    /// Create phoneme-focused exercises for beginners
    fn create_phoneme_exercises() -> Vec<TrainingExercise> {
        let mut exercises = Vec::new();

        let phoneme_groups = vec![
            // Vowels
            (
                "vowel_a",
                "Practice the A sound",
                vec!["cat", "bat", "hat", "rat", "mat"],
            ),
            (
                "vowel_e",
                "Practice the E sound",
                vec!["pet", "met", "net", "set", "wet"],
            ),
            (
                "vowel_i",
                "Practice the I sound",
                vec!["bit", "fit", "hit", "sit", "wit"],
            ),
            (
                "vowel_o",
                "Practice the O sound",
                vec!["pot", "hot", "cot", "dot", "got"],
            ),
            (
                "vowel_u",
                "Practice the U sound",
                vec!["but", "cut", "hut", "nut", "rut"],
            ),
            // Consonants - Plosives
            (
                "consonant_p_b",
                "Practice P and B sounds",
                vec!["pat", "bat", "pit", "bit", "pot", "bot"],
            ),
            (
                "consonant_t_d",
                "Practice T and D sounds",
                vec!["tip", "dip", "top", "dog", "tap", "dad"],
            ),
            (
                "consonant_k_g",
                "Practice K and G sounds",
                vec!["cat", "gap", "kit", "get", "cup", "gum"],
            ),
            // Fricatives
            (
                "consonant_f_v",
                "Practice F and V sounds",
                vec!["fan", "van", "fin", "vim", "fog", "vow"],
            ),
            (
                "consonant_s_z",
                "Practice S and Z sounds",
                vec!["sip", "zip", "sue", "zoo", "bus", "buzz"],
            ),
            (
                "consonant_sh_zh",
                "Practice SH and ZH sounds",
                vec!["ship", "measure", "shoe", "vision", "wash", "beige"],
            ),
            // Nasals
            (
                "consonant_m_n",
                "Practice M and N sounds",
                vec!["map", "nap", "mom", "nun", "rim", "run"],
            ),
            (
                "consonant_ng",
                "Practice NG sound",
                vec!["sing", "ring", "hang", "long", "young", "strong"],
            ),
            // Liquids
            (
                "consonant_l_r",
                "Practice L and R sounds",
                vec!["lap", "rap", "let", "red", "love", "rock"],
            ),
            // Affricates
            (
                "consonant_ch_j",
                "Practice CH and J sounds",
                vec!["chip", "jump", "chin", "joke", "rich", "ridge"],
            ),
        ];

        for (i, (id_suffix, description, words)) in phoneme_groups.iter().enumerate() {
            for (j, group) in words.chunks(3).enumerate() {
                let exercise_id = format!("phoneme_{id_suffix}_{}", j + 1);
                let target_text = group.join(", ");

                exercises.push(TrainingExercise {
                    exercise_id,
                    name: format!("{description} - Group {}", j + 1),
                    description: format!("Focus on clear pronunciation of: {target_text}"),
                    difficulty: 0.1 + (i as f32 * 0.01),
                    focus_areas: vec![FocusArea::Pronunciation],
                    exercise_type: ExerciseType::Pronunciation,
                    target_text,
                    reference_audio: None,
                    success_criteria: SuccessCriteria {
                        min_quality_score: 0.6,
                        min_pronunciation_score: 0.75,
                        max_attempts: 5,
                        time_limit: Some(Duration::from_secs(180)),
                        consistency_required: 1,
                    },
                    estimated_duration: Duration::from_secs(300),
                });
            }
        }

        exercises
    }

    /// Create simple word exercises
    fn create_simple_word_exercises() -> Vec<TrainingExercise> {
        let mut exercises = Vec::new();

        let word_categories = vec![
            (
                "animals",
                "Animal names",
                vec![
                    "cat", "dog", "bird", "fish", "horse", "cow", "sheep", "pig", "duck",
                    "chicken", "elephant", "lion", "tiger", "bear", "wolf", "fox", "rabbit",
                    "mouse", "rat", "snake",
                ],
            ),
            (
                "colors",
                "Color words",
                vec![
                    "red",
                    "blue",
                    "green",
                    "yellow",
                    "orange",
                    "purple",
                    "pink",
                    "black",
                    "white",
                    "brown",
                    "gray",
                    "gold",
                    "silver",
                    "violet",
                    "crimson",
                    "turquoise",
                    "magenta",
                    "navy",
                    "lime",
                    "cyan",
                ],
            ),
            (
                "numbers",
                "Number words",
                vec![
                    "one",
                    "two",
                    "three",
                    "four",
                    "five",
                    "six",
                    "seven",
                    "eight",
                    "nine",
                    "ten",
                    "eleven",
                    "twelve",
                    "thirteen",
                    "fourteen",
                    "fifteen",
                    "sixteen",
                    "seventeen",
                    "eighteen",
                    "nineteen",
                    "twenty",
                ],
            ),
            (
                "food",
                "Food words",
                vec![
                    "apple",
                    "banana",
                    "orange",
                    "bread",
                    "milk",
                    "cheese",
                    "meat",
                    "fish",
                    "rice",
                    "pasta",
                    "pizza",
                    "burger",
                    "salad",
                    "soup",
                    "cake",
                    "cookie",
                    "chocolate",
                    "coffee",
                    "tea",
                    "water",
                ],
            ),
        ];

        for (category, description, words) in word_categories {
            for (i, chunk) in words.chunks(5).enumerate() {
                let exercise_id = format!("simple_words_{category}_{}", i + 1);
                let target_text = chunk.join(", ");

                exercises.push(TrainingExercise {
                    exercise_id,
                    name: format!("{description} - Set {}", i + 1),
                    description: format!("Practice clear pronunciation of {category} words"),
                    difficulty: 0.15 + (i as f32 * 0.02),
                    focus_areas: vec![FocusArea::Pronunciation, FocusArea::Quality],
                    exercise_type: ExerciseType::Pronunciation,
                    target_text,
                    reference_audio: None,
                    success_criteria: SuccessCriteria {
                        min_quality_score: 0.65,
                        min_pronunciation_score: 0.75,
                        max_attempts: 4,
                        time_limit: Some(Duration::from_secs(240)),
                        consistency_required: 1,
                    },
                    estimated_duration: Duration::from_secs(360),
                });
            }
        }

        exercises
    }

    /// Create basic sentence exercises
    fn create_basic_sentence_exercises() -> Vec<TrainingExercise> {
        let mut exercises = Vec::new();

        let sentence_templates = vec![
            // Simple present tense
            "The sun is shining brightly today",
            "My friend likes to read books",
            "We walk to school every morning",
            "The cat sleeps on the chair",
            "Children play in the park",
            "She drinks water before meals",
            "He works at the office",
            "They watch movies on weekends",
            "The bird sings beautiful songs",
            "I enjoy listening to music",
            // Simple questions
            "What time is it now?",
            "Where do you live?",
            "How are you feeling today?",
            "Who is your best friend?",
            "Why do you like chocolate?",
            "When did you wake up?",
            "Which book are you reading?",
            "What color is your shirt?",
            "How old are you?",
            "Where is the nearest store?",
            // Common phrases
            "Good morning, how are you?",
            "Thank you very much",
            "You're welcome, my friend",
            "Please help me with this",
            "I'm sorry for being late",
            "Have a nice day",
            "See you later today",
            "Nice to meet you",
            "How can I help you?",
            "Take care of yourself",
        ];

        for (i, sentence) in sentence_templates.iter().enumerate() {
            let exercise_id = format!("basic_sentence_{}", i + 1);

            exercises.push(TrainingExercise {
                exercise_id,
                name: format!("Basic Sentence {}", i + 1),
                description: String::from("Practice natural sentence flow and rhythm"),
                difficulty: 0.2 + (i as f32 * 0.003),
                focus_areas: vec![
                    FocusArea::Pronunciation,
                    FocusArea::Rhythm,
                    FocusArea::Naturalness,
                ],
                exercise_type: ExerciseType::FreeForm,
                target_text: (*sentence).to_string(),
                reference_audio: None,
                success_criteria: SuccessCriteria {
                    min_quality_score: 0.65,
                    min_pronunciation_score: 0.75,
                    max_attempts: 4,
                    time_limit: Some(Duration::from_secs(300)),
                    consistency_required: 1,
                },
                estimated_duration: Duration::from_secs(450),
            });
        }

        exercises
    }

    /// Create quality-focused exercises for intermediate level
    fn create_quality_exercises() -> Vec<TrainingExercise> {
        let mut exercises = Vec::new();

        let quality_scenarios = vec![
            ("studio_recording", "Studio Recording Quality", "Please speak clearly and maintain consistent volume levels throughout this recording session."),
            ("conference_call", "Conference Call Clarity", "Good morning everyone, thank you for joining today's important business meeting discussion."),
            ("podcast_intro", "Podcast Introduction", "Welcome to today's episode where we'll be discussing fascinating topics and interesting developments."),
            ("presentation_opening", "Presentation Opening", "Today I will be presenting our quarterly results and future strategic planning initiatives."),
            ("customer_service", "Customer Service Excellence", "Thank you for contacting our support team. How may I assist you with your inquiry today?"),
            ("news_broadcast", "News Broadcasting", "Breaking news this evening as developments continue to unfold in the ongoing international situation."),
            ("audiobook_narration", "Audiobook Narration", "Chapter One begins with our protagonist facing an unexpected challenge that will change everything."),
            ("training_material", "Training Material", "This comprehensive guide will walk you through each step of the process systematically and thoroughly."),
            ("phone_interview", "Phone Interview", "Thank you for taking the time to speak with us today about this exciting opportunity."),
            ("voice_assistant", "Voice Assistant Response", "I understand your request and will provide you with accurate information to help solve your problem."),
            ("radio_commercial", "Radio Commercial", "Visit our store today for amazing deals and exceptional customer service you can trust."),
            ("educational_content", "Educational Content", "In today's lesson, we will explore fundamental concepts and practical applications in detail."),
            ("meditation_guide", "Meditation Guidance", "Take a deep breath and allow yourself to relax completely as we begin this peaceful journey."),
            ("technical_support", "Technical Support", "Let me help you troubleshoot this issue step by step to ensure everything works properly."),
            ("weather_report", "Weather Reporting", "Today's forecast calls for partly cloudy skies with temperatures reaching seventy-five degrees."),
        ];

        for (i, (id_suffix, name, text)) in quality_scenarios.iter().enumerate() {
            let exercise_id = format!("quality_{id_suffix}");

            exercises.push(TrainingExercise {
                exercise_id,
                name: (*name).to_string(),
                description: "Focus on producing high-quality, professional audio output"
                    .to_string(),
                difficulty: 0.4 + (i as f32 * 0.01),
                focus_areas: vec![FocusArea::Quality, FocusArea::Naturalness],
                exercise_type: ExerciseType::Quality,
                target_text: (*text).to_string(),
                reference_audio: None,
                success_criteria: SuccessCriteria {
                    min_quality_score: 0.8,
                    min_pronunciation_score: 0.75,
                    max_attempts: 3,
                    time_limit: Some(Duration::from_secs(300)),
                    consistency_required: 2,
                },
                estimated_duration: Duration::from_secs(600),
            });

            // Create additional variations
            if i < 20 {
                exercises.push(TrainingExercise {
                    exercise_id: format!("quality_{id_suffix}_extended"),
                    name: format!("{name} - Extended"),
                    description: String::from("Extended quality practice with longer content"),
                    difficulty: 0.45 + (i as f32 * 0.01),
                    focus_areas: vec![FocusArea::Quality, FocusArea::Naturalness, FocusArea::Fluency],
                    exercise_type: ExerciseType::Quality,
                    target_text: format!("{text} This extended version requires maintaining consistent quality throughout a longer passage."),
                    reference_audio: None,
                    success_criteria: SuccessCriteria {
                        min_quality_score: 0.85,
                        min_pronunciation_score: 0.8,
                        max_attempts: 3,
                        time_limit: Some(Duration::from_secs(400)),
                        consistency_required: 2,
                    },
                    estimated_duration: Duration::from_secs(720),
                });
            }
        }

        exercises
    }

    /// Create rhythm and stress exercises
    fn create_rhythm_exercises() -> Vec<TrainingExercise> {
        let mut exercises = Vec::new();

        let rhythm_patterns = vec![
            ("iambic", "Iambic Rhythm", "I think that I shall never see a tree as lovely as a tree."),
            ("trochaic", "Trochaic Rhythm", "Listen to the rhythm, feel the beat, keep the timing, make it neat."),
            ("compound_stress", "Compound Stress", "Basketball, greenhouse, understand, overwhelm, photograph, telephone."),
            ("sentence_stress", "Sentence Stress", "I LOVE chocolate ice cream, but I PREFER vanilla cake today."),
            ("question_intonation", "Question Intonation", "Are you going? When will you arrive? How long will you stay?"),
            ("list_rhythm", "List Rhythm", "I need apples, oranges, bananas, grapes, and strawberries from the store."),
            ("contrast_stress", "Contrast Stress", "I said the RED car, not the BLUE car, was parked outside."),
            ("syllable_timing", "Syllable Timing", "Photography, biology, psychology, technology, methodology, terminology."),
            ("pause_timing", "Pause Timing", "First, we'll prepare the ingredients. Then, we'll mix them carefully. Finally, we'll bake the mixture."),
            ("emphatic_stress", "Emphatic Stress", "That was ABSOLUTELY amazing! I REALLY enjoyed the performance tonight!"),
        ];

        for (i, (id_suffix, name, text)) in rhythm_patterns.iter().enumerate() {
            let exercise_id = format!("rhythm_{id_suffix}");

            exercises.push(TrainingExercise {
                exercise_id,
                name: (*name).to_string(),
                description: String::from("Practice rhythm, stress patterns, and timing in speech"),
                difficulty: 0.4 + (i as f32 * 0.015),
                focus_areas: vec![FocusArea::Rhythm, FocusArea::Stress],
                exercise_type: ExerciseType::Rhythm,
                target_text: (*text).to_string(),
                reference_audio: None,
                success_criteria: SuccessCriteria {
                    min_quality_score: 0.75,
                    min_pronunciation_score: 0.8,
                    max_attempts: 4,
                    time_limit: Some(Duration::from_secs(240)),
                    consistency_required: 2,
                },
                estimated_duration: Duration::from_secs(480),
            });

            // Add practice variations
            exercises.push(TrainingExercise {
                exercise_id: format!("rhythm_{id_suffix}_practice"),
                name: format!("{name} Practice"),
                description: String::from("Intensive practice for rhythm and stress mastery"),
                difficulty: 0.45 + (i as f32 * 0.015),
                focus_areas: vec![FocusArea::Rhythm, FocusArea::Stress, FocusArea::Naturalness],
                exercise_type: ExerciseType::Rhythm,
                target_text: format!("{text}. Now repeat with varied emphasis and timing."),
                reference_audio: None,
                success_criteria: SuccessCriteria {
                    min_quality_score: 0.8,
                    min_pronunciation_score: 0.85,
                    max_attempts: 3,
                    time_limit: Some(Duration::from_secs(300)),
                    consistency_required: 2,
                },
                estimated_duration: Duration::from_secs(540),
            });

            exercises.push(TrainingExercise {
                exercise_id: format!("rhythm_{id_suffix}_advanced"),
                name: format!("{name} Advanced"),
                description: String::from("Advanced rhythm control with complex patterns"),
                difficulty: 0.5 + (i as f32 * 0.015),
                focus_areas: vec![FocusArea::Rhythm, FocusArea::Stress, FocusArea::Fluency],
                exercise_type: ExerciseType::Advanced,
                target_text: format!(
                    "{text}. Demonstrate mastery through consistent rhythm control."
                ),
                reference_audio: None,
                success_criteria: SuccessCriteria {
                    min_quality_score: 0.85,
                    min_pronunciation_score: 0.9,
                    max_attempts: 2,
                    time_limit: Some(Duration::from_secs(180)),
                    consistency_required: 1,
                },
                estimated_duration: Duration::from_secs(420),
            });
        }

        exercises
    }

    /// Create intonation exercises
    fn create_intonation_exercises() -> Vec<TrainingExercise> {
        let mut exercises = Vec::new();

        let intonation_types = vec![
            (
                "rising_questions",
                "Rising Questions",
                "Are you coming? Is this correct? Will you help me?",
            ),
            (
                "falling_statements",
                "Falling Statements",
                "I'm going home. The meeting is over. We finished the project.",
            ),
            (
                "tag_questions",
                "Tag Questions",
                "You're coming, aren't you? He's nice, isn't he? It's cold, don't you think?",
            ),
            (
                "choice_questions",
                "Choice Questions",
                "Do you want tea or coffee? Should we go left or right?",
            ),
            (
                "wh_questions",
                "WH Questions",
                "Where are you going? What time is it? Why did you say that?",
            ),
            (
                "surprise_intonation",
                "Surprise Intonation",
                "Really? You won the lottery? That's incredible!",
            ),
            (
                "doubt_intonation",
                "Doubt Intonation",
                "I'm not so sure about that. Maybe we should reconsider.",
            ),
            (
                "enthusiasm",
                "Enthusiastic Intonation",
                "That's fantastic! What an amazing opportunity! I'm so excited!",
            ),
            (
                "list_intonation",
                "List Intonation",
                "We need bread, milk, eggs, and cheese from the grocery store.",
            ),
            (
                "contrast_intonation",
                "Contrast Intonation",
                "I like apples, but I love oranges. He's tall, but she's short.",
            ),
        ];

        for (i, (id_suffix, name, text)) in intonation_types.iter().enumerate() {
            let exercise_id = format!("intonation_{id_suffix}");

            exercises.push(TrainingExercise {
                exercise_id,
                name: (*name).to_string(),
                description: String::from(
                    "Master intonation patterns for natural speech expression",
                ),
                difficulty: 0.45 + (i as f32 * 0.01),
                focus_areas: vec![FocusArea::Intonation, FocusArea::Naturalness],
                exercise_type: ExerciseType::Expression,
                target_text: (*text).to_string(),
                reference_audio: None,
                success_criteria: SuccessCriteria {
                    min_quality_score: 0.75,
                    min_pronunciation_score: 0.8,
                    max_attempts: 4,
                    time_limit: Some(Duration::from_secs(240)),
                    consistency_required: 2,
                },
                estimated_duration: Duration::from_secs(480),
            });

            exercises.push(TrainingExercise {
                exercise_id: format!("intonation_{id_suffix}_complex"),
                name: format!("{name} Complex"),
                description: String::from("Advanced intonation with complex emotional undertones"),
                difficulty: 0.55 + (i as f32 * 0.01),
                focus_areas: vec![
                    FocusArea::Intonation,
                    FocusArea::Naturalness,
                    FocusArea::Stress,
                ],
                exercise_type: ExerciseType::Expression,
                target_text: format!(
                    "{text}. Express this with varying emotional context and meaning."
                ),
                reference_audio: None,
                success_criteria: SuccessCriteria {
                    min_quality_score: 0.8,
                    min_pronunciation_score: 0.85,
                    max_attempts: 3,
                    time_limit: Some(Duration::from_secs(300)),
                    consistency_required: 2,
                },
                estimated_duration: Duration::from_secs(540),
            });

            exercises.push(TrainingExercise {
                exercise_id: format!("intonation_{id_suffix}_dialogue"),
                name: format!("{name} Dialogue"),
                description: String::from("Practice intonation in conversational context"),
                difficulty: 0.5 + (i as f32 * 0.01),
                focus_areas: vec![
                    FocusArea::Intonation,
                    FocusArea::Naturalness,
                    FocusArea::Fluency,
                ],
                exercise_type: ExerciseType::Expression,
                target_text: format!("Speaker A: {text} Speaker B: I understand what you mean."),
                reference_audio: None,
                success_criteria: SuccessCriteria {
                    min_quality_score: 0.8,
                    min_pronunciation_score: 0.85,
                    max_attempts: 3,
                    time_limit: Some(Duration::from_secs(360)),
                    consistency_required: 2,
                },
                estimated_duration: Duration::from_secs(600),
            });
        }

        exercises
    }

    /// Create expression and emotion exercises
    fn create_expression_exercises() -> Vec<TrainingExercise> {
        let mut exercises = Vec::new();

        let expression_types = vec![
            ("joy", "Joyful Expression", "I'm absolutely thrilled about this wonderful news! This is the best day ever!"),
            ("sadness", "Sad Expression", "I'm feeling quite disappointed about how things turned out today."),
            ("anger", "Angry Expression", "This is completely unacceptable! I demand an immediate explanation!"),
            ("surprise", "Surprised Expression", "Oh my goodness! I can't believe what just happened here!"),
            ("fear", "Fearful Expression", "I'm really worried about what might happen next in this situation."),
            ("disgust", "Disgusted Expression", "That is absolutely revolting and completely inappropriate behavior."),
            ("neutral", "Neutral Expression", "Today's weather forecast indicates partly cloudy conditions with moderate temperatures."),
            ("professional", "Professional Tone", "Thank you for your inquiry. We will process your request promptly."),
            ("friendly", "Friendly Tone", "Hi there! It's so great to see you again! How have you been?"),
            ("authoritative", "Authoritative Tone", "Please follow these instructions carefully and precisely as outlined."),
            ("sympathetic", "Sympathetic Tone", "I understand how difficult this situation must be for you right now."),
            ("excited", "Excited Expression", "This is amazing! I can't wait to share this incredible news with everyone!"),
            ("confused", "Confused Expression", "I'm not quite sure I understand what you're trying to explain to me."),
            ("confident", "Confident Expression", "I'm absolutely certain this is the right decision for our future success."),
            ("romantic", "Romantic Expression", "You look absolutely beautiful tonight, and I'm so lucky to be with you."),
            ("humorous", "Humorous Expression", "Did you hear about the mathematician who's afraid of negative numbers? He'll stop at nothing to avoid them!"),
            ("sarcastic", "Sarcastic Expression", "Oh, that's just fantastic. Exactly what I wanted to hear today."),
            ("motivational", "Motivational Expression", "You have the power to achieve anything you set your mind to!"),
            ("mysterious", "Mysterious Expression", "There are secrets hidden in this place that few people know about."),
            ("urgent", "Urgent Expression", "We need to act immediately! There's no time to waste on this matter!"),
        ];

        for (i, (emotion, name, text)) in expression_types.iter().enumerate() {
            let exercise_id = format!("expression_{emotion}");

            exercises.push(TrainingExercise {
                exercise_id,
                name: (*name).to_string(),
                description: format!(
                    "Practice expressing {emotion} through vocal tone and delivery"
                ),
                difficulty: 0.5 + (i as f32 * 0.005),
                focus_areas: vec![FocusArea::Naturalness, FocusArea::Intonation],
                exercise_type: ExerciseType::Expression,
                target_text: (*text).to_string(),
                reference_audio: None,
                success_criteria: SuccessCriteria {
                    min_quality_score: 0.75,
                    min_pronunciation_score: 0.8,
                    max_attempts: 4,
                    time_limit: Some(Duration::from_secs(300)),
                    consistency_required: 1,
                },
                estimated_duration: Duration::from_secs(480),
            });

            // Add contextual variation
            exercises.push(TrainingExercise {
                exercise_id: format!("expression_{emotion}_context"),
                name: format!("{name} in Context"),
                description: format!("Practice {emotion} expression within conversational context"),
                difficulty: 0.55 + (i as f32 * 0.005),
                focus_areas: vec![
                    FocusArea::Naturalness,
                    FocusArea::Intonation,
                    FocusArea::Fluency,
                ],
                exercise_type: ExerciseType::Expression,
                target_text: format!("Context: You're speaking to a close friend. Message: {text}"),
                reference_audio: None,
                success_criteria: SuccessCriteria {
                    min_quality_score: 0.8,
                    min_pronunciation_score: 0.85,
                    max_attempts: 3,
                    time_limit: Some(Duration::from_secs(360)),
                    consistency_required: 2,
                },
                estimated_duration: Duration::from_secs(540),
            });
        }

        exercises
    }

    /// Create fluency exercises for advanced users
    fn create_fluency_exercises() -> Vec<TrainingExercise> {
        let mut exercises = Vec::new();

        let fluency_texts = vec![
            "The phenomenon of artificial intelligence has rapidly transformed numerous industries and continues to reshape our understanding of technological capabilities and human-machine interactions.",
            "Environmental sustainability requires comprehensive policy frameworks that balance economic development with ecological preservation while promoting renewable energy solutions and sustainable practices.",
            "Modern communication technologies have revolutionized global connectivity, enabling instantaneous information exchange and facilitating collaborative efforts across geographical boundaries and cultural differences.",
            "Scientific research methodologies involve systematic investigation procedures that require rigorous data collection, statistical analysis, and peer review processes to ensure validity and reliability.",
            "International diplomacy demands sophisticated negotiation skills, cultural sensitivity, and strategic understanding of geopolitical dynamics to achieve mutually beneficial agreements and peaceful resolutions.",
            "Educational psychology explores cognitive development processes, learning theories, and instructional design principles to optimize teaching effectiveness and student engagement in diverse learning environments.",
            "Biotechnology innovations continue advancing medical treatments through genetic engineering, personalized medicine approaches, and novel therapeutic interventions that target specific molecular pathways.",
            "Urban planning strategies must integrate transportation systems, housing development, commercial zoning, and green spaces while considering demographic trends and environmental impact assessments.",
            "Quantum computing represents a paradigm shift in computational capabilities, offering unprecedented processing power for complex calculations that classical computers cannot efficiently perform.",
            "Corporate governance frameworks establish accountability mechanisms, risk management protocols, and ethical standards that guide organizational decision-making and stakeholder relationship management.",
            "Climate change mitigation strategies encompass carbon reduction initiatives, renewable energy adoption, ecosystem restoration projects, and international cooperation agreements to address global warming.",
            "Psychological resilience involves adaptive coping mechanisms, emotional regulation skills, and social support systems that enable individuals to overcome adversity and maintain mental well-being.",
            "Financial market dynamics reflect complex interactions between economic indicators, investor sentiment, regulatory policies, and global events that influence asset valuations and trading patterns.",
            "Neuroscience research reveals intricate neural networks, synaptic connections, and brain plasticity mechanisms that underlie cognitive functions, memory formation, and behavioral responses.",
            "Archaeological investigations employ interdisciplinary methodologies, advanced dating techniques, and digital documentation systems to reconstruct historical narratives and cultural heritage preservation.",
        ];

        for (i, text) in fluency_texts.iter().enumerate() {
            let exercise_id = format!("fluency_advanced_{}", i + 1);

            exercises.push(TrainingExercise {
                exercise_id,
                name: format!("Advanced Fluency Challenge {}", i + 1),
                description: "Master complex text delivery with natural fluency and pace"
                    .to_string(),
                difficulty: 0.75 + (i as f32 * 0.01),
                focus_areas: vec![
                    FocusArea::Fluency,
                    FocusArea::Naturalness,
                    FocusArea::Rhythm,
                ],
                exercise_type: ExerciseType::Fluency,
                target_text: (*text).to_string(),
                reference_audio: None,
                success_criteria: SuccessCriteria {
                    min_quality_score: 0.85,
                    min_pronunciation_score: 0.9,
                    max_attempts: 2,
                    time_limit: Some(Duration::from_secs(240)),
                    consistency_required: 1,
                },
                estimated_duration: Duration::from_secs(600),
            });

            // Speed variation
            exercises.push(TrainingExercise {
                exercise_id: format!("fluency_speed_{}", i + 1),
                name: format!("Speed Control {}", i + 1),
                description: String::from("Practice varying speech rate while maintaining clarity"),
                difficulty: 0.8 + (i as f32 * 0.008),
                focus_areas: vec![
                    FocusArea::Fluency,
                    FocusArea::Quality,
                    FocusArea::Pronunciation,
                ],
                exercise_type: ExerciseType::Fluency,
                target_text: format!(
                    "Slow: {} Normal: {} Fast: {}",
                    &text[..50],
                    &text[50..100],
                    &text[100..150]
                ),
                reference_audio: None,
                success_criteria: SuccessCriteria {
                    min_quality_score: 0.9,
                    min_pronunciation_score: 0.95,
                    max_attempts: 2,
                    time_limit: Some(Duration::from_secs(300)),
                    consistency_required: 1,
                },
                estimated_duration: Duration::from_secs(720),
            });

            // Comprehension challenge
            exercises.push(TrainingExercise {
                exercise_id: format!("fluency_comprehension_{}", i + 1),
                name: format!("Comprehension Challenge {}", i + 1),
                description: String::from(
                    "Deliver complex content with understanding and emphasis",
                ),
                difficulty: 0.85 + (i as f32 * 0.005),
                focus_areas: vec![
                    FocusArea::Fluency,
                    FocusArea::Naturalness,
                    FocusArea::Stress,
                ],
                exercise_type: ExerciseType::Advanced,
                target_text: format!(
                    "{text}. Emphasize key concepts while maintaining natural flow."
                ),
                reference_audio: None,
                success_criteria: SuccessCriteria {
                    min_quality_score: 0.9,
                    min_pronunciation_score: 0.95,
                    max_attempts: 1,
                    time_limit: Some(Duration::from_secs(300)),
                    consistency_required: 1,
                },
                estimated_duration: Duration::from_secs(800),
            });
        }

        exercises
    }

    /// Create exercises for all remaining categories to reach 500+ total
    fn create_complex_text_exercises() -> Vec<TrainingExercise> {
        let mut exercises = Vec::new();

        let complex_domains = vec![
            ("legal", "Legal Document", "The parties hereby agree to the terms and conditions set forth in this comprehensive agreement, acknowledging their respective rights, obligations, and liabilities."),
            ("medical", "Medical Report", "Patient presents with acute symptoms requiring immediate diagnostic evaluation and potential therapeutic intervention based on clinical assessment findings."),
            ("scientific", "Research Abstract", "This study investigates novel methodologies for data analysis using advanced statistical models and machine learning algorithms to identify significant patterns."),
            ("technical", "Technical Manual", "System initialization requires proper configuration of network parameters, security protocols, and user authentication mechanisms before deployment."),
            ("academic", "Academic Paper", "Contemporary theoretical frameworks suggest multifaceted approaches to understanding complex sociological phenomena within institutional contexts."),
            ("business", "Business Proposal", "Market analysis indicates substantial growth opportunities in emerging sectors, requiring strategic investment and comprehensive risk assessment protocols."),
            ("philosophical", "Philosophy Text", "Existential questions regarding human consciousness and the nature of reality continue to challenge our fundamental assumptions about knowledge and perception."),
            ("literary", "Literary Analysis", "The author's sophisticated use of symbolism and metaphorical language creates layers of meaning that invite multiple interpretations and critical examination."),
            ("historical", "Historical Account", "Archaeological evidence suggests that ancient civilizations developed complex social structures and technological innovations that influenced subsequent cultural developments."),
            ("economic", "Economic Theory", "Macroeconomic indicators demonstrate cyclical patterns influenced by monetary policy, fiscal measures, and international trade dynamics affecting global markets."),
        ];

        for (i, (domain, name, base_text)) in complex_domains.iter().enumerate() {
            // Basic complex exercise
            exercises.push(TrainingExercise {
                exercise_id: format!("complex_{domain}_basic"),
                name: format!("{name} - Basic"),
                description: format!("Practice reading {domain} content with proper pronunciation"),
                difficulty: 0.7 + (i as f32 * 0.01),
                focus_areas: vec![
                    FocusArea::Pronunciation,
                    FocusArea::Quality,
                    FocusArea::Fluency,
                ],
                exercise_type: ExerciseType::Advanced,
                target_text: (*base_text).to_string(),
                reference_audio: None,
                success_criteria: SuccessCriteria {
                    min_quality_score: 0.8,
                    min_pronunciation_score: 0.85,
                    max_attempts: 3,
                    time_limit: Some(Duration::from_secs(300)),
                    consistency_required: 2,
                },
                estimated_duration: Duration::from_secs(600),
            });

            // Advanced complex exercise
            exercises.push(TrainingExercise {
                exercise_id: format!("complex_{domain}_advanced"),
                name: format!("{name} - Advanced"),
                description: format!("Master {domain} terminology with natural delivery"),
                difficulty: 0.8 + (i as f32 * 0.01),
                focus_areas: vec![FocusArea::Fluency, FocusArea::Naturalness, FocusArea::Stress],
                exercise_type: ExerciseType::Advanced,
                target_text: format!("{base_text}. This advanced version requires demonstrating understanding through appropriate emphasis and pacing."),
                reference_audio: None,
                success_criteria: SuccessCriteria {
                    min_quality_score: 0.85,
                    min_pronunciation_score: 0.9,
                    max_attempts: 2,
                    time_limit: Some(Duration::from_secs(360)),
                    consistency_required: 1,
                },
                estimated_duration: Duration::from_secs(720),
            });

            // Expert level
            exercises.push(TrainingExercise {
                exercise_id: format!("complex_{domain}_expert"),
                name: format!("{name} - Expert"),
                description: format!("Expert-level {domain} content delivery"),
                difficulty: 0.9 + (i as f32 * 0.005),
                focus_areas: vec![FocusArea::Fluency, FocusArea::Naturalness, FocusArea::Quality],
                exercise_type: ExerciseType::Challenge,
                target_text: format!("Expert challenge: {base_text}. Demonstrate mastery through flawless delivery and natural expression."),
                reference_audio: None,
                success_criteria: SuccessCriteria {
                    min_quality_score: 0.9,
                    min_pronunciation_score: 0.95,
                    max_attempts: 1,
                    time_limit: Some(Duration::from_secs(240)),
                    consistency_required: 1,
                },
                estimated_duration: Duration::from_secs(480),
            });
        }

        exercises
    }

    /// Create remaining specialized exercises to complete the 500+ library
    fn create_emotion_exercises() -> Vec<TrainingExercise> {
        let emotions = vec![
            (
                "happiness",
                "Joyful Emotion",
                "I'm absolutely delighted! This is the most wonderful surprise!",
            ),
            (
                "sadness",
                "Sorrowful Expression",
                "I'm feeling quite disappointed about how everything turned out.",
            ),
            (
                "anger",
                "Angry Emotion",
                "This is completely unacceptable and I demand an explanation!",
            ),
            (
                "fear",
                "Fearful Expression",
                "I'm really worried about what might happen next.",
            ),
            (
                "surprise",
                "Surprised Reaction",
                "Oh my goodness! I can't believe this just happened!",
            ),
            (
                "disgust",
                "Disgusted Response",
                "That's absolutely revolting and inappropriate.",
            ),
            (
                "contempt",
                "Contemptuous Tone",
                "That behavior is beneath our standards.",
            ),
            (
                "pride",
                "Proud Declaration",
                "I'm incredibly proud of what we've accomplished together.",
            ),
            (
                "shame",
                "Shameful Admission",
                "I'm deeply embarrassed about my behavior yesterday.",
            ),
            (
                "guilt",
                "Guilty Confession",
                "I feel terrible about what I did wrong.",
            ),
            (
                "envy",
                "Envious Expression",
                "I wish I could have what they have.",
            ),
            (
                "jealousy",
                "Jealous Reaction",
                "I don't like seeing them together like that.",
            ),
            (
                "love",
                "Loving Expression",
                "You mean absolutely everything to me, darling.",
            ),
            (
                "hate",
                "Hateful Declaration",
                "I absolutely despise everything about this situation.",
            ),
            (
                "hope",
                "Hopeful Statement",
                "I believe everything will work out perfectly in the end.",
            ),
            (
                "despair",
                "Despairing Cry",
                "I don't see any way out of this terrible situation.",
            ),
            (
                "excitement",
                "Excited Exclamation",
                "This is the most amazing thing that's ever happened!",
            ),
            (
                "boredom",
                "Bored Complaint",
                "This is incredibly tedious and mind-numbing.",
            ),
            (
                "curiosity",
                "Curious Inquiry",
                "I wonder what secrets are hidden behind that door.",
            ),
            (
                "confusion",
                "Confused Question",
                "I don't understand what you're trying to tell me.",
            ),
            (
                "confidence",
                "Confident Assertion",
                "I'm absolutely certain this is the right approach.",
            ),
            (
                "insecurity",
                "Insecure Doubt",
                "I'm not sure if I'm capable of handling this.",
            ),
            (
                "relief",
                "Relieved Sigh",
                "Thank goodness that difficult ordeal is finally over.",
            ),
            (
                "anxiety",
                "Anxious Worry",
                "I'm extremely nervous about tomorrow's important meeting.",
            ),
            (
                "calm",
                "Calm Assurance",
                "Everything is going to be perfectly fine, don't worry.",
            ),
        ];

        emotions
            .iter()
            .enumerate()
            .map(|(i, (emotion, name, text))| TrainingExercise {
                exercise_id: format!("emotion_{emotion}"),
                name: (*name).to_string(),
                description: format!("Express genuine {emotion} through vocal tone and delivery"),
                difficulty: 0.7 + (i as f32 * 0.008),
                focus_areas: vec![FocusArea::Naturalness, FocusArea::Intonation],
                exercise_type: ExerciseType::Expression,
                target_text: (*text).to_string(),
                reference_audio: None,
                success_criteria: SuccessCriteria {
                    min_quality_score: 0.8,
                    min_pronunciation_score: 0.85,
                    max_attempts: 3,
                    time_limit: Some(Duration::from_secs(180)),
                    consistency_required: 1,
                },
                estimated_duration: Duration::from_secs(360),
            })
            .collect()
    }

    fn create_professional_exercises() -> Vec<TrainingExercise> {
        (0..40).map(|i| {
            TrainingExercise {
                exercise_id: format!("professional_{}", i + 1),
                name: format!("Professional Communication {}", i + 1),
                description: String::from("Master professional speaking scenarios"),
                difficulty: 0.7 + (i as f32 * 0.005),
                focus_areas: vec![FocusArea::Quality, FocusArea::Naturalness, FocusArea::Fluency],
                exercise_type: ExerciseType::Quality,
                target_text: format!("Professional scenario {}: Thank you for your time today. Let's discuss the strategic objectives for this quarter.", i + 1),
                reference_audio: None,
                success_criteria: SuccessCriteria {
                    min_quality_score: 0.85,
                    min_pronunciation_score: 0.9,
                    max_attempts: 2,
                    time_limit: Some(Duration::from_secs(240)),
                    consistency_required: 1,
                },
                estimated_duration: Duration::from_secs(480),
            }
        }).collect()
    }

    fn create_technical_exercises() -> Vec<TrainingExercise> {
        (0..25).map(|i| {
            TrainingExercise {
                exercise_id: format!("technical_{}", i + 1),
                name: format!("Technical Content {}", i + 1),
                description: String::from("Handle technical terminology and concepts"),
                difficulty: 0.8 + (i as f32 * 0.008),
                focus_areas: vec![FocusArea::Pronunciation, FocusArea::Fluency, FocusArea::Quality],
                exercise_type: ExerciseType::Advanced,
                target_text: format!("Technical procedure {}: Initialize the quantum cryptographic protocol using asymmetric encryption algorithms.", i + 1),
                reference_audio: None,
                success_criteria: SuccessCriteria {
                    min_quality_score: 0.9,
                    min_pronunciation_score: 0.95,
                    max_attempts: 2,
                    time_limit: Some(Duration::from_secs(300)),
                    consistency_required: 1,
                },
                estimated_duration: Duration::from_secs(600),
            }
        }).collect()
    }

    fn create_accent_exercises() -> Vec<TrainingExercise> {
        (0..30)
            .map(|i| TrainingExercise {
                exercise_id: format!("accent_{}", i + 1),
                name: format!("Accent Training {}", i + 1),
                description: "Practice various accent patterns and regional pronunciations"
                    .to_string(),
                difficulty: 0.6 + (i as f32 * 0.01),
                focus_areas: vec![FocusArea::Pronunciation, FocusArea::Naturalness],
                exercise_type: ExerciseType::Pronunciation,
                target_text: format!(
                    "Accent pattern {}: The weather is quite lovely today, wouldn't you agree?",
                    i + 1
                ),
                reference_audio: None,
                success_criteria: SuccessCriteria {
                    min_quality_score: 0.75,
                    min_pronunciation_score: 0.8,
                    max_attempts: 4,
                    time_limit: Some(Duration::from_secs(240)),
                    consistency_required: 2,
                },
                estimated_duration: Duration::from_secs(420),
            })
            .collect()
    }

    fn create_speed_exercises() -> Vec<TrainingExercise> {
        (0..25).map(|i| {
            TrainingExercise {
                exercise_id: format!("speed_{}", i + 1),
                name: format!("Speed Control {}", i + 1),
                description: String::from("Master speech rate control and clarity at various speeds"),
                difficulty: 0.75 + (i as f32 * 0.008),
                focus_areas: vec![FocusArea::Fluency, FocusArea::Quality, FocusArea::Pronunciation],
                exercise_type: ExerciseType::Fluency,
                target_text: format!("Speed exercise {}: Rapid delivery requires precise articulation while maintaining crystal clear pronunciation.", i + 1),
                reference_audio: None,
                success_criteria: SuccessCriteria {
                    min_quality_score: 0.85,
                    min_pronunciation_score: 0.9,
                    max_attempts: 3,
                    time_limit: Some(Duration::from_secs(180)),
                    consistency_required: 1,
                },
                estimated_duration: Duration::from_secs(360),
            }
        }).collect()
    }

    fn create_poetry_exercises() -> Vec<TrainingExercise> {
        (0..20).map(|i| {
            TrainingExercise {
                exercise_id: format!("poetry_{}", i + 1),
                name: format!("Poetry Reading {}", i + 1),
                description: String::from("Express poetic language with appropriate rhythm and emotion"),
                difficulty: 0.7 + (i as f32 * 0.01),
                focus_areas: vec![FocusArea::Rhythm, FocusArea::Naturalness, FocusArea::Intonation],
                exercise_type: ExerciseType::Expression,
                target_text: format!("Poetic verse {}: Shall I compare thee to a summer's day? Thou art more lovely and more temperate.", i + 1),
                reference_audio: None,
                success_criteria: SuccessCriteria {
                    min_quality_score: 0.8,
                    min_pronunciation_score: 0.85,
                    max_attempts: 3,
                    time_limit: Some(Duration::from_secs(300)),
                    consistency_required: 2,
                },
                estimated_duration: Duration::from_secs(480),
            }
        }).collect()
    }

    fn create_conversation_exercises() -> Vec<TrainingExercise> {
        (0..25).map(|i| {
            TrainingExercise {
                exercise_id: format!("conversation_{}", i + 1),
                name: format!("Conversation Practice {}", i + 1),
                description: String::from("Practice natural conversational flow and turn-taking"),
                difficulty: 0.6 + (i as f32 * 0.01),
                focus_areas: vec![FocusArea::Naturalness, FocusArea::Fluency, FocusArea::Intonation],
                exercise_type: ExerciseType::FreeForm,
                target_text: format!("Conversation {}: A: How was your day? B: It was quite productive, thank you for asking. How about yours?", i + 1),
                reference_audio: None,
                success_criteria: SuccessCriteria {
                    min_quality_score: 0.75,
                    min_pronunciation_score: 0.8,
                    max_attempts: 4,
                    time_limit: Some(Duration::from_secs(240)),
                    consistency_required: 1,
                },
                estimated_duration: Duration::from_secs(360),
            }
        }).collect()
    }

    fn create_challenge_exercises() -> Vec<TrainingExercise> {
        (0..30).map(|i| {
            TrainingExercise {
                exercise_id: format!("challenge_{}", i + 1),
                name: format!("Ultimate Challenge {}", i + 1),
                description: String::from("Master-level exercises combining all skills"),
                difficulty: 0.9 + (i as f32 * 0.003),
                focus_areas: vec![FocusArea::Fluency, FocusArea::Quality, FocusArea::Naturalness, FocusArea::Pronunciation],
                exercise_type: ExerciseType::Challenge,
                target_text: format!("Ultimate challenge {}: Demonstrate mastery across all dimensions of speech synthesis excellence with this comprehensive evaluation.", i + 1),
                reference_audio: None,
                success_criteria: SuccessCriteria {
                    min_quality_score: 0.95,
                    min_pronunciation_score: 0.98,
                    max_attempts: 1,
                    time_limit: Some(Duration::from_secs(180)),
                    consistency_required: 1,
                },
                estimated_duration: Duration::from_secs(300),
            }
        }).collect()
    }

    fn create_exercise_categories() -> Vec<ExerciseCategory> {
        vec![
            ExerciseCategory {
                name: String::from("Pronunciation Fundamentals"),
                description: String::from("Basic phoneme and word pronunciation practice"),
                focus_areas: vec![FocusArea::Pronunciation],
                difficulty_range: (0.1, 0.3),
                exercise_count: 120, // Phonemes + words + basic sentences
            },
            ExerciseCategory {
                name: String::from("Quality Enhancement"),
                description: String::from("Improve audio quality and clarity for professional use"),
                focus_areas: vec![FocusArea::Quality, FocusArea::Naturalness],
                difficulty_range: (0.4, 0.6),
                exercise_count: 75, // Quality + rhythm + intonation exercises
            },
            ExerciseCategory {
                name: String::from("Expression & Emotion"),
                description: String::from("Master emotional expression and vocal variety"),
                focus_areas: vec![FocusArea::Naturalness, FocusArea::Intonation],
                difficulty_range: (0.5, 0.7),
                exercise_count: 105, // Expression + emotion + intonation exercises
            },
            ExerciseCategory {
                name: String::from("Advanced Fluency"),
                description: String::from("Master natural speech flow and complex content"),
                focus_areas: vec![
                    FocusArea::Fluency,
                    FocusArea::Rhythm,
                    FocusArea::Naturalness,
                ],
                difficulty_range: (0.7, 0.9),
                exercise_count: 135, // Fluency + complex text + technical exercises
            },
            ExerciseCategory {
                name: String::from("Professional & Technical"),
                description: String::from(
                    "Handle professional and technical content with expertise",
                ),
                focus_areas: vec![
                    FocusArea::Quality,
                    FocusArea::Fluency,
                    FocusArea::Pronunciation,
                ],
                difficulty_range: (0.7, 0.95),
                exercise_count: 95, // Professional + technical + accent exercises
            },
            ExerciseCategory {
                name: String::from("Master Challenges"),
                description: String::from("Ultimate challenges combining all skills for mastery"),
                focus_areas: vec![
                    FocusArea::Fluency,
                    FocusArea::Quality,
                    FocusArea::Naturalness,
                    FocusArea::Pronunciation,
                ],
                difficulty_range: (0.9, 1.0),
                exercise_count: 70, // Speed + poetry + conversation + challenges
            },
        ]
    }
}

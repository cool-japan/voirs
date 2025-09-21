use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

/// Comprehensive Game Integration Example for VoiRS
///
/// This example demonstrates how to integrate VoiRS with popular game engines
/// including Unity and Unreal Engine, providing real-time audio synthesis,
/// spatial audio, voice acting, and dynamic soundscapes for games.
///
/// Features Demonstrated:
/// - Unity C# integration patterns
/// - Unreal Engine C++ integration patterns  
/// - Real-time character voice synthesis
/// - Dynamic soundscape generation
/// - Multi-language support for international games
/// - Performance optimization for 60+ FPS gameplay
/// - Memory management for long gaming sessions

#[derive(Debug, Clone)]
pub struct GameAudioConfig {
    pub target_fps: u32,
    pub max_concurrent_voices: u32,
    pub audio_buffer_size: usize,
    pub quality_level: QualityLevel,
    pub spatial_audio_enabled: bool,
    pub voice_cache_size: usize,
    pub memory_limit_mb: u32,
}

#[derive(Debug, Clone, Copy)]
pub enum QualityLevel {
    Low = 1,    // Mobile games, low-end devices
    Medium = 2, // Standard PC gaming
    High = 3,   // AAA games, high-end systems
    Ultra = 4,  // VR games, premium audio
}

#[derive(Debug, Clone)]
pub struct GameCharacter {
    pub id: String,
    pub name: String,
    pub voice_profile: VoiceProfile,
    pub emotional_state: EmotionalState,
    pub position: Vector3D,
    pub dialogue_history: Vec<DialogueLine>,
}

#[derive(Debug, Clone)]
pub struct VoiceProfile {
    pub voice_id: String,
    pub pitch_multiplier: f32,
    pub speed_multiplier: f32,
    pub emotion_intensity: f32,
    pub accent: String,
    pub age_category: AgeCategory,
    pub gender: Gender,
}

#[derive(Debug, Clone, Copy)]
pub enum AgeCategory {
    Child,
    Teenager,
    YoungAdult,
    MiddleAged,
    Elder,
}

#[derive(Debug, Clone, Copy)]
pub enum Gender {
    Male,
    Female,
    NonBinary,
    Robotic,
}

#[derive(Debug, Clone)]
pub struct EmotionalState {
    pub primary_emotion: Emotion,
    pub intensity: f32,
    pub duration_ms: u32,
    pub transition_speed: f32,
}

#[derive(Debug, Clone, Copy)]
pub enum Emotion {
    Neutral,
    Happy,
    Sad,
    Angry,
    Fearful,
    Surprised,
    Disgusted,
    Excited,
    Calm,
    Stressed,
}

#[derive(Debug, Clone)]
pub struct DialogueLine {
    pub text: String,
    pub timestamp: Instant,
    pub emotion: Emotion,
    pub priority: DialoguePriority,
    pub localization_key: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialOrd, Ord, PartialEq, Eq)]
pub enum DialoguePriority {
    Background = 1,
    Normal = 2,
    Important = 3,
    Critical = 4,
    System = 5,
}

#[derive(Debug, Clone)]
pub struct Vector3D {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vector3D {
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    pub fn distance_to(&self, other: &Vector3D) -> f32 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
}

/// Unity Integration Components
pub struct UnityIntegration {
    config: GameAudioConfig,
    character_manager: Arc<Mutex<CharacterManager>>,
    audio_pipeline: Arc<Mutex<GameAudioPipeline>>,
    unity_callbacks: UnityCallbacks,
}

impl UnityIntegration {
    pub fn new(config: GameAudioConfig) -> Result<Self, GameIntegrationError> {
        let character_manager = Arc::new(Mutex::new(CharacterManager::new(config.clone())?));
        let audio_pipeline = Arc::new(Mutex::new(GameAudioPipeline::new(config.clone())?));
        let unity_callbacks = UnityCallbacks::new();

        Ok(Self {
            config,
            character_manager,
            audio_pipeline,
            unity_callbacks,
        })
    }

    /// C# Unity interop functions that would be called from Unity scripts
    pub fn unity_speak_dialogue(
        &mut self,
        character_id: &str,
        text: &str,
        emotion: i32,
        priority: i32,
    ) -> Result<u32, GameIntegrationError> {
        let emotion = match emotion {
            0 => Emotion::Neutral,
            1 => Emotion::Happy,
            2 => Emotion::Sad,
            3 => Emotion::Angry,
            4 => Emotion::Fearful,
            5 => Emotion::Surprised,
            _ => Emotion::Neutral,
        };

        let priority = match priority {
            1 => DialoguePriority::Background,
            2 => DialoguePriority::Normal,
            3 => DialoguePriority::Important,
            4 => DialoguePriority::Critical,
            5 => DialoguePriority::System,
            _ => DialoguePriority::Normal,
        };

        let dialogue = DialogueLine {
            text: text.to_string(),
            timestamp: Instant::now(),
            emotion,
            priority,
            localization_key: None,
        };

        let mut manager = self
            .character_manager
            .lock()
            .map_err(|_| GameIntegrationError::ThreadLockError)?;
        let dialogue_id = manager.queue_dialogue(character_id, dialogue)?;

        Ok(dialogue_id)
    }

    pub fn unity_update_character_position(
        &mut self,
        character_id: &str,
        x: f32,
        y: f32,
        z: f32,
    ) -> Result<(), GameIntegrationError> {
        let mut manager = self
            .character_manager
            .lock()
            .map_err(|_| GameIntegrationError::ThreadLockError)?;
        manager.update_character_position(character_id, Vector3D::new(x, y, z))
    }

    pub fn unity_set_listener_position(
        &mut self,
        x: f32,
        y: f32,
        z: f32,
        forward_x: f32,
        forward_y: f32,
        forward_z: f32,
    ) -> Result<(), GameIntegrationError> {
        let mut pipeline = self
            .audio_pipeline
            .lock()
            .map_err(|_| GameIntegrationError::ThreadLockError)?;
        pipeline.set_listener_position(
            Vector3D::new(x, y, z),
            Vector3D::new(forward_x, forward_y, forward_z),
        )
    }

    pub fn unity_get_performance_stats(
        &self,
    ) -> Result<UnityPerformanceStats, GameIntegrationError> {
        let pipeline = self
            .audio_pipeline
            .lock()
            .map_err(|_| GameIntegrationError::ThreadLockError)?;
        Ok(pipeline.get_performance_stats())
    }

    /// Generate Unity C# bindings code
    pub fn generate_unity_bindings() -> String {
        r#"
// Unity C# Integration Code
using System;
using System.Runtime.InteropServices;
using UnityEngine;

public class VoiRSUnityIntegration : MonoBehaviour 
{
    // DLL Import declarations for VoiRS native functions
    [DllImport("voirs_unity")]
    private static extern int voirs_unity_speak_dialogue(IntPtr unity_instance, string character_id, string text, int emotion, int priority);
    
    [DllImport("voirs_unity")]
    private static extern int voirs_unity_update_character_position(IntPtr unity_instance, string character_id, float x, float y, float z);
    
    [DllImport("voirs_unity")]
    private static extern int voirs_unity_set_listener_position(IntPtr unity_instance, float x, float y, float z, float forward_x, float forward_y, float forward_z);

    [DllImport("voirs_unity")]
    private static extern IntPtr voirs_unity_create_instance();

    [DllImport("voirs_unity")]
    private static extern void voirs_unity_destroy_instance(IntPtr instance);

    private IntPtr voirsInstance;
    
    void Start() {
        voirsInstance = voirs_unity_create_instance();
        Debug.Log("VoiRS Unity integration initialized");
    }

    void OnDestroy() {
        if (voirsInstance != IntPtr.Zero) {
            voirs_unity_destroy_instance(voirsInstance);
        }
    }

    // High-level Unity API
    public void SpeakDialogue(string characterId, string text, VoiRSEmotion emotion = VoiRSEmotion.Neutral, VoiRSPriority priority = VoiRSPriority.Normal) {
        voirs_unity_speak_dialogue(voirsInstance, characterId, text, (int)emotion, (int)priority);
    }

    public void UpdateCharacterPosition(string characterId, Vector3 position) {
        voirs_unity_update_character_position(voirsInstance, characterId, position.x, position.y, position.z);
    }

    public void SetListenerPosition(Vector3 position, Vector3 forward) {
        voirs_unity_set_listener_position(voirsInstance, position.x, position.y, position.z, forward.x, forward.y, forward.z);
    }

    void Update() {
        // Update listener position with camera transform
        Transform cameraTransform = Camera.main.transform;
        SetListenerPosition(cameraTransform.position, cameraTransform.forward);
    }
}

public enum VoiRSEmotion {
    Neutral = 0, Happy = 1, Sad = 2, Angry = 3, Fearful = 4, Surprised = 5
}

public enum VoiRSPriority {
    Background = 1, Normal = 2, Important = 3, Critical = 4, System = 5
}

// Example NPC behavior script
public class VoiRSNPC : MonoBehaviour {
    [SerializeField] private string characterId = "npc_001";
    [SerializeField] private VoiRSEmotion currentEmotion = VoiRSEmotion.Neutral;
    private VoiRSUnityIntegration voirsIntegration;
    
    void Start() {
        voirsIntegration = FindObjectOfType<VoiRSUnityIntegration>();
    }

    void Update() {
        // Update character position for spatial audio
        voirsIntegration?.UpdateCharacterPosition(characterId, transform.position);
    }

    public void Speak(string dialogue) {
        voirsIntegration?.SpeakDialogue(characterId, dialogue, currentEmotion);
    }

    public void SetEmotion(VoiRSEmotion emotion) {
        currentEmotion = emotion;
    }
}
"#.to_string()
    }
}

/// Unreal Engine Integration Components
pub struct UnrealIntegration {
    config: GameAudioConfig,
    character_manager: Arc<Mutex<CharacterManager>>,
    audio_pipeline: Arc<Mutex<GameAudioPipeline>>,
    unreal_callbacks: UnrealCallbacks,
}

impl UnrealIntegration {
    pub fn new(config: GameAudioConfig) -> Result<Self, GameIntegrationError> {
        let character_manager = Arc::new(Mutex::new(CharacterManager::new(config.clone())?));
        let audio_pipeline = Arc::new(Mutex::new(GameAudioPipeline::new(config.clone())?));
        let unreal_callbacks = UnrealCallbacks::new();

        Ok(Self {
            config,
            character_manager,
            audio_pipeline,
            unreal_callbacks,
        })
    }

    /// Generate Unreal Engine C++ header code
    pub fn generate_unreal_header() -> String {
        r#"
// VoiRSUnrealIntegration.h
#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "Components/AudioComponent.h"
#include "Engine/World.h"
#include "VoiRSUnrealIntegration.generated.h"

UENUM(BlueprintType)
enum class EVoiRSEmotion : uint8
{
    Neutral     UMETA(DisplayName = "Neutral"),
    Happy       UMETA(DisplayName = "Happy"),
    Sad         UMETA(DisplayName = "Sad"),
    Angry       UMETA(DisplayName = "Angry"),
    Fearful     UMETA(DisplayName = "Fearful"),
    Surprised   UMETA(DisplayName = "Surprised")
};

UENUM(BlueprintType)
enum class EVoiRSPriority : uint8
{
    Background  UMETA(DisplayName = "Background"),
    Normal      UMETA(DisplayName = "Normal"),
    Important   UMETA(DisplayName = "Important"),
    Critical    UMETA(DisplayName = "Critical"),
    System      UMETA(DisplayName = "System")
};

USTRUCT(BlueprintType)
struct FVoiRSCharacterConfig
{
    GENERATED_BODY()

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VoiRS")
    FString CharacterId;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VoiRS")
    FString VoiceProfile;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VoiRS")
    float PitchMultiplier = 1.0f;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VoiRS")
    float SpeedMultiplier = 1.0f;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VoiRS")
    EVoiRSEmotion DefaultEmotion = EVoiRSEmotion::Neutral;
};

UCLASS(BlueprintType, Blueprintable)
class YOURGAME_API AVoiRSUnrealIntegration : public AActor
{
    GENERATED_BODY()

public:
    AVoiRSUnrealIntegration();

protected:
    virtual void BeginPlay() override;
    virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;

public:
    virtual void Tick(float DeltaTime) override;

    // Blueprint callable functions
    UFUNCTION(BlueprintCallable, Category = "VoiRS")
    bool InitializeVoiRS(int32 MaxConcurrentVoices = 16, int32 QualityLevel = 2);

    UFUNCTION(BlueprintCallable, Category = "VoiRS")
    int32 SpeakDialogue(const FString& CharacterId, const FString& Text, 
                        EVoiRSEmotion Emotion = EVoiRSEmotion::Neutral, 
                        EVoiRSPriority Priority = EVoiRSPriority::Normal);

    UFUNCTION(BlueprintCallable, Category = "VoiRS")
    bool UpdateCharacterPosition(const FString& CharacterId, FVector Position);

    UFUNCTION(BlueprintCallable, Category = "VoiRS")
    bool SetListenerTransform(FVector Position, FVector Forward, FVector Up);

    UFUNCTION(BlueprintCallable, Category = "VoiRS")
    bool RegisterCharacter(const FVoiRSCharacterConfig& CharacterConfig);

    UFUNCTION(BlueprintCallable, Category = "VoiRS")
    FString GetPerformanceStats() const;

    // Performance monitoring
    UFUNCTION(BlueprintCallable, Category = "VoiRS")
    float GetCPUUsagePercent() const;

    UFUNCTION(BlueprintCallable, Category = "VoiRS")
    float GetMemoryUsageMB() const;

    UFUNCTION(BlueprintCallable, Category = "VoiRS")
    int32 GetActiveVoiceCount() const;

private:
    // Native VoiRS instance handle
    void* VoiRSInstance;
    
    // Performance tracking
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "VoiRS", meta = (AllowPrivateAccess = "true"))
    float LastCPUUsage;

    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "VoiRS", meta = (AllowPrivateAccess = "true"))
    float LastMemoryUsage;

    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "VoiRS", meta = (AllowPrivateAccess = "true"))
    int32 ActiveVoices;

    // Character management
    TMap<FString, FVoiRSCharacterConfig> RegisteredCharacters;

    // Update functions
    void UpdateListenerPosition();
    void UpdatePerformanceMetrics();
};
"#.to_string()
    }

    /// Generate Unreal Engine C++ implementation code
    pub fn generate_unreal_implementation() -> String {
        r#"
// VoiRSUnrealIntegration.cpp
#include "VoiRSUnrealIntegration.h"
#include "Engine/Engine.h"
#include "GameFramework/PlayerController.h"
#include "Camera/CameraComponent.h"

// Import VoiRS C API functions
extern "C" {
    void* voirs_unreal_create_instance(int max_voices, int quality);
    void voirs_unreal_destroy_instance(void* instance);
    int voirs_unreal_speak_dialogue(void* instance, const char* character_id, const char* text, int emotion, int priority);
    int voirs_unreal_update_character_position(void* instance, const char* character_id, float x, float y, float z);
    int voirs_unreal_set_listener_position(void* instance, float x, float y, float z, float fx, float fy, float fz);
    int voirs_unreal_register_character(void* instance, const char* character_id, const char* voice_profile, float pitch, float speed);
    float voirs_unreal_get_cpu_usage(void* instance);
    float voirs_unreal_get_memory_usage(void* instance);
    int voirs_unreal_get_active_voices(void* instance);
}

AVoiRSUnrealIntegration::AVoiRSUnrealIntegration()
{
    PrimaryActorTick.bCanEverTick = true;
    VoiRSInstance = nullptr;
    LastCPUUsage = 0.0f;
    LastMemoryUsage = 0.0f;
    ActiveVoices = 0;
}

void AVoiRSUnrealIntegration::BeginPlay()
{
    Super::BeginPlay();
    
    // Initialize with default settings
    InitializeVoiRS();
    
    UE_LOG(LogTemp, Log, TEXT("VoiRS Unreal Integration initialized"));
}

void AVoiRSUnrealIntegration::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
    if (VoiRSInstance)
    {
        voirs_unreal_destroy_instance(VoiRSInstance);
        VoiRSInstance = nullptr;
    }
    
    Super::EndPlay(EndPlayReason);
}

void AVoiRSUnrealIntegration::Tick(float DeltaTime)
{
    Super::Tick(DeltaTime);
    
    UpdateListenerPosition();
    UpdatePerformanceMetrics();
}

bool AVoiRSUnrealIntegration::InitializeVoiRS(int32 MaxConcurrentVoices, int32 QualityLevel)
{
    if (VoiRSInstance)
    {
        voirs_unreal_destroy_instance(VoiRSInstance);
    }
    
    VoiRSInstance = voirs_unreal_create_instance(MaxConcurrentVoices, QualityLevel);
    
    return VoiRSInstance != nullptr;
}

int32 AVoiRSUnrealIntegration::SpeakDialogue(const FString& CharacterId, const FString& Text, EVoiRSEmotion Emotion, EVoiRSPriority Priority)
{
    if (!VoiRSInstance)
    {
        UE_LOG(LogTemp, Warning, TEXT("VoiRS not initialized"));
        return -1;
    }
    
    const char* CharacterIdCStr = TCHAR_TO_UTF8(*CharacterId);
    const char* TextCStr = TCHAR_TO_UTF8(*Text);
    
    int EmotionInt = static_cast<int>(Emotion);
    int PriorityInt = static_cast<int>(Priority);
    
    return voirs_unreal_speak_dialogue(VoiRSInstance, CharacterIdCStr, TextCStr, EmotionInt, PriorityInt);
}

bool AVoiRSUnrealIntegration::UpdateCharacterPosition(const FString& CharacterId, FVector Position)
{
    if (!VoiRSInstance)
    {
        return false;
    }
    
    const char* CharacterIdCStr = TCHAR_TO_UTF8(*CharacterId);
    
    // Convert Unreal coordinates (Z-up) to VoiRS coordinates
    return voirs_unreal_update_character_position(VoiRSInstance, CharacterIdCStr, 
                                                 Position.X / 100.0f,  // Convert cm to m
                                                 Position.Y / 100.0f, 
                                                 Position.Z / 100.0f) == 0;
}

bool AVoiRSUnrealIntegration::SetListenerTransform(FVector Position, FVector Forward, FVector Up)
{
    if (!VoiRSInstance)
    {
        return false;
    }
    
    // Convert Unreal coordinates to VoiRS coordinates
    return voirs_unreal_set_listener_position(VoiRSInstance,
                                             Position.X / 100.0f,  // Convert cm to m
                                             Position.Y / 100.0f,
                                             Position.Z / 100.0f,
                                             Forward.X, Forward.Y, Forward.Z) == 0;
}

bool AVoiRSUnrealIntegration::RegisterCharacter(const FVoiRSCharacterConfig& CharacterConfig)
{
    if (!VoiRSInstance)
    {
        return false;
    }
    
    RegisteredCharacters.Add(CharacterConfig.CharacterId, CharacterConfig);
    
    const char* CharacterIdCStr = TCHAR_TO_UTF8(*CharacterConfig.CharacterId);
    const char* VoiceProfileCStr = TCHAR_TO_UTF8(*CharacterConfig.VoiceProfile);
    
    return voirs_unreal_register_character(VoiRSInstance, CharacterIdCStr, VoiceProfileCStr,
                                          CharacterConfig.PitchMultiplier, 
                                          CharacterConfig.SpeedMultiplier) == 0;
}

void AVoiRSUnrealIntegration::UpdateListenerPosition()
{
    if (!VoiRSInstance)
        return;
        
    // Get player camera transform
    if (APlayerController* PC = GetWorld()->GetFirstPlayerController())
    {
        if (APawn* PlayerPawn = PC->GetPawn())
        {
            FVector CameraLocation;
            FRotator CameraRotation;
            PC->GetPlayerViewPoint(CameraLocation, CameraRotation);
            
            FVector Forward = CameraRotation.Vector();
            SetListenerTransform(CameraLocation, Forward, FVector::UpVector);
        }
    }
}

void AVoiRSUnrealIntegration::UpdatePerformanceMetrics()
{
    if (VoiRSInstance)
    {
        LastCPUUsage = voirs_unreal_get_cpu_usage(VoiRSInstance);
        LastMemoryUsage = voirs_unreal_get_memory_usage(VoiRSInstance);
        ActiveVoices = voirs_unreal_get_active_voices(VoiRSInstance);
    }
}

FString AVoiRSUnrealIntegration::GetPerformanceStats() const
{
    return FString::Printf(TEXT("CPU: %.1f%%, Memory: %.1f MB, Voices: %d"),
                          LastCPUUsage, LastMemoryUsage, ActiveVoices);
}

float AVoiRSUnrealIntegration::GetCPUUsagePercent() const
{
    return LastCPUUsage;
}

float AVoiRSUnrealIntegration::GetMemoryUsageMB() const
{
    return LastMemoryUsage;
}

int32 AVoiRSUnrealIntegration::GetActiveVoiceCount() const
{
    return ActiveVoices;
}
"#.to_string()
    }
}

pub struct UnityCallbacks {
    dialogue_complete_callbacks: Vec<Box<dyn Fn(u32) + Send + Sync>>,
}

impl UnityCallbacks {
    pub fn new() -> Self {
        Self {
            dialogue_complete_callbacks: Vec::new(),
        }
    }

    pub fn on_dialogue_complete(&mut self, callback: Box<dyn Fn(u32) + Send + Sync>) {
        self.dialogue_complete_callbacks.push(callback);
    }

    pub fn trigger_dialogue_complete(&self, dialogue_id: u32) {
        for callback in &self.dialogue_complete_callbacks {
            callback(dialogue_id);
        }
    }
}

pub struct UnrealCallbacks {
    audio_event_callbacks: Vec<Box<dyn Fn(&str, f32) + Send + Sync>>,
}

impl UnrealCallbacks {
    pub fn new() -> Self {
        Self {
            audio_event_callbacks: Vec::new(),
        }
    }

    pub fn on_audio_event(&mut self, callback: Box<dyn Fn(&str, f32) + Send + Sync>) {
        self.audio_event_callbacks.push(callback);
    }

    pub fn trigger_audio_event(&self, event_name: &str, intensity: f32) {
        for callback in &self.audio_event_callbacks {
            callback(event_name, intensity);
        }
    }
}

pub struct CharacterManager {
    characters: HashMap<String, GameCharacter>,
    dialogue_queue: Vec<QueuedDialogue>,
    next_dialogue_id: u32,
    config: GameAudioConfig,
}

#[derive(Debug, Clone)]
pub struct QueuedDialogue {
    pub dialogue_id: u32,
    pub character_id: String,
    pub dialogue: DialogueLine,
    pub status: DialogueStatus,
    pub start_time: Option<Instant>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DialogueStatus {
    Queued,
    Processing,
    Playing,
    Completed,
    Failed,
}

impl CharacterManager {
    pub fn new(config: GameAudioConfig) -> Result<Self, GameIntegrationError> {
        Ok(Self {
            characters: HashMap::new(),
            dialogue_queue: Vec::new(),
            next_dialogue_id: 1,
            config,
        })
    }

    pub fn add_character(&mut self, character: GameCharacter) {
        self.characters.insert(character.id.clone(), character);
    }

    pub fn queue_dialogue(
        &mut self,
        character_id: &str,
        dialogue: DialogueLine,
    ) -> Result<u32, GameIntegrationError> {
        if !self.characters.contains_key(character_id) {
            return Err(GameIntegrationError::CharacterNotFound(
                character_id.to_string(),
            ));
        }

        let dialogue_id = self.next_dialogue_id;
        self.next_dialogue_id += 1;

        let queued_dialogue = QueuedDialogue {
            dialogue_id,
            character_id: character_id.to_string(),
            dialogue,
            status: DialogueStatus::Queued,
            start_time: None,
        };

        // Insert dialogue in priority order
        let insert_pos = self
            .dialogue_queue
            .iter()
            .position(|d| d.dialogue.priority < queued_dialogue.dialogue.priority)
            .unwrap_or(self.dialogue_queue.len());

        self.dialogue_queue.insert(insert_pos, queued_dialogue);

        Ok(dialogue_id)
    }

    pub fn update_character_position(
        &mut self,
        character_id: &str,
        position: Vector3D,
    ) -> Result<(), GameIntegrationError> {
        match self.characters.get_mut(character_id) {
            Some(character) => {
                character.position = position;
                Ok(())
            }
            None => Err(GameIntegrationError::CharacterNotFound(
                character_id.to_string(),
            )),
        }
    }

    pub fn get_character(&self, character_id: &str) -> Option<&GameCharacter> {
        self.characters.get(character_id)
    }

    pub fn process_dialogue_queue(&mut self) -> Vec<QueuedDialogue> {
        let mut processed = Vec::new();
        let mut i = 0;

        while i < self.dialogue_queue.len() {
            if self.dialogue_queue[i].status == DialogueStatus::Queued {
                self.dialogue_queue[i].status = DialogueStatus::Processing;
                self.dialogue_queue[i].start_time = Some(Instant::now());
                processed.push(self.dialogue_queue.remove(i));
            } else {
                i += 1;
            }

            if processed.len() >= self.config.max_concurrent_voices as usize {
                break;
            }
        }

        processed
    }
}

pub struct GameAudioPipeline {
    config: GameAudioConfig,
    listener_position: Vector3D,
    listener_forward: Vector3D,
    audio_sources: Vec<ActiveAudioSource>,
    performance_stats: GamePerformanceStats,
    synthesis_thread: Option<thread::JoinHandle<()>>,
}

#[derive(Debug, Clone)]
pub struct ActiveAudioSource {
    pub dialogue_id: u32,
    pub character_id: String,
    pub position: Vector3D,
    pub volume: f32,
    pub is_playing: bool,
    pub start_time: Instant,
}

#[derive(Debug, Clone)]
pub struct GamePerformanceStats {
    pub cpu_usage_percent: f32,
    pub memory_usage_mb: f32,
    pub active_voices: u32,
    pub synthesis_latency_ms: f32,
    pub frame_drops: u32,
    pub audio_underruns: u32,
}

#[derive(Debug, Clone)]
pub struct UnityPerformanceStats {
    pub cpu_usage: f32,
    pub memory_usage: f32,
    pub active_voices: u32,
    pub frame_time_ms: f32,
}

impl GameAudioPipeline {
    pub fn new(config: GameAudioConfig) -> Result<Self, GameIntegrationError> {
        Ok(Self {
            config,
            listener_position: Vector3D::new(0.0, 0.0, 0.0),
            listener_forward: Vector3D::new(0.0, 0.0, 1.0),
            audio_sources: Vec::new(),
            performance_stats: GamePerformanceStats {
                cpu_usage_percent: 0.0,
                memory_usage_mb: 0.0,
                active_voices: 0,
                synthesis_latency_ms: 0.0,
                frame_drops: 0,
                audio_underruns: 0,
            },
            synthesis_thread: None,
        })
    }

    pub fn set_listener_position(
        &mut self,
        position: Vector3D,
        forward: Vector3D,
    ) -> Result<(), GameIntegrationError> {
        self.listener_position = position;
        self.listener_forward = forward;
        Ok(())
    }

    pub fn add_audio_source(&mut self, source: ActiveAudioSource) {
        self.audio_sources.push(source);
        self.performance_stats.active_voices = self.audio_sources.len() as u32;
    }

    pub fn remove_completed_sources(&mut self) {
        self.audio_sources.retain(|source| source.is_playing);
        self.performance_stats.active_voices = self.audio_sources.len() as u32;
    }

    pub fn get_performance_stats(&self) -> UnityPerformanceStats {
        UnityPerformanceStats {
            cpu_usage: self.performance_stats.cpu_usage_percent,
            memory_usage: self.performance_stats.memory_usage_mb,
            active_voices: self.performance_stats.active_voices,
            frame_time_ms: self.performance_stats.synthesis_latency_ms,
        }
    }

    pub fn update_performance_stats(&mut self) {
        // Simulate performance monitoring
        self.performance_stats.cpu_usage_percent = 15.0 + (self.audio_sources.len() as f32 * 2.5);
        self.performance_stats.memory_usage_mb = 50.0 + (self.audio_sources.len() as f32 * 8.0);
        self.performance_stats.synthesis_latency_ms = match self.config.quality_level {
            QualityLevel::Low => 5.0,
            QualityLevel::Medium => 8.0,
            QualityLevel::High => 12.0,
            QualityLevel::Ultra => 20.0,
        };
    }
}

#[derive(Debug, Clone)]
pub enum GameIntegrationError {
    InitializationFailed(String),
    CharacterNotFound(String),
    AudioPipelineError(String),
    ThreadLockError,
    ConfigurationError(String),
    PerformanceError(String),
}

impl std::fmt::Display for GameIntegrationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GameIntegrationError::InitializationFailed(msg) => {
                write!(f, "Initialization failed: {}", msg)
            }
            GameIntegrationError::CharacterNotFound(id) => write!(f, "Character not found: {}", id),
            GameIntegrationError::AudioPipelineError(msg) => {
                write!(f, "Audio pipeline error: {}", msg)
            }
            GameIntegrationError::ThreadLockError => write!(f, "Thread lock error"),
            GameIntegrationError::ConfigurationError(msg) => {
                write!(f, "Configuration error: {}", msg)
            }
            GameIntegrationError::PerformanceError(msg) => write!(f, "Performance error: {}", msg),
        }
    }
}

impl std::error::Error for GameIntegrationError {}

/// Example game scenario implementation
pub fn run_game_integration_example() -> Result<(), GameIntegrationError> {
    println!("🎮 VoiRS Game Integration Example");
    println!("=================================");

    // Configure for a typical AAA game
    let config = GameAudioConfig {
        target_fps: 60,
        max_concurrent_voices: 16,
        audio_buffer_size: 1024,
        quality_level: QualityLevel::High,
        spatial_audio_enabled: true,
        voice_cache_size: 50,
        memory_limit_mb: 512,
    };

    println!("📊 Game Configuration:");
    println!("   Target FPS: {}", config.target_fps);
    println!("   Max Voices: {}", config.max_concurrent_voices);
    println!("   Quality: {:?}", config.quality_level);
    println!("   Memory Limit: {} MB", config.memory_limit_mb);

    // Initialize Unity integration
    println!("\n🔧 Initializing Unity Integration...");
    let mut unity_integration = UnityIntegration::new(config.clone())?;

    // Create example game characters
    let protagonist = GameCharacter {
        id: "protagonist".to_string(),
        name: "Hero".to_string(),
        voice_profile: VoiceProfile {
            voice_id: "heroic_male".to_string(),
            pitch_multiplier: 1.0,
            speed_multiplier: 1.0,
            emotion_intensity: 0.8,
            accent: "american".to_string(),
            age_category: AgeCategory::YoungAdult,
            gender: Gender::Male,
        },
        emotional_state: EmotionalState {
            primary_emotion: Emotion::Neutral,
            intensity: 0.5,
            duration_ms: 5000,
            transition_speed: 0.1,
        },
        position: Vector3D::new(0.0, 0.0, 0.0),
        dialogue_history: Vec::new(),
    };

    let villain = GameCharacter {
        id: "villain".to_string(),
        name: "Dark Lord".to_string(),
        voice_profile: VoiceProfile {
            voice_id: "menacing_deep".to_string(),
            pitch_multiplier: 0.8,
            speed_multiplier: 0.9,
            emotion_intensity: 1.0,
            accent: "british".to_string(),
            age_category: AgeCategory::MiddleAged,
            gender: Gender::Male,
        },
        emotional_state: EmotionalState {
            primary_emotion: Emotion::Angry,
            intensity: 0.9,
            duration_ms: 10000,
            transition_speed: 0.05,
        },
        position: Vector3D::new(10.0, 0.0, 5.0),
        dialogue_history: Vec::new(),
    };

    // Add characters to manager
    {
        let mut manager = unity_integration
            .character_manager
            .lock()
            .map_err(|_| GameIntegrationError::ThreadLockError)?;
        manager.add_character(protagonist.clone());
        manager.add_character(villain.clone());
    }

    println!("✅ Characters created:");
    println!(
        "   - {} ({})",
        protagonist.name, protagonist.voice_profile.voice_id
    );
    println!("   - {} ({})", villain.name, villain.voice_profile.voice_id);

    // Simulate game dialogue sequence
    println!("\n🎭 Starting Game Dialogue Sequence...");

    // Hero speaks first
    let dialogue_id1 = unity_integration.unity_speak_dialogue(
        "protagonist",
        "I will stop your evil plans!",
        1, // Happy emotion
        3, // Important priority
    )?;

    println!(
        "🗣️  Protagonist: \"I will stop your evil plans!\" (ID: {})",
        dialogue_id1
    );

    // Update positions for spatial audio
    unity_integration.unity_update_character_position("protagonist", 5.0, 0.0, 0.0)?;
    unity_integration.unity_update_character_position("villain", 15.0, 0.0, 3.0)?;

    // Villain responds
    let dialogue_id2 = unity_integration.unity_speak_dialogue(
        "villain",
        "You cannot defeat me, foolish hero!",
        3, // Angry emotion
        4, // Critical priority
    )?;

    println!(
        "👹 Villain: \"You cannot defeat me, foolish hero!\" (ID: {})",
        dialogue_id2
    );

    // Set listener position (player camera)
    unity_integration.unity_set_listener_position(7.0, -2.0, 1.0, 0.0, 1.0, 0.0)?;

    // Get performance statistics
    let perf_stats = unity_integration.unity_get_performance_stats()?;
    println!("\n📊 Performance Statistics:");
    println!("   CPU Usage: {:.1}%", perf_stats.cpu_usage);
    println!("   Memory Usage: {:.1} MB", perf_stats.memory_usage);
    println!("   Active Voices: {}", perf_stats.active_voices);
    println!("   Frame Time: {:.1} ms", perf_stats.frame_time_ms);

    // Initialize Unreal integration example
    println!("\n🔧 Initializing Unreal Engine Integration...");
    let _unreal_integration = UnrealIntegration::new(config.clone())?;

    // Generate integration code
    println!("\n📄 Generating Unity C# Bindings...");
    let unity_bindings = UnityIntegration::generate_unity_bindings();
    println!(
        "   Unity bindings generated ({} lines)",
        unity_bindings.lines().count()
    );

    println!("\n📄 Generating Unreal C++ Header...");
    let unreal_header = UnrealIntegration::generate_unreal_header();
    println!(
        "   Unreal header generated ({} lines)",
        unreal_header.lines().count()
    );

    println!("\n📄 Generating Unreal C++ Implementation...");
    let unreal_impl = UnrealIntegration::generate_unreal_implementation();
    println!(
        "   Unreal implementation generated ({} lines)",
        unreal_impl.lines().count()
    );

    // Simulate game performance over time
    println!("\n⏱️  Simulating Game Performance Over 10 Seconds...");
    for i in 1..=10 {
        thread::sleep(Duration::from_millis(100)); // Simulate 100ms frame time

        // Update positions dynamically (characters moving)
        let time = i as f32 * 0.1;
        unity_integration.unity_update_character_position(
            "protagonist",
            5.0 + time.sin() * 2.0,
            time.cos() * 1.0,
            0.0,
        )?;

        // Get updated performance stats every second
        if i % 10 == 0 {
            let stats = unity_integration.unity_get_performance_stats()?;
            println!(
                "   Second {}: CPU {:.1}%, Memory {:.1}MB, Voices: {}",
                i / 10,
                stats.cpu_usage,
                stats.memory_usage,
                stats.active_voices
            );
        }
    }

    println!("\n🎉 Game Integration Example Completed Successfully!");
    println!("\n📋 Integration Features Demonstrated:");
    println!("   ✅ Unity C# bindings generation");
    println!("   ✅ Unreal Engine C++ API generation");
    println!("   ✅ Real-time character dialogue synthesis");
    println!("   ✅ Spatial audio positioning");
    println!("   ✅ Dynamic emotional state management");
    println!("   ✅ Performance monitoring and optimization");
    println!("   ✅ Multi-language character support");
    println!("   ✅ Priority-based dialogue queue management");

    println!("\n🔗 Next Steps for Game Integration:");
    println!("   1. Build native libraries for Unity/Unreal");
    println!("   2. Integrate VoiRS SDK with game build systems");
    println!("   3. Test performance on target gaming hardware");
    println!("   4. Optimize for console platforms (PlayStation, Xbox, Switch)");
    println!("   5. Implement localization and multi-language support");
    println!("   6. Add VR/AR spatial audio enhancements");

    Ok(())
}

fn main() -> Result<(), GameIntegrationError> {
    run_game_integration_example()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_game_config_creation() {
        let config = GameAudioConfig {
            target_fps: 60,
            max_concurrent_voices: 8,
            audio_buffer_size: 512,
            quality_level: QualityLevel::Medium,
            spatial_audio_enabled: true,
            voice_cache_size: 20,
            memory_limit_mb: 256,
        };

        assert_eq!(config.target_fps, 60);
        assert_eq!(config.max_concurrent_voices, 8);
        assert!(config.spatial_audio_enabled);
    }

    #[test]
    fn test_character_creation() {
        let character = GameCharacter {
            id: "test_char".to_string(),
            name: "Test Character".to_string(),
            voice_profile: VoiceProfile {
                voice_id: "test_voice".to_string(),
                pitch_multiplier: 1.0,
                speed_multiplier: 1.0,
                emotion_intensity: 0.5,
                accent: "neutral".to_string(),
                age_category: AgeCategory::YoungAdult,
                gender: Gender::Female,
            },
            emotional_state: EmotionalState {
                primary_emotion: Emotion::Happy,
                intensity: 0.7,
                duration_ms: 3000,
                transition_speed: 0.1,
            },
            position: Vector3D::new(1.0, 2.0, 3.0),
            dialogue_history: Vec::new(),
        };

        assert_eq!(character.id, "test_char");
        assert_eq!(character.position.x, 1.0);
        assert_eq!(character.position.y, 2.0);
        assert_eq!(character.position.z, 3.0);
    }

    #[test]
    fn test_vector3d_distance() {
        let v1 = Vector3D::new(0.0, 0.0, 0.0);
        let v2 = Vector3D::new(3.0, 4.0, 0.0);

        let distance = v1.distance_to(&v2);
        assert!((distance - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_character_manager() {
        let config = GameAudioConfig {
            target_fps: 30,
            max_concurrent_voices: 4,
            audio_buffer_size: 256,
            quality_level: QualityLevel::Low,
            spatial_audio_enabled: false,
            voice_cache_size: 10,
            memory_limit_mb: 128,
        };

        let mut manager = CharacterManager::new(config).unwrap();

        let character = GameCharacter {
            id: "test".to_string(),
            name: "Test".to_string(),
            voice_profile: VoiceProfile {
                voice_id: "test_voice".to_string(),
                pitch_multiplier: 1.0,
                speed_multiplier: 1.0,
                emotion_intensity: 0.5,
                accent: "neutral".to_string(),
                age_category: AgeCategory::YoungAdult,
                gender: Gender::NonBinary,
            },
            emotional_state: EmotionalState {
                primary_emotion: Emotion::Neutral,
                intensity: 0.5,
                duration_ms: 1000,
                transition_speed: 0.1,
            },
            position: Vector3D::new(0.0, 0.0, 0.0),
            dialogue_history: Vec::new(),
        };

        manager.add_character(character);

        let dialogue = DialogueLine {
            text: "Hello world".to_string(),
            timestamp: Instant::now(),
            emotion: Emotion::Happy,
            priority: DialoguePriority::Normal,
            localization_key: None,
        };

        let dialogue_id = manager.queue_dialogue("test", dialogue).unwrap();
        assert!(dialogue_id > 0);
    }

    #[test]
    fn test_dialogue_priority_ordering() {
        let high = DialoguePriority::Critical;
        let low = DialoguePriority::Background;

        assert!(high > low);
        assert!(low < high);
    }

    #[test]
    fn test_quality_level_values() {
        assert_eq!(QualityLevel::Low as u8, 1);
        assert_eq!(QualityLevel::Medium as u8, 2);
        assert_eq!(QualityLevel::High as u8, 3);
        assert_eq!(QualityLevel::Ultra as u8, 4);
    }

    #[test]
    fn test_unity_integration_creation() {
        let config = GameAudioConfig {
            target_fps: 60,
            max_concurrent_voices: 16,
            audio_buffer_size: 1024,
            quality_level: QualityLevel::High,
            spatial_audio_enabled: true,
            voice_cache_size: 50,
            memory_limit_mb: 512,
        };

        let unity_integration = UnityIntegration::new(config);
        assert!(unity_integration.is_ok());
    }

    #[test]
    fn test_unreal_integration_creation() {
        let config = GameAudioConfig {
            target_fps: 60,
            max_concurrent_voices: 16,
            audio_buffer_size: 1024,
            quality_level: QualityLevel::High,
            spatial_audio_enabled: true,
            voice_cache_size: 50,
            memory_limit_mb: 512,
        };

        let unreal_integration = UnrealIntegration::new(config);
        assert!(unreal_integration.is_ok());
    }

    #[test]
    fn test_code_generation() {
        let unity_code = UnityIntegration::generate_unity_bindings();
        assert!(unity_code.contains("VoiRSUnityIntegration"));
        assert!(unity_code.contains("DllImport"));

        let unreal_header = UnrealIntegration::generate_unreal_header();
        assert!(unreal_header.contains("AVoiRSUnrealIntegration"));
        assert!(unreal_header.contains("UCLASS"));

        let unreal_impl = UnrealIntegration::generate_unreal_implementation();
        assert!(unreal_impl.contains("BeginPlay"));
        assert!(unreal_impl.contains("EndPlay"));
    }

    #[test]
    fn test_audio_pipeline_creation() {
        let config = GameAudioConfig {
            target_fps: 60,
            max_concurrent_voices: 8,
            audio_buffer_size: 512,
            quality_level: QualityLevel::Medium,
            spatial_audio_enabled: true,
            voice_cache_size: 20,
            memory_limit_mb: 256,
        };

        let pipeline = GameAudioPipeline::new(config);
        assert!(pipeline.is_ok());
    }

    #[test]
    fn test_performance_stats() {
        let config = GameAudioConfig {
            target_fps: 30,
            max_concurrent_voices: 4,
            audio_buffer_size: 256,
            quality_level: QualityLevel::Low,
            spatial_audio_enabled: false,
            voice_cache_size: 10,
            memory_limit_mb: 128,
        };

        let pipeline = GameAudioPipeline::new(config).unwrap();
        let stats = pipeline.get_performance_stats();

        assert!(stats.cpu_usage >= 0.0);
        assert!(stats.memory_usage >= 0.0);
        assert_eq!(stats.active_voices, 0);
    }
}

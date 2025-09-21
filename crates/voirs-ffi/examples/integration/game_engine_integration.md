# Game Engine Integration Examples

This document provides comprehensive examples for integrating VoiRS FFI with popular game engines for dynamic voice synthesis, character dialogue, and narrative systems.

## Table of Contents

1. [Unity Integration](#unity-integration)
2. [Unreal Engine Integration](#unreal-engine-integration)
3. [Godot Integration](#godot-integration)
4. [Custom Engine Integration](#custom-engine-integration)
5. [Performance Optimization](#performance-optimization)

## Unity Integration

### Basic Setup

#### C# Wrapper Script (VoiRSManager.cs)

```csharp
using System;
using System.Runtime.InteropServices;
using UnityEngine;
using UnityEngine.Audio;

public class VoiRSManager : MonoBehaviour
{
    [Header("VoiRS Configuration")]
    [SerializeField] private VoiRSQuality quality = VoiRSQuality.High;
    [SerializeField] private int maxConcurrentSynthesis = 4;
    [SerializeField] private bool useAudioMixer = true;
    [SerializeField] private AudioMixerGroup dialogueMixerGroup;
    
    private static VoiRSManager instance;
    private IntPtr pipeline;
    private bool initialized = false;
    
    // P/Invoke declarations
    [DllImport("voirs_ffi")]
    private static extern IntPtr voirs_pipeline_create(ref VoiRSSynthesisConfig config);
    
    [DllImport("voirs_ffi")]
    private static extern IntPtr voirs_synthesize(IntPtr pipeline, string text, IntPtr config);
    
    [DllImport("voirs_ffi")]
    private static extern void voirs_synthesis_result_destroy(IntPtr result);
    
    [DllImport("voirs_ffi")]
    private static extern void voirs_pipeline_destroy(IntPtr pipeline);
    
    [DllImport("voirs_ffi")]
    private static extern IntPtr voirs_synthesis_result_get_audio_data(IntPtr result);
    
    [DllImport("voirs_ffi")]
    private static extern int voirs_synthesis_result_get_audio_size(IntPtr result);
    
    [DllImport("voirs_ffi")]
    private static extern int voirs_synthesis_result_get_sample_rate(IntPtr result);
    
    public static VoiRSManager Instance => instance;
    
    [Serializable]
    public struct VoiRSSynthesisConfig
    {
        public VoiRSQuality quality;
        public float speed;
        public float volume;
        public VoiRSFormat outputFormat;
        public string voiceId;
    }
    
    public enum VoiRSQuality
    {
        Low = 0,
        Medium = 1,
        High = 2,
        Ultra = 3
    }
    
    public enum VoiRSFormat
    {
        WAV = 0,
        MP3 = 1,
        FLAC = 2
    }
    
    private void Awake()
    {
        if (instance == null)
        {
            instance = this;
            DontDestroyOnLoad(gameObject);
            InitializeVoiRS();
        }
        else
        {
            Destroy(gameObject);
        }
    }
    
    private void InitializeVoiRS()
    {
        var config = new VoiRSSynthesisConfig
        {
            quality = quality,
            speed = 1.0f,
            volume = 1.0f,
            outputFormat = VoiRSFormat.WAV,
            voiceId = "default"
        };
        
        pipeline = voirs_pipeline_create(ref config);
        initialized = pipeline != IntPtr.Zero;
        
        if (!initialized)
        {
            Debug.LogError("Failed to initialize VoiRS pipeline");
        }
        else
        {
            Debug.Log("VoiRS initialized successfully");
        }
    }
    
    public void SynthesizeDialogue(string text, string characterVoice, 
                                 Action<AudioClip> onComplete, Action<string> onError = null)
    {
        if (!initialized)
        {
            onError?.Invoke("VoiRS not initialized");
            return;
        }
        
        StartCoroutine(SynthesizeCoroutine(text, characterVoice, onComplete, onError));
    }
    
    private System.Collections.IEnumerator SynthesizeCoroutine(string text, string voiceId,
                                                              Action<AudioClip> onComplete, 
                                                              Action<string> onError)
    {
        // Perform synthesis on background thread
        AudioClip resultClip = null;
        string error = null;
        bool completed = false;
        
        var thread = new System.Threading.Thread(() =>
        {
            try
            {
                var result = voirs_synthesize(pipeline, text, IntPtr.Zero);
                if (result != IntPtr.Zero)
                {
                    var audioData = voirs_synthesis_result_get_audio_data(result);
                    var audioSize = voirs_synthesis_result_get_audio_size(result);
                    var sampleRate = voirs_synthesis_result_get_sample_rate(result);
                    
                    // Convert to Unity AudioClip on main thread
                    var samples = new float[audioSize / sizeof(float)];
                    Marshal.Copy(audioData, samples, 0, samples.Length);
                    
                    UnityMainThreadDispatcher.Instance.Enqueue(() =>
                    {
                        resultClip = AudioClip.Create("VoiRS_Synthesis", 
                                                    samples.Length, 1, sampleRate, false);
                        resultClip.SetData(samples, 0);
                    });
                    
                    voirs_synthesis_result_destroy(result);
                }
                else
                {
                    error = "Synthesis failed";
                }
            }
            catch (Exception e)
            {
                error = e.Message;
            }
            finally
            {
                completed = true;
            }
        });
        
        thread.Start();
        
        // Wait for completion
        while (!completed)
        {
            yield return null;
        }
        
        if (error != null)
        {
            onError?.Invoke(error);
        }
        else
        {
            onComplete?.Invoke(resultClip);
        }
    }
    
    private void OnDestroy()
    {
        if (initialized && pipeline != IntPtr.Zero)
        {
            voirs_pipeline_destroy(pipeline);
        }
    }
}
```

#### Character Dialogue System (DialogueSystem.cs)

```csharp
using UnityEngine;
using UnityEngine.UI;
using System.Collections.Generic;

public class DialogueSystem : MonoBehaviour
{
    [Header("UI Components")]
    [SerializeField] private Text dialogueText;
    [SerializeField] private Text characterNameText;
    [SerializeField] private Button nextButton;
    [SerializeField] private AudioSource audioSource;
    
    [Header("Character Voices")]
    [SerializeField] private CharacterVoiceMap[] characterVoices;
    
    private Queue<DialogueEntry> dialogueQueue = new Queue<DialogueEntry>();
    private bool isPlaying = false;
    
    [System.Serializable]
    public struct CharacterVoiceMap
    {
        public string characterName;
        public string voiceId;
        public float speedModifier;
        public float pitchModifier;
    }
    
    [System.Serializable]
    public struct DialogueEntry
    {
        public string characterName;
        public string text;
        public Sprite characterPortrait;
    }
    
    private void Start()
    {
        nextButton.onClick.AddListener(PlayNextDialogue);
    }
    
    public void StartDialogue(DialogueEntry[] dialogue)
    {
        dialogueQueue.Clear();
        
        foreach (var entry in dialogue)
        {
            dialogueQueue.Enqueue(entry);
        }
        
        PlayNextDialogue();
    }
    
    private void PlayNextDialogue()
    {
        if (isPlaying) return;
        
        if (dialogueQueue.Count == 0)
        {
            EndDialogue();
            return;
        }
        
        var entry = dialogueQueue.Dequeue();
        var voice = GetVoiceForCharacter(entry.characterName);
        
        characterNameText.text = entry.characterName;
        dialogueText.text = "..."; // Show loading
        
        isPlaying = true;
        nextButton.interactable = false;
        
        VoiRSManager.Instance.SynthesizeDialogue(
            entry.text,
            voice.voiceId,
            OnSynthesisComplete,
            OnSynthesisError
        );
    }
    
    private CharacterVoiceMap GetVoiceForCharacter(string characterName)
    {
        foreach (var voice in characterVoices)
        {
            if (voice.characterName == characterName)
                return voice;
        }
        
        // Return default voice if not found
        return new CharacterVoiceMap
        {
            characterName = characterName,
            voiceId = "default",
            speedModifier = 1.0f,
            pitchModifier = 1.0f
        };
    }
    
    private void OnSynthesisComplete(AudioClip clip)
    {
        dialogueText.text = dialogueQueue.Count > 0 ? 
                           dialogueQueue.Peek().text : 
                           "End of dialogue";
        
        audioSource.clip = clip;
        audioSource.Play();
        
        // Enable next button after audio starts
        StartCoroutine(EnableNextButtonAfterAudio());
    }
    
    private System.Collections.IEnumerator EnableNextButtonAfterAudio()
    {
        yield return new WaitForSeconds(0.5f); // Brief delay
        
        nextButton.interactable = true;
        isPlaying = false;
        
        // Auto-advance after audio finishes
        yield return new WaitUntil(() => !audioSource.isPlaying);
        yield return new WaitForSeconds(1.0f); // Pause before next
        
        if (dialogueQueue.Count > 0)
        {
            PlayNextDialogue();
        }
    }
    
    private void OnSynthesisError(string error)
    {
        Debug.LogError($"Dialogue synthesis error: {error}");
        dialogueText.text = $"Error: {error}";
        
        isPlaying = false;
        nextButton.interactable = true;
    }
    
    private void EndDialogue()
    {
        gameObject.SetActive(false);
        // Trigger any end-of-dialogue events
    }
}
```

### Unity Package Integration

#### Package.json

```json
{
  "name": "com.voirs.ffi.unity",
  "version": "0.1.0",
  "displayName": "VoiRS FFI for Unity",
  "description": "Real-time voice synthesis for Unity games",
  "unity": "2021.3",
  "unityRelease": "0f1",
  "keywords": [
    "audio",
    "voice",
    "synthesis",
    "tts",
    "game-audio"
  ],
  "author": {
    "name": "VoiRS Team"
  },
  "dependencies": {},
  "samples": [
    {
      "displayName": "Basic Dialogue System",
      "description": "Simple character dialogue with voice synthesis",
      "path": "Samples~/BasicDialogue"
    },
    {
      "displayName": "Dynamic Narration",
      "description": "Procedural narrative voice generation",
      "path": "Samples~/DynamicNarration"
    }
  ]
}
```

## Unreal Engine Integration

### Blueprint Integration

#### VoiRS Component (C++)

```cpp
// VoiRSComponent.h
#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "Sound/SoundWave.h"
#include "VoiRSComponent.generated.h"

UENUM(BlueprintType)
enum class EVoiRSQuality : uint8
{
    Low = 0,
    Medium = 1,
    High = 2,
    Ultra = 3
};

USTRUCT(BlueprintType)
struct FVoiRSConfig
{
    GENERATED_BODY()
    
    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    EVoiRSQuality Quality = EVoiRSQuality::High;
    
    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    float Speed = 1.0f;
    
    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    float Volume = 1.0f;
    
    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    FString VoiceId = TEXT("default");
};

DECLARE_DYNAMIC_MULTICAST_DELEGATE_TwoParams(FOnVoiRSSynthesisComplete, 
    class USoundWave*, GeneratedAudio, bool, bSuccess);

UCLASS(ClassGroup=(Audio), meta=(BlueprintSpawnableComponent))
class YOURGAME_API UVoiRSComponent : public UActorComponent
{
    GENERATED_BODY()

public:
    UVoiRSComponent();

protected:
    virtual void BeginPlay() override;
    virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;

public:
    UFUNCTION(BlueprintCallable, Category = "VoiRS")
    void SynthesizeText(const FString& Text, const FVoiRSConfig& Config);
    
    UFUNCTION(BlueprintCallable, Category = "VoiRS")
    void SynthesizeTextAsync(const FString& Text, const FVoiRSConfig& Config);
    
    UFUNCTION(BlueprintCallable, Category = "VoiRS")
    bool IsInitialized() const { return bInitialized; }
    
    UPROPERTY(BlueprintAssignable)
    FOnVoiRSSynthesisComplete OnSynthesisComplete;

private:
    void InitializeVoiRS();
    void CleanupVoiRS();
    
    USoundWave* CreateSoundWave(const TArray<uint8>& AudioData, 
                               int32 SampleRate, int32 NumChannels);
    
    void ProcessSynthesisResult(void* Result);
    
    UPROPERTY(EditAnywhere, Category = "VoiRS")
    FVoiRSConfig DefaultConfig;
    
    void* Pipeline = nullptr;
    bool bInitialized = false;
    
    // Thread safety
    FCriticalSection SynthesisMutex;
    TQueue<TPair<FString, FVoiRSConfig>> SynthesisQueue;
};
```

#### Implementation (VoiRSComponent.cpp)

```cpp
// VoiRSComponent.cpp
#include "VoiRSComponent.h"
#include "Sound/SoundWave.h"
#include "Engine/Engine.h"
#include "Async/Async.h"
#include "HAL/PlatformFilemanager.h"

// VoiRS FFI function declarations
extern "C" 
{
    void* voirs_pipeline_create(const struct VoiRSSynthesisConfig* config);
    void* voirs_synthesize(void* pipeline, const char* text, const struct VoiRSSynthesisConfig* config);
    void voirs_synthesis_result_destroy(void* result);
    void voirs_pipeline_destroy(void* pipeline);
    const void* voirs_synthesis_result_get_audio_data(void* result);
    int32 voirs_synthesis_result_get_audio_size(void* result);
    int32 voirs_synthesis_result_get_sample_rate(void* result);
    int32 voirs_synthesis_result_get_channels(void* result);
}

struct VoiRSSynthesisConfig
{
    int32 quality;
    float speed;
    float volume;
    int32 output_format;
    const char* voice_id;
};

UVoiRSComponent::UVoiRSComponent()
{
    PrimaryComponentTick.bCanEverTick = false;
    
    DefaultConfig.Quality = EVoiRSQuality::High;
    DefaultConfig.Speed = 1.0f;
    DefaultConfig.Volume = 1.0f;
    DefaultConfig.VoiceId = TEXT("default");
}

void UVoiRSComponent::BeginPlay()
{
    Super::BeginPlay();
    InitializeVoiRS();
}

void UVoiRSComponent::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
    CleanupVoiRS();
    Super::EndPlay(EndPlayReason);
}

void UVoiRSComponent::InitializeVoiRS()
{
    VoiRSSynthesisConfig Config;
    Config.quality = static_cast<int32>(DefaultConfig.Quality);
    Config.speed = DefaultConfig.Speed;
    Config.volume = DefaultConfig.Volume;
    Config.output_format = 0; // WAV
    
    FString VoiceIdAnsi = DefaultConfig.VoiceId;
    Config.voice_id = TCHAR_TO_ANSI(*VoiceIdAnsi);
    
    Pipeline = voirs_pipeline_create(&Config);
    bInitialized = (Pipeline != nullptr);
    
    if (!bInitialized)
    {
        UE_LOG(LogTemp, Error, TEXT("Failed to initialize VoiRS pipeline"));
    }
    else
    {
        UE_LOG(LogTemp, Log, TEXT("VoiRS pipeline initialized successfully"));
    }
}

void UVoiRSComponent::CleanupVoiRS()
{
    if (Pipeline)
    {
        voirs_pipeline_destroy(Pipeline);
        Pipeline = nullptr;
    }
    bInitialized = false;
}

void UVoiRSComponent::SynthesizeText(const FString& Text, const FVoiRSConfig& Config)
{
    if (!bInitialized)
    {
        UE_LOG(LogTemp, Error, TEXT("VoiRS not initialized"));
        OnSynthesisComplete.Broadcast(nullptr, false);
        return;
    }
    
    VoiRSSynthesisConfig NativeConfig;
    NativeConfig.quality = static_cast<int32>(Config.Quality);
    NativeConfig.speed = Config.Speed;
    NativeConfig.volume = Config.Volume;
    NativeConfig.output_format = 0; // WAV
    
    FString VoiceIdAnsi = Config.VoiceId;
    NativeConfig.voice_id = TCHAR_TO_ANSI(*VoiceIdAnsi);
    
    FString TextAnsi = Text;
    const char* TextPtr = TCHAR_TO_ANSI(*TextAnsi);
    
    void* Result = voirs_synthesize(Pipeline, TextPtr, &NativeConfig);
    
    if (Result)
    {
        ProcessSynthesisResult(Result);
        voirs_synthesis_result_destroy(Result);
    }
    else
    {
        UE_LOG(LogTemp, Error, TEXT("Synthesis failed"));
        OnSynthesisComplete.Broadcast(nullptr, false);
    }
}

void UVoiRSComponent::SynthesizeTextAsync(const FString& Text, const FVoiRSConfig& Config)
{
    if (!bInitialized)
    {
        UE_LOG(LogTemp, Error, TEXT("VoiRS not initialized"));
        OnSynthesisComplete.Broadcast(nullptr, false);
        return;
    }
    
    // Capture by value for async execution
    AsyncTask(ENamedThreads::AnyBackgroundHiPriTask, [this, Text, Config]()
    {
        VoiRSSynthesisConfig NativeConfig;
        NativeConfig.quality = static_cast<int32>(Config.Quality);
        NativeConfig.speed = Config.Speed;
        NativeConfig.volume = Config.Volume;
        NativeConfig.output_format = 0; // WAV
        
        FString VoiceIdAnsi = Config.VoiceId;
        NativeConfig.voice_id = TCHAR_TO_ANSI(*VoiceIdAnsi);
        
        FString TextAnsi = Text;
        const char* TextPtr = TCHAR_TO_ANSI(*TextAnsi);
        
        void* Result = voirs_synthesize(Pipeline, TextPtr, &NativeConfig);
        
        // Switch back to game thread for result processing
        AsyncTask(ENamedThreads::GameThread, [this, Result]()
        {
            if (Result)
            {
                ProcessSynthesisResult(Result);
                voirs_synthesis_result_destroy(Result);
            }
            else
            {
                UE_LOG(LogTemp, Error, TEXT("Async synthesis failed"));
                OnSynthesisComplete.Broadcast(nullptr, false);
            }
        });
    });
}

void UVoiRSComponent::ProcessSynthesisResult(void* Result)
{
    const void* AudioData = voirs_synthesis_result_get_audio_data(Result);
    int32 AudioSize = voirs_synthesis_result_get_audio_size(Result);
    int32 SampleRate = voirs_synthesis_result_get_sample_rate(Result);
    int32 NumChannels = voirs_synthesis_result_get_channels(Result);
    
    if (AudioData && AudioSize > 0)
    {
        TArray<uint8> AudioBytes;
        AudioBytes.AddUninitialized(AudioSize);
        FMemory::Memcpy(AudioBytes.GetData(), AudioData, AudioSize);
        
        USoundWave* SoundWave = CreateSoundWave(AudioBytes, SampleRate, NumChannels);
        OnSynthesisComplete.Broadcast(SoundWave, true);
    }
    else
    {
        OnSynthesisComplete.Broadcast(nullptr, false);
    }
}

USoundWave* UVoiRSComponent::CreateSoundWave(const TArray<uint8>& AudioData, 
                                           int32 SampleRate, int32 NumChannels)
{
    USoundWave* SoundWave = NewObject<USoundWave>();
    
    SoundWave->SetSampleRate(SampleRate);
    SoundWave->NumChannels = NumChannels;
    SoundWave->Duration = AudioData.Num() / (sizeof(int16) * NumChannels * SampleRate);
    SoundWave->SetImportedSampleRate(SampleRate);
    SoundWave->TotalSamples = AudioData.Num() / (sizeof(int16) * NumChannels);
    
    // Copy audio data
    SoundWave->RawData.Lock(LOCK_READ_WRITE);
    void* LockedData = SoundWave->RawData.Realloc(AudioData.Num());
    FMemory::Memcpy(LockedData, AudioData.GetData(), AudioData.Num());
    SoundWave->RawData.Unlock();
    
    return SoundWave;
}
```

### Blueprint Nodes

Create custom Blueprint nodes for easier integration:

#### Custom Blueprint Nodes (VoiRSBlueprintLibrary.h)

```cpp
#pragma once

#include "CoreMinimal.h"
#include "Kismet/BlueprintFunctionLibrary.h"
#include "VoiRSComponent.h"
#include "VoiRSBlueprintLibrary.generated.h"

UCLASS()
class YOURGAME_API UVoiRSBlueprintLibrary : public UBlueprintFunctionLibrary
{
    GENERATED_BODY()

public:
    UFUNCTION(BlueprintCallable, Category = "VoiRS|Helpers", 
              meta = (ToolTip = "Create a VoiRS configuration"))
    static FVoiRSConfig MakeVoiRSConfig(EVoiRSQuality Quality, 
                                       float Speed, float Volume, 
                                       const FString& VoiceId);
    
    UFUNCTION(BlueprintCallable, Category = "VoiRS|Helpers",
              meta = (ToolTip = "Get recommended quality for platform"))
    static EVoiRSQuality GetRecommendedQuality();
    
    UFUNCTION(BlueprintCallable, Category = "VoiRS|Character",
              meta = (ToolTip = "Create character-specific voice config"))
    static FVoiRSConfig CreateCharacterVoice(const FString& CharacterName,
                                            float AgeFactor = 1.0f,
                                            float EmotionIntensity = 0.5f);
    
    UFUNCTION(BlueprintCallable, Category = "VoiRS|Utilities",
              meta = (ToolTip = "Estimate synthesis duration"))
    static float EstimateSynthesisDuration(const FString& Text, 
                                         float SpeechRate = 1.0f);
};
```

## Godot Integration

### GDScript Integration

#### VoiRS GDNative Module (voirs_godot.cpp)

```cpp
#include <Godot.hpp>
#include <Node.hpp>
#include <AudioStreamSample.hpp>
#include <AudioStreamPlayer.hpp>
#include <PoolArrays.hpp>

using namespace godot;

class VoiRSNode : public Node {
    GODOT_CLASS(VoiRSNode, Node)

private:
    void* pipeline;
    bool initialized;

public:
    static void _register_methods() {
        register_method("initialize", &VoiRSNode::initialize);
        register_method("synthesize", &VoiRSNode::synthesize);
        register_method("synthesize_async", &VoiRSNode::synthesize_async);
        register_method("is_initialized", &VoiRSNode::is_initialized);
        
        register_property<VoiRSNode, String>("default_voice", 
                                           &VoiRSNode::default_voice, "default");
        register_property<VoiRSNode, int>("quality", 
                                        &VoiRSNode::quality, 2);
        register_property<VoiRSNode, float>("speed", 
                                          &VoiRSNode::speed, 1.0f);
    }

    VoiRSNode() {
        pipeline = nullptr;
        initialized = false;
        default_voice = "default";
        quality = 2; // High quality
        speed = 1.0f;
    }

    ~VoiRSNode() {
        if (pipeline) {
            voirs_pipeline_destroy(pipeline);
        }
    }

    void _init() {
        // Initialize when ready
    }

    void _ready() {
        initialize();
    }

    void initialize() {
        struct VoiRSSynthesisConfig config;
        config.quality = quality;
        config.speed = speed;
        config.volume = 1.0f;
        config.output_format = 0; // WAV
        config.voice_id = default_voice.alloc_c_string();
        
        pipeline = voirs_pipeline_create(&config);
        initialized = (pipeline != nullptr);
        
        if (initialized) {
            Godot::print("VoiRS initialized successfully");
        } else {
            Godot::print_error("Failed to initialize VoiRS", __FUNCTION__, __FILE__, __LINE__);
        }
    }

    Ref<AudioStreamSample> synthesize(String text) {
        if (!initialized) {
            Godot::print_error("VoiRS not initialized", __FUNCTION__, __FILE__, __LINE__);
            return Ref<AudioStreamSample>();
        }

        char* text_cstr = text.alloc_c_string();
        void* result = voirs_synthesize(pipeline, text_cstr, nullptr);
        
        if (!result) {
            Godot::print_error("Synthesis failed", __FUNCTION__, __FILE__, __LINE__);
            api->godot_free(text_cstr);
            return Ref<AudioStreamSample>();
        }

        // Get audio data
        const void* audio_data = voirs_synthesis_result_get_audio_data(result);
        int32_t audio_size = voirs_synthesis_result_get_audio_size(result);
        int32_t sample_rate = voirs_synthesis_result_get_sample_rate(result);
        int32_t channels = voirs_synthesis_result_get_channels(result);

        // Create Godot audio stream
        Ref<AudioStreamSample> stream = AudioStreamSample::_new();
        
        PoolByteArray audio_pool;
        audio_pool.resize(audio_size);
        
        PoolByteArray::Write write = audio_pool.write();
        memcpy(write.ptr(), audio_data, audio_size);
        
        stream->set_data(audio_pool);
        stream->set_format(AudioStreamSample::FORMAT_16_BITS);
        stream->set_mix_rate(sample_rate);
        stream->set_stereo(channels == 2);

        // Cleanup
        voirs_synthesis_result_destroy(result);
        api->godot_free(text_cstr);

        return stream;
    }

    void synthesize_async(String text, String callback_method) {
        if (!initialized) {
            call_deferred(callback_method, Ref<AudioStreamSample>(), false);
            return;
        }

        // In a real implementation, you'd use Godot's thread system
        // For simplicity, this example uses call_deferred
        Ref<AudioStreamSample> result = synthesize(text);
        call_deferred(callback_method, result, result.is_valid());
    }

    bool is_initialized() const {
        return initialized;
    }

    String default_voice;
    int quality;
    float speed;
};

extern "C" void GDN_EXPORT godot_gdnative_init(godot_gdnative_init_options *o) {
    godot::Godot::gdnative_init(o);
}

extern "C" void GDN_EXPORT godot_gdnative_terminate(godot_gdnative_terminate_options *o) {
    godot::Godot::gdnative_terminate(o);
}

extern "C" void GDN_EXPORT godot_nativescript_init(void *handle) {
    godot::Godot::nativescript_init(handle);
    godot::register_class<VoiRSNode>();
}
```

#### Godot Script Example (DialogueSystem.gd)

```gdscript
extends Control

@onready var dialogue_text = $VBoxContainer/DialogueText
@onready var character_name = $VBoxContainer/CharacterName
@onready var audio_player = $AudioStreamPlayer
@onready var next_button = $VBoxContainer/NextButton
@onready var voirs_node = $VoiRSNode

var dialogue_queue = []
var current_dialogue_index = 0
var is_playing = false

var character_voices = {
    "Hero": {"voice_id": "male_young", "speed": 1.0},
    "Villain": {"voice_id": "male_deep", "speed": 0.8},
    "Narrator": {"voice_id": "neutral", "speed": 1.1}
}

func _ready():
    next_button.pressed.connect(_on_next_button_pressed)
    if not voirs_node.is_initialized():
        push_error("VoiRS failed to initialize")

func start_dialogue(dialogue_data: Array):
    dialogue_queue = dialogue_data
    current_dialogue_index = 0
    show_next_dialogue()

func show_next_dialogue():
    if current_dialogue_index >= dialogue_queue.size():
        end_dialogue()
        return
    
    if is_playing:
        return
    
    var dialogue = dialogue_queue[current_dialogue_index]
    var character = dialogue.get("character", "Narrator")
    var text = dialogue.get("text", "")
    
    character_name.text = character
    dialogue_text.text = "..."
    
    is_playing = true
    next_button.disabled = true
    
    # Get voice configuration for character
    var voice_config = character_voices.get(character, {"voice_id": "default", "speed": 1.0})
    
    # Synthesize speech
    voirs_node.synthesize_async(text, "_on_synthesis_complete")

func _on_synthesis_complete(audio_stream: AudioStreamSample, success: bool):
    if success and audio_stream:
        var dialogue = dialogue_queue[current_dialogue_index]
        dialogue_text.text = dialogue.get("text", "")
        
        audio_player.stream = audio_stream
        audio_player.play()
        
        # Wait for audio to finish, then enable next button
        await audio_player.finished
        await get_tree().create_timer(0.5).timeout  # Brief pause
        
        next_button.disabled = false
        is_playing = false
    else:
        push_error("Failed to synthesize dialogue")
        dialogue_text.text = "Error: Failed to generate speech"
        next_button.disabled = false
        is_playing = false

func _on_next_button_pressed():
    current_dialogue_index += 1
    show_next_dialogue()

func end_dialogue():
    hide()
    # Emit signal or call callback to notify dialogue completion
    print("Dialogue sequence completed")
```

## Custom Engine Integration

### Generic C++ Integration

#### Header File (VoiRSIntegration.h)

```cpp
#pragma once

#include <string>
#include <vector>
#include <functional>
#include <memory>
#include <thread>
#include <mutex>
#include <queue>

namespace VoiRS {

enum class Quality {
    Low = 0,
    Medium = 1,
    High = 2,
    Ultra = 3
};

enum class Format {
    WAV = 0,
    MP3 = 1,
    FLAC = 2
};

struct Config {
    Quality quality = Quality::High;
    float speed = 1.0f;
    float volume = 1.0f;
    Format format = Format::WAV;
    std::string voice_id = "default";
};

struct AudioResult {
    std::vector<uint8_t> data;
    int sample_rate;
    int channels;
    float duration;
    bool success;
};

using SynthesisCallback = std::function<void(const AudioResult&)>;

class Engine {
public:
    Engine();
    ~Engine();
    
    bool Initialize(const Config& config = Config{});
    void Shutdown();
    
    // Synchronous synthesis
    AudioResult Synthesize(const std::string& text, const Config& config = Config{});
    
    // Asynchronous synthesis
    void SynthesizeAsync(const std::string& text, SynthesisCallback callback, 
                        const Config& config = Config{});
    
    // Batch synthesis
    std::vector<AudioResult> SynthesizeBatch(const std::vector<std::string>& texts,
                                           const Config& config = Config{});
    
    // Streaming synthesis
    class StreamingSynthesis {
    public:
        StreamingSynthesis(void* pipeline, const Config& config);
        ~StreamingSynthesis();
        
        void WriteText(const std::string& text);
        AudioResult ReadAudio();
        bool IsComplete() const;
        
    private:
        void* stream_handle;
        bool complete;
    };
    
    std::unique_ptr<StreamingSynthesis> CreateStream(const Config& config = Config{});
    
    bool IsInitialized() const { return initialized; }
    
private:
    void* pipeline;
    bool initialized;
    
    // Async processing
    std::thread worker_thread;
    std::mutex queue_mutex;
    std::queue<std::pair<std::string, SynthesisCallback>> async_queue;
    bool stop_worker;
    
    void WorkerThreadFunction();
    void ProcessAsyncRequest(const std::string& text, SynthesisCallback callback);
};

// Utility functions
std::string GetVersion();
std::vector<std::string> GetAvailableVoices();
Config GetOptimalConfig(Quality target_quality, bool real_time = false);

} // namespace VoiRS
```

#### Implementation (VoiRSIntegration.cpp)

```cpp
#include "VoiRSIntegration.h"
#include <iostream>
#include <chrono>

// VoiRS FFI declarations
extern "C" {
    void* voirs_pipeline_create(const struct VoiRSSynthesisConfig* config);
    void* voirs_synthesize(void* pipeline, const char* text, const struct VoiRSSynthesisConfig* config);
    void voirs_synthesis_result_destroy(void* result);
    void voirs_pipeline_destroy(void* pipeline);
    const void* voirs_synthesis_result_get_audio_data(void* result);
    int32_t voirs_synthesis_result_get_audio_size(void* result);
    int32_t voirs_synthesis_result_get_sample_rate(void* result);
    int32_t voirs_synthesis_result_get_channels(void* result);
    const char* voirs_get_version();
}

struct VoiRSSynthesisConfig {
    int32_t quality;
    float speed;
    float volume;
    int32_t output_format;
    const char* voice_id;
};

namespace VoiRS {

Engine::Engine() : pipeline(nullptr), initialized(false), stop_worker(false) {
}

Engine::~Engine() {
    Shutdown();
}

bool Engine::Initialize(const Config& config) {
    if (initialized) {
        return true;
    }
    
    VoiRSSynthesisConfig native_config;
    native_config.quality = static_cast<int32_t>(config.quality);
    native_config.speed = config.speed;
    native_config.volume = config.volume;
    native_config.output_format = static_cast<int32_t>(config.format);
    native_config.voice_id = config.voice_id.c_str();
    
    pipeline = voirs_pipeline_create(&native_config);
    initialized = (pipeline != nullptr);
    
    if (initialized) {
        // Start worker thread for async operations
        worker_thread = std::thread(&Engine::WorkerThreadFunction, this);
        std::cout << "VoiRS Engine initialized successfully" << std::endl;
    } else {
        std::cerr << "Failed to initialize VoiRS Engine" << std::endl;
    }
    
    return initialized;
}

void Engine::Shutdown() {
    if (!initialized) {
        return;
    }
    
    // Stop worker thread
    stop_worker = true;
    if (worker_thread.joinable()) {
        worker_thread.join();
    }
    
    // Cleanup pipeline
    if (pipeline) {
        voirs_pipeline_destroy(pipeline);
        pipeline = nullptr;
    }
    
    initialized = false;
    std::cout << "VoiRS Engine shutdown complete" << std::endl;
}

AudioResult Engine::Synthesize(const std::string& text, const Config& config) {
    AudioResult result;
    result.success = false;
    
    if (!initialized) {
        std::cerr << "Engine not initialized" << std::endl;
        return result;
    }
    
    VoiRSSynthesisConfig native_config;
    native_config.quality = static_cast<int32_t>(config.quality);
    native_config.speed = config.speed;
    native_config.volume = config.volume;
    native_config.output_format = static_cast<int32_t>(config.format);
    native_config.voice_id = config.voice_id.c_str();
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    void* synthesis_result = voirs_synthesize(pipeline, text.c_str(), &native_config);
    
    if (synthesis_result) {
        const void* audio_data = voirs_synthesis_result_get_audio_data(synthesis_result);
        int32_t audio_size = voirs_synthesis_result_get_audio_size(synthesis_result);
        int32_t sample_rate = voirs_synthesis_result_get_sample_rate(synthesis_result);
        int32_t channels = voirs_synthesis_result_get_channels(synthesis_result);
        
        result.data.resize(audio_size);
        std::memcpy(result.data.data(), audio_data, audio_size);
        result.sample_rate = sample_rate;
        result.channels = channels;
        result.success = true;
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time).count();
        
        result.duration = static_cast<float>(duration_ms) / 1000.0f;
        
        voirs_synthesis_result_destroy(synthesis_result);
    }
    
    return result;
}

void Engine::SynthesizeAsync(const std::string& text, SynthesisCallback callback, 
                           const Config& config) {
    if (!initialized) {
        AudioResult error_result;
        error_result.success = false;
        callback(error_result);
        return;
    }
    
    std::lock_guard<std::mutex> lock(queue_mutex);
    async_queue.push({text, callback});
}

void Engine::WorkerThreadFunction() {
    while (!stop_worker) {
        std::unique_lock<std::mutex> lock(queue_mutex);
        
        if (!async_queue.empty()) {
            auto request = async_queue.front();
            async_queue.pop();
            lock.unlock();
            
            ProcessAsyncRequest(request.first, request.second);
        } else {
            lock.unlock();
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
}

void Engine::ProcessAsyncRequest(const std::string& text, SynthesisCallback callback) {
    AudioResult result = Synthesize(text);
    callback(result);
}

std::string GetVersion() {
    return std::string(voirs_get_version());
}

Config GetOptimalConfig(Quality target_quality, bool real_time) {
    Config config;
    config.quality = target_quality;
    
    if (real_time) {
        // Optimize for low latency
        if (target_quality == Quality::Ultra) {
            config.quality = Quality::High; // Step down for real-time
        }
        config.speed = 1.0f; // Normal speed for predictable timing
    }
    
    return config;
}

} // namespace VoiRS
```

## Performance Optimization

### Memory Management

```cpp
// Pool allocator for game engines
class VoiRSAudioPool {
public:
    VoiRSAudioPool(size_t pool_size = 10 * 1024 * 1024) { // 10MB default
        voirs_allocator_config_t config;
        config.type = VOIRS_ALLOCATOR_POOL;
        config.pool_size = pool_size;
        voirs_set_allocator(&config);
    }
    
    ~VoiRSAudioPool() {
        voirs_allocator_config_t config;
        config.type = VOIRS_ALLOCATOR_SYSTEM;
        voirs_set_allocator(&config);
    }
};

// Use RAII for automatic cleanup
class VoiRSResourceManager {
    VoiRSAudioPool pool;
public:
    VoiRSResourceManager() = default;
    // Pool automatically configured and cleaned up
};
```

### Threading Optimization

```cpp
// Game engine integration with work-stealing
void ConfigureForGameEngine() {
    voirs_performance_config_t perf_config;
    perf_config.thread_count = std::thread::hardware_concurrency() - 1; // Leave one for game
    perf_config.use_work_stealing = true;
    perf_config.cpu_affinity_mask = 0xFE; // Avoid core 0 (game thread)
    
    voirs_set_performance_config(&perf_config);
}
```

These examples provide comprehensive integration patterns for major game engines, enabling dynamic voice synthesis, character dialogue systems, and narrative generation with optimal performance for interactive applications.
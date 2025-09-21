# Integration Examples

This guide provides real-world examples of integrating VoiRS Python bindings into various applications and frameworks.

## Table of Contents

1. [Web Applications](#web-applications)
2. [Desktop Applications](#desktop-applications)
3. [API Services](#api-services)
4. [Voice Assistants](#voice-assistants)
5. [Content Management Systems](#content-management-systems)
6. [Machine Learning Pipelines](#machine-learning-pipelines)
7. [Real-time Applications](#real-time-applications)
8. [Batch Processing](#batch-processing)

## Web Applications

### Django Integration

```python
# views.py
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from voirs_ffi import VoirsPipeline
import json
import io

# Initialize pipeline globally (singleton pattern)
pipeline = VoirsPipeline.with_config(use_gpu=True)

@csrf_exempt
def synthesize_text(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        text = data.get('text', '')
        voice = data.get('voice', None)
        
        try:
            # Synthesize text
            audio = pipeline.synthesize(text, voice=voice)
            
            # Return audio as binary response
            response = HttpResponse(
                audio.to_wav_bytes(),
                content_type='audio/wav'
            )
            response['Content-Disposition'] = 'attachment; filename="speech.wav"'
            return response
            
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Method not allowed'}, status=405)

def get_voices(request):
    voices = pipeline.get_voices()
    voice_data = [
        {
            'id': voice.id,
            'name': voice.name,
            'language': voice.language,
            'gender': voice.gender
        }
        for voice in voices
    ]
    return JsonResponse({'voices': voice_data})
```

### Flask Integration

```python
# app.py
from flask import Flask, request, jsonify, send_file
from voirs_ffi import VoirsPipeline
import io
import tempfile

app = Flask(__name__)

# Initialize pipeline
pipeline = VoirsPipeline.with_config(use_gpu=True)

@app.route('/synthesize', methods=['POST'])
def synthesize():
    data = request.get_json()
    text = data.get('text', '')
    voice = data.get('voice', None)
    
    try:
        audio = pipeline.synthesize(text, voice=voice)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            audio.save(tmp.name)
            return send_file(tmp.name, mimetype='audio/wav')
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/voices', methods=['GET'])
def voices():
    voices = pipeline.get_voices()
    return jsonify([
        {
            'id': v.id,
            'name': v.name,
            'language': v.language,
            'gender': v.gender
        }
        for v in voices
    ])

if __name__ == '__main__':
    app.run(debug=True)
```

### FastAPI Integration

```python
# main.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from voirs_ffi import VoirsPipeline
import io

app = FastAPI()

# Initialize pipeline
pipeline = VoirsPipeline.with_config(use_gpu=True)

class SynthesisRequest(BaseModel):
    text: str
    voice: str = None
    format: str = "wav"

class VoiceResponse(BaseModel):
    id: str
    name: str
    language: str
    gender: str

@app.post("/synthesize")
async def synthesize(request: SynthesisRequest):
    try:
        audio = pipeline.synthesize(request.text, voice=request.voice)
        
        # Convert to bytes
        audio_bytes = audio.to_bytes(format=request.format)
        
        return StreamingResponse(
            io.BytesIO(audio_bytes),
            media_type=f"audio/{request.format}"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/voices", response_model=List[VoiceResponse])
async def get_voices():
    voices = pipeline.get_voices()
    return [
        VoiceResponse(
            id=v.id,
            name=v.name,
            language=v.language,
            gender=v.gender
        )
        for v in voices
    ]
```

## Desktop Applications

### Tkinter GUI Application

```python
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from voirs_ffi import VoirsPipeline
import threading

class VoiRSGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("VoiRS Text-to-Speech")
        self.root.geometry("600x400")
        
        # Initialize pipeline
        self.pipeline = VoirsPipeline.with_config(use_gpu=True)
        
        self.create_widgets()
        self.load_voices()
    
    def create_widgets(self):
        # Text input
        text_label = ttk.Label(self.root, text="Text to synthesize:")
        text_label.pack(pady=5)
        
        self.text_entry = tk.Text(self.root, height=8, width=70)
        self.text_entry.pack(pady=5)
        
        # Voice selection
        voice_label = ttk.Label(self.root, text="Voice:")
        voice_label.pack(pady=5)
        
        self.voice_var = tk.StringVar()
        self.voice_combo = ttk.Combobox(self.root, textvariable=self.voice_var)
        self.voice_combo.pack(pady=5)
        
        # Buttons
        button_frame = ttk.Frame(self.root)
        button_frame.pack(pady=10)
        
        self.synthesize_btn = ttk.Button(
            button_frame, 
            text="Synthesize", 
            command=self.synthesize_text
        )
        self.synthesize_btn.pack(side=tk.LEFT, padx=5)
        
        self.play_btn = ttk.Button(
            button_frame, 
            text="Play", 
            command=self.play_audio,
            state=tk.DISABLED
        )
        self.play_btn.pack(side=tk.LEFT, padx=5)
        
        self.save_btn = ttk.Button(
            button_frame, 
            text="Save", 
            command=self.save_audio,
            state=tk.DISABLED
        )
        self.save_btn.pack(side=tk.LEFT, padx=5)
        
        # Progress bar
        self.progress = ttk.Progressbar(
            self.root, 
            mode='indeterminate'
        )
        self.progress.pack(pady=10, fill=tk.X)
        
        # Status label
        self.status_label = ttk.Label(self.root, text="Ready")
        self.status_label.pack(pady=5)
        
        self.audio = None
    
    def load_voices(self):
        voices = self.pipeline.get_voices()
        voice_names = [f"{v.name} ({v.id})" for v in voices]
        self.voice_combo['values'] = voice_names
        if voice_names:
            self.voice_combo.set(voice_names[0])
    
    def synthesize_text(self):
        text = self.text_entry.get("1.0", tk.END).strip()
        if not text:
            messagebox.showerror("Error", "Please enter text to synthesize")
            return
        
        # Extract voice ID
        voice_selection = self.voice_var.get()
        voice_id = voice_selection.split('(')[1].split(')')[0]
        
        # Start synthesis in background thread
        self.synthesize_btn.config(state=tk.DISABLED)
        self.progress.start()
        self.status_label.config(text="Synthesizing...")
        
        thread = threading.Thread(
            target=self._synthesize_worker,
            args=(text, voice_id)
        )
        thread.daemon = True
        thread.start()
    
    def _synthesize_worker(self, text, voice_id):
        try:
            self.audio = self.pipeline.synthesize(text, voice=voice_id)
            self.root.after(0, self._synthesis_complete)
        except Exception as e:
            self.root.after(0, self._synthesis_error, str(e))
    
    def _synthesis_complete(self):
        self.progress.stop()
        self.synthesize_btn.config(state=tk.NORMAL)
        self.play_btn.config(state=tk.NORMAL)
        self.save_btn.config(state=tk.NORMAL)
        self.status_label.config(text="Synthesis complete")
    
    def _synthesis_error(self, error):
        self.progress.stop()
        self.synthesize_btn.config(state=tk.NORMAL)
        self.status_label.config(text="Error")
        messagebox.showerror("Synthesis Error", error)
    
    def play_audio(self):
        if self.audio:
            self.audio.play()
    
    def save_audio(self):
        if self.audio:
            filename = filedialog.asksaveasfilename(
                defaultextension=".wav",
                filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
            )
            if filename:
                self.audio.save(filename)
                messagebox.showinfo("Success", f"Audio saved to {filename}")

if __name__ == "__main__":
    root = tk.Tk()
    app = VoiRSGUI(root)
    root.mainloop()
```

### PyQt5 Integration

```python
import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, 
                             QHBoxLayout, QTextEdit, QComboBox, QPushButton, 
                             QLabel, QWidget, QProgressBar, QFileDialog, QMessageBox)
from PyQt5.QtCore import QThread, pyqtSignal
from voirs_ffi import VoirsPipeline

class SynthesisThread(QThread):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    
    def __init__(self, pipeline, text, voice_id):
        super().__init__()
        self.pipeline = pipeline
        self.text = text
        self.voice_id = voice_id
    
    def run(self):
        try:
            audio = self.pipeline.synthesize(self.text, voice=self.voice_id)
            self.finished.emit(audio)
        except Exception as e:
            self.error.emit(str(e))

class VoiRSMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VoiRS Text-to-Speech")
        self.setGeometry(100, 100, 600, 400)
        
        # Initialize pipeline
        self.pipeline = VoirsPipeline.with_config(use_gpu=True)
        self.audio = None
        
        self.init_ui()
        self.load_voices()
    
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout()
        
        # Text input
        layout.addWidget(QLabel("Text to synthesize:"))
        self.text_edit = QTextEdit()
        layout.addWidget(self.text_edit)
        
        # Voice selection
        layout.addWidget(QLabel("Voice:"))
        self.voice_combo = QComboBox()
        layout.addWidget(self.voice_combo)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.synthesize_btn = QPushButton("Synthesize")
        self.synthesize_btn.clicked.connect(self.synthesize_text)
        button_layout.addWidget(self.synthesize_btn)
        
        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self.play_audio)
        self.play_btn.setEnabled(False)
        button_layout.addWidget(self.play_btn)
        
        self.save_btn = QPushButton("Save")
        self.save_btn.clicked.connect(self.save_audio)
        self.save_btn.setEnabled(False)
        button_layout.addWidget(self.save_btn)
        
        layout.addLayout(button_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Status label
        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)
        
        central_widget.setLayout(layout)
    
    def load_voices(self):
        voices = self.pipeline.get_voices()
        for voice in voices:
            self.voice_combo.addItem(f"{voice.name} ({voice.id})", voice.id)
    
    def synthesize_text(self):
        text = self.text_edit.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "Error", "Please enter text to synthesize")
            return
        
        voice_id = self.voice_combo.currentData()
        
        # Start synthesis thread
        self.synthesize_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate
        self.status_label.setText("Synthesizing...")
        
        self.synthesis_thread = SynthesisThread(self.pipeline, text, voice_id)
        self.synthesis_thread.finished.connect(self.on_synthesis_finished)
        self.synthesis_thread.error.connect(self.on_synthesis_error)
        self.synthesis_thread.start()
    
    def on_synthesis_finished(self, audio):
        self.audio = audio
        self.synthesize_btn.setEnabled(True)
        self.play_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText("Synthesis complete")
    
    def on_synthesis_error(self, error):
        self.synthesize_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText("Error")
        QMessageBox.critical(self, "Synthesis Error", error)
    
    def play_audio(self):
        if self.audio:
            self.audio.play()
    
    def save_audio(self):
        if self.audio:
            filename, _ = QFileDialog.getSaveFileName(
                self, "Save Audio", "", "WAV files (*.wav);;All files (*.*)"
            )
            if filename:
                self.audio.save(filename)
                QMessageBox.information(self, "Success", f"Audio saved to {filename}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VoiRSMainWindow()
    window.show()
    sys.exit(app.exec_())
```

## API Services

### REST API with Authentication

```python
from flask import Flask, request, jsonify, send_file
from flask_jwt_extended import JWTManager, create_access_token, jwt_required
from voirs_ffi import VoirsPipeline
import tempfile
import os

app = Flask(__name__)
app.config['JWT_SECRET_KEY'] = 'your-secret-key'
jwt = JWTManager(app)

# Initialize pipeline
pipeline = VoirsPipeline.with_config(use_gpu=True)

@app.route('/auth/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    
    # Validate credentials (implement your own logic)
    if username == 'admin' and password == 'password':
        access_token = create_access_token(identity=username)
        return jsonify({'access_token': access_token})
    
    return jsonify({'error': 'Invalid credentials'}), 401

@app.route('/api/synthesize', methods=['POST'])
@jwt_required()
def synthesize():
    data = request.get_json()
    text = data.get('text', '')
    voice = data.get('voice', None)
    format = data.get('format', 'wav')
    
    if not text:
        return jsonify({'error': 'Text is required'}), 400
    
    try:
        audio = pipeline.synthesize(text, voice=voice)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix=f'.{format}', delete=False) as tmp:
            audio.save(tmp.name, format=format)
            
            return send_file(
                tmp.name,
                mimetype=f'audio/{format}',
                as_attachment=True,
                download_name=f'speech.{format}'
            )
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/voices', methods=['GET'])
@jwt_required()
def get_voices():
    voices = pipeline.get_voices()
    return jsonify([
        {
            'id': v.id,
            'name': v.name,
            'language': v.language,
            'gender': v.gender,
            'age': v.age
        }
        for v in voices
    ])

if __name__ == '__main__':
    app.run(debug=True)
```

### GraphQL API

```python
import graphene
from graphene import ObjectType, String, List, Field, Mutation, Schema
from voirs_ffi import VoirsPipeline
import base64

# Initialize pipeline
pipeline = VoirsPipeline.with_config(use_gpu=True)

class Voice(ObjectType):
    id = String()
    name = String()
    language = String()
    gender = String()
    age = String()

class AudioResult(ObjectType):
    audio_data = String()  # Base64 encoded audio
    format = String()
    duration = String()

class SynthesizeText(Mutation):
    class Arguments:
        text = String(required=True)
        voice = String()
        format = String()
    
    Output = AudioResult
    
    def mutate(self, info, text, voice=None, format="wav"):
        try:
            audio = pipeline.synthesize(text, voice=voice)
            
            # Convert audio to base64
            audio_bytes = audio.to_bytes(format=format)
            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
            
            return AudioResult(
                audio_data=audio_base64,
                format=format,
                duration=str(audio.duration)
            )
            
        except Exception as e:
            raise Exception(f"Synthesis failed: {str(e)}")

class Query(ObjectType):
    voices = List(Voice)
    
    def resolve_voices(self, info):
        voices = pipeline.get_voices()
        return [
            Voice(
                id=v.id,
                name=v.name,
                language=v.language,
                gender=v.gender,
                age=v.age
            )
            for v in voices
        ]

class Mutation(ObjectType):
    synthesize_text = SynthesizeText.Field()

schema = Schema(query=Query, mutation=Mutation)

# Usage with Flask-GraphQL
from flask import Flask
from flask_graphql import GraphQLView

app = Flask(__name__)
app.add_url_rule('/graphql', view_func=GraphQLView.as_view('graphql', schema=schema, graphiql=True))

if __name__ == '__main__':
    app.run(debug=True)
```

## Voice Assistants

### Simple Voice Assistant

```python
import speech_recognition as sr
from voirs_ffi import VoirsPipeline
import os

class VoiceAssistant:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.pipeline = VoirsPipeline.with_config(use_gpu=True)
        
        # Calibrate microphone
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
    
    def listen(self):
        """Listen for voice input"""
        with self.microphone as source:
            print("Listening...")
            audio = self.recognizer.listen(source, timeout=5)
        
        try:
            text = self.recognizer.recognize_google(audio)
            print(f"You said: {text}")
            return text
        except sr.UnknownValueError:
            print("Could not understand audio")
            return None
        except sr.RequestError as e:
            print(f"Error with speech recognition: {e}")
            return None
    
    def speak(self, text):
        """Convert text to speech and play"""
        try:
            audio = self.pipeline.synthesize(text)
            audio.play()
        except Exception as e:
            print(f"Error in speech synthesis: {e}")
    
    def process_command(self, command):
        """Process voice command"""
        if not command:
            return
        
        command = command.lower()
        
        if "hello" in command:
            self.speak("Hello! How can I help you?")
        elif "time" in command:
            import datetime
            now = datetime.datetime.now()
            time_str = now.strftime("%I:%M %p")
            self.speak(f"The current time is {time_str}")
        elif "date" in command:
            import datetime
            today = datetime.date.today()
            date_str = today.strftime("%B %d, %Y")
            self.speak(f"Today is {date_str}")
        elif "goodbye" in command or "exit" in command:
            self.speak("Goodbye!")
            return False
        else:
            self.speak("I'm sorry, I didn't understand that command.")
        
        return True
    
    def run(self):
        """Main assistant loop"""
        self.speak("Voice assistant started. Say 'hello' to begin.")
        
        while True:
            command = self.listen()
            if not self.process_command(command):
                break

if __name__ == "__main__":
    assistant = VoiceAssistant()
    assistant.run()
```

### Conversational AI Integration

```python
import openai
from voirs_ffi import VoirsPipeline
import speech_recognition as sr

class ConversationalAI:
    def __init__(self, openai_key):
        openai.api_key = openai_key
        self.pipeline = VoirsPipeline.with_config(use_gpu=True)
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # Conversation history
        self.messages = [
            {"role": "system", "content": "You are a helpful AI assistant."}
        ]
    
    def listen(self):
        """Listen for voice input"""
        with self.microphone as source:
            print("Listening...")
            audio = self.recognizer.listen(source)
        
        try:
            text = self.recognizer.recognize_google(audio)
            print(f"You said: {text}")
            return text
        except sr.UnknownValueError:
            return None
        except sr.RequestError as e:
            print(f"Error: {e}")
            return None
    
    def get_ai_response(self, user_input):
        """Get response from OpenAI"""
        self.messages.append({"role": "user", "content": user_input})
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=self.messages,
                max_tokens=150
            )
            
            ai_response = response.choices[0].message.content
            self.messages.append({"role": "assistant", "content": ai_response})
            
            return ai_response
            
        except Exception as e:
            return f"Sorry, I encountered an error: {str(e)}"
    
    def speak(self, text):
        """Convert text to speech"""
        try:
            audio = self.pipeline.synthesize(text)
            audio.play()
        except Exception as e:
            print(f"Speech synthesis error: {e}")
    
    def conversation_loop(self):
        """Main conversation loop"""
        self.speak("Hello! I'm your AI assistant. How can I help you today?")
        
        while True:
            user_input = self.listen()
            
            if user_input:
                if "goodbye" in user_input.lower() or "exit" in user_input.lower():
                    self.speak("Goodbye! Have a great day!")
                    break
                
                # Get AI response
                ai_response = self.get_ai_response(user_input)
                print(f"AI: {ai_response}")
                
                # Speak the response
                self.speak(ai_response)

if __name__ == "__main__":
    # Initialize with your OpenAI API key
    ai = ConversationalAI("your-openai-api-key")
    ai.conversation_loop()
```

## Content Management Systems

### WordPress Plugin (Python Backend)

```python
# wordpress_tts.py
from flask import Flask, request, jsonify
from voirs_ffi import VoirsPipeline
import os
import hashlib

app = Flask(__name__)

# Initialize pipeline
pipeline = VoirsPipeline.with_config(use_gpu=True)

# Audio cache directory
AUDIO_CACHE_DIR = "/var/www/html/wp-content/uploads/tts-audio/"
os.makedirs(AUDIO_CACHE_DIR, exist_ok=True)

@app.route('/generate-audio', methods=['POST'])
def generate_audio():
    data = request.get_json()
    text = data.get('text', '')
    voice = data.get('voice', 'default')
    post_id = data.get('post_id', '')
    
    if not text:
        return jsonify({'error': 'Text is required'}), 400
    
    # Create cache key
    cache_key = hashlib.md5(f"{text}_{voice}".encode()).hexdigest()
    audio_filename = f"{post_id}_{cache_key}.wav"
    audio_path = os.path.join(AUDIO_CACHE_DIR, audio_filename)
    
    # Check if audio already exists
    if os.path.exists(audio_path):
        return jsonify({
            'success': True,
            'audio_url': f"/wp-content/uploads/tts-audio/{audio_filename}"
        })
    
    try:
        # Generate audio
        audio = pipeline.synthesize(text, voice=voice)
        audio.save(audio_path)
        
        return jsonify({
            'success': True,
            'audio_url': f"/wp-content/uploads/tts-audio/{audio_filename}"
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/voices', methods=['GET'])
def get_voices():
    voices = pipeline.get_voices()
    return jsonify([
        {
            'id': v.id,
            'name': v.name,
            'language': v.language
        }
        for v in voices
    ])

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### Drupal Integration

```python
# drupal_tts_service.py
from flask import Flask, request, jsonify
from voirs_ffi import VoirsPipeline
import mysql.connector
import os

app = Flask(__name__)

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'user': 'drupal_user',
    'password': 'drupal_password',
    'database': 'drupal_db'
}

# Initialize pipeline
pipeline = VoirsPipeline.with_config(use_gpu=True)

def get_db_connection():
    return mysql.connector.connect(**DB_CONFIG)

@app.route('/process-node', methods=['POST'])
def process_node():
    data = request.get_json()
    node_id = data.get('node_id')
    content = data.get('content')
    
    if not node_id or not content:
        return jsonify({'error': 'Node ID and content are required'}), 400
    
    try:
        # Generate audio
        audio = pipeline.synthesize(content)
        
        # Save audio file
        audio_filename = f"node_{node_id}_audio.wav"
        audio_path = f"/var/www/html/sites/default/files/audio/{audio_filename}"
        audio.save(audio_path)
        
        # Update database
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO node_audio (node_id, audio_filename, created)
            VALUES (%s, %s, NOW())
            ON DUPLICATE KEY UPDATE
            audio_filename = VALUES(audio_filename),
            updated = NOW()
        """, (node_id, audio_filename))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return jsonify({
            'success': True,
            'audio_url': f"/sites/default/files/audio/{audio_filename}"
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
```

## Machine Learning Pipelines

### Data Pipeline Integration

```python
# ml_pipeline.py
from voirs_ffi import VoirsPipeline
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

class TextToSpeechDataPipeline:
    def __init__(self):
        self.pipeline = VoirsPipeline.with_config(use_gpu=True)
        self.scaler = StandardScaler()
    
    def generate_audio_features(self, text_data):
        """Generate audio features from text data"""
        audio_features = []
        
        for text in text_data:
            try:
                # Synthesize text
                audio = self.pipeline.synthesize(text)
                
                # Extract features
                features = {
                    'duration': audio.duration,
                    'sample_rate': audio.sample_rate,
                    'rms': np.sqrt(np.mean(np.square(audio.samples))),
                    'max_amplitude': np.max(np.abs(audio.samples)),
                    'zero_crossing_rate': np.mean(np.diff(np.signbit(audio.samples))),
                    'spectral_centroid': self._calculate_spectral_centroid(audio.samples),
                    'text_length': len(text),
                    'word_count': len(text.split())
                }
                
                audio_features.append(features)
                
            except Exception as e:
                print(f"Error processing text: {e}")
                audio_features.append(None)
        
        return pd.DataFrame(audio_features)
    
    def _calculate_spectral_centroid(self, audio_samples):
        """Calculate spectral centroid"""
        fft = np.fft.fft(audio_samples)
        magnitude = np.abs(fft)
        frequencies = np.fft.fftfreq(len(audio_samples))
        
        return np.sum(frequencies * magnitude) / np.sum(magnitude)
    
    def train_quality_predictor(self, texts, quality_scores):
        """Train a model to predict audio quality"""
        # Generate features
        features_df = self.generate_audio_features(texts)
        
        # Remove rows with missing data
        mask = features_df.notna().all(axis=1)
        features_df = features_df[mask]
        quality_scores = np.array(quality_scores)[mask]
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features_df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features_scaled, quality_scores, test_size=0.2, random_state=42
        )
        
        # Train model
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        print(f"Training score: {train_score:.3f}")
        print(f"Test score: {test_score:.3f}")
        
        # Save model
        joblib.dump(model, 'audio_quality_model.pkl')
        joblib.dump(self.scaler, 'audio_features_scaler.pkl')
        
        return model
    
    def predict_quality(self, text):
        """Predict audio quality for given text"""
        # Load model
        model = joblib.load('audio_quality_model.pkl')
        scaler = joblib.load('audio_features_scaler.pkl')
        
        # Generate features
        features_df = self.generate_audio_features([text])
        features_scaled = scaler.transform(features_df)
        
        # Predict
        quality_score = model.predict(features_scaled)[0]
        
        return quality_score

# Usage example
if __name__ == "__main__":
    pipeline = TextToSpeechDataPipeline()
    
    # Sample data
    texts = [
        "Hello, world!",
        "This is a test sentence.",
        "Machine learning is fascinating.",
        "Audio processing with Python is powerful."
    ]
    
    quality_scores = [4.2, 3.8, 4.5, 4.1]  # Example quality scores
    
    # Train model
    model = pipeline.train_quality_predictor(texts, quality_scores)
    
    # Predict quality for new text
    new_text = "This is a new sentence for quality prediction."
    predicted_quality = pipeline.predict_quality(new_text)
    print(f"Predicted quality: {predicted_quality:.2f}")
```

## Real-time Applications

### WebSocket Server

```python
# websocket_server.py
import asyncio
import websockets
import json
from voirs_ffi import VoirsPipeline
import base64
import threading

class VoiRSWebSocketServer:
    def __init__(self):
        self.pipeline = VoirsPipeline.with_config(use_gpu=True)
        self.clients = set()
    
    async def register_client(self, websocket):
        """Register a new client"""
        self.clients.add(websocket)
        print(f"Client connected: {websocket.remote_address}")
    
    async def unregister_client(self, websocket):
        """Unregister a client"""
        self.clients.discard(websocket)
        print(f"Client disconnected: {websocket.remote_address}")
    
    async def handle_message(self, websocket, message):
        """Handle incoming message from client"""
        try:
            data = json.loads(message)
            message_type = data.get('type')
            
            if message_type == 'synthesize':
                await self.handle_synthesis(websocket, data)
            elif message_type == 'get_voices':
                await self.handle_get_voices(websocket)
            else:
                await websocket.send(json.dumps({
                    'type': 'error',
                    'message': f'Unknown message type: {message_type}'
                }))
        
        except json.JSONDecodeError:
            await websocket.send(json.dumps({
                'type': 'error',
                'message': 'Invalid JSON format'
            }))
        except Exception as e:
            await websocket.send(json.dumps({
                'type': 'error',
                'message': str(e)
            }))
    
    async def handle_synthesis(self, websocket, data):
        """Handle synthesis request"""
        text = data.get('text', '')
        voice = data.get('voice', None)
        request_id = data.get('request_id', None)
        
        if not text:
            await websocket.send(json.dumps({
                'type': 'error',
                'message': 'Text is required',
                'request_id': request_id
            }))
            return
        
        # Send progress update
        await websocket.send(json.dumps({
            'type': 'progress',
            'message': 'Starting synthesis...',
            'request_id': request_id
        }))
        
        try:
            # Synthesize in thread to avoid blocking
            def synthesize():
                return self.pipeline.synthesize(text, voice=voice)
            
            loop = asyncio.get_event_loop()
            audio = await loop.run_in_executor(None, synthesize)
            
            # Convert audio to base64
            audio_bytes = audio.to_bytes(format='wav')
            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
            
            # Send result
            await websocket.send(json.dumps({
                'type': 'synthesis_complete',
                'audio_data': audio_base64,
                'format': 'wav',
                'duration': audio.duration,
                'request_id': request_id
            }))
            
        except Exception as e:
            await websocket.send(json.dumps({
                'type': 'error',
                'message': str(e),
                'request_id': request_id
            }))
    
    async def handle_get_voices(self, websocket):
        """Handle get voices request"""
        try:
            voices = self.pipeline.get_voices()
            voice_data = [
                {
                    'id': v.id,
                    'name': v.name,
                    'language': v.language,
                    'gender': v.gender
                }
                for v in voices
            ]
            
            await websocket.send(json.dumps({
                'type': 'voices',
                'voices': voice_data
            }))
            
        except Exception as e:
            await websocket.send(json.dumps({
                'type': 'error',
                'message': str(e)
            }))
    
    async def client_handler(self, websocket, path):
        """Handle client connection"""
        await self.register_client(websocket)
        
        try:
            async for message in websocket:
                await self.handle_message(websocket, message)
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            await self.unregister_client(websocket)

if __name__ == "__main__":
    server = VoiRSWebSocketServer()
    
    print("Starting VoiRS WebSocket server on ws://localhost:8765")
    start_server = websockets.serve(server.client_handler, "localhost", 8765)
    
    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()
```

### Streaming Audio Server

```python
# streaming_server.py
from flask import Flask, request, Response
from voirs_ffi import VoirsPipeline
import queue
import threading
import time

app = Flask(__name__)

# Initialize pipeline
pipeline = VoirsPipeline.with_config(use_gpu=True, streaming=True)

class StreamingSession:
    def __init__(self, session_id):
        self.session_id = session_id
        self.audio_queue = queue.Queue()
        self.is_active = True
        self.last_activity = time.time()
    
    def add_audio_chunk(self, audio_chunk):
        if self.is_active:
            self.audio_queue.put(audio_chunk)
            self.last_activity = time.time()
    
    def get_audio_chunk(self, timeout=1.0):
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def close(self):
        self.is_active = False

# Session management
sessions = {}
session_lock = threading.Lock()

def cleanup_sessions():
    """Clean up inactive sessions"""
    while True:
        current_time = time.time()
        with session_lock:
            inactive_sessions = [
                session_id for session_id, session in sessions.items()
                if current_time - session.last_activity > 300  # 5 minutes
            ]
            
            for session_id in inactive_sessions:
                sessions[session_id].close()
                del sessions[session_id]
        
        time.sleep(60)  # Check every minute

# Start cleanup thread
cleanup_thread = threading.Thread(target=cleanup_sessions, daemon=True)
cleanup_thread.start()

@app.route('/start_stream', methods=['POST'])
def start_stream():
    """Start a new streaming session"""
    data = request.get_json()
    session_id = data.get('session_id')
    
    if not session_id:
        return {'error': 'Session ID is required'}, 400
    
    with session_lock:
        if session_id in sessions:
            sessions[session_id].close()
        
        sessions[session_id] = StreamingSession(session_id)
    
    return {'success': True, 'session_id': session_id}

@app.route('/synthesize_stream', methods=['POST'])
def synthesize_stream():
    """Add text to streaming synthesis"""
    data = request.get_json()
    session_id = data.get('session_id')
    text = data.get('text', '')
    
    if not session_id or session_id not in sessions:
        return {'error': 'Invalid session ID'}, 400
    
    if not text:
        return {'error': 'Text is required'}, 400
    
    session = sessions[session_id]
    
    # Synthesize in background thread
    def synthesize_background():
        try:
            audio_stream = pipeline.synthesize_streaming(text)
            for chunk in audio_stream:
                session.add_audio_chunk(chunk.to_bytes())
        except Exception as e:
            session.add_audio_chunk(f"ERROR: {str(e)}".encode())
    
    thread = threading.Thread(target=synthesize_background, daemon=True)
    thread.start()
    
    return {'success': True}

@app.route('/stream/<session_id>')
def stream_audio(session_id):
    """Stream audio chunks to client"""
    if session_id not in sessions:
        return {'error': 'Session not found'}, 404
    
    session = sessions[session_id]
    
    def generate_audio():
        while session.is_active:
            chunk = session.get_audio_chunk()
            if chunk:
                yield chunk
            else:
                yield b''  # Keep connection alive
    
    return Response(
        generate_audio(),
        mimetype='audio/wav',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
        }
    )

@app.route('/stop_stream', methods=['POST'])
def stop_stream():
    """Stop streaming session"""
    data = request.get_json()
    session_id = data.get('session_id')
    
    if session_id and session_id in sessions:
        with session_lock:
            sessions[session_id].close()
            del sessions[session_id]
    
    return {'success': True}

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
```

## Batch Processing

### Large-Scale Batch Processing

```python
# batch_processor.py
from voirs_ffi import VoirsPipeline
import pandas as pd
import multiprocessing as mp
import os
import time
from pathlib import Path

class BatchTTSProcessor:
    def __init__(self, use_gpu=True, num_processes=None):
        self.use_gpu = use_gpu
        self.num_processes = num_processes or mp.cpu_count()
        
        # Create output directory
        self.output_dir = Path("batch_output")
        self.output_dir.mkdir(exist_ok=True)
    
    def process_batch_from_csv(self, csv_file, text_column, output_format='wav'):
        """Process batch from CSV file"""
        # Read CSV
        df = pd.read_csv(csv_file)
        
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in CSV")
        
        # Prepare tasks
        tasks = []
        for index, row in df.iterrows():
            task = {
                'id': index,
                'text': row[text_column],
                'voice': row.get('voice', None),
                'output_file': self.output_dir / f"audio_{index}.{output_format}"
            }
            tasks.append(task)
        
        # Process in parallel
        self.process_tasks_parallel(tasks)
    
    def process_tasks_parallel(self, tasks):
        """Process tasks in parallel"""
        print(f"Processing {len(tasks)} tasks with {self.num_processes} processes...")
        
        start_time = time.time()
        
        # Split tasks among processes
        chunk_size = len(tasks) // self.num_processes
        task_chunks = [
            tasks[i:i + chunk_size] 
            for i in range(0, len(tasks), chunk_size)
        ]
        
        # Process chunks
        with mp.Pool(processes=self.num_processes) as pool:
            results = pool.map(self.process_task_chunk, task_chunks)
        
        # Combine results
        total_processed = sum(results)
        elapsed_time = time.time() - start_time
        
        print(f"Processed {total_processed} tasks in {elapsed_time:.2f} seconds")
        print(f"Average time per task: {elapsed_time / total_processed:.3f} seconds")
    
    def process_task_chunk(self, task_chunk):
        """Process a chunk of tasks in a single process"""
        # Initialize pipeline in each process
        pipeline = VoirsPipeline.with_config(use_gpu=self.use_gpu)
        
        processed_count = 0
        
        for task in task_chunk:
            try:
                # Synthesize text
                audio = pipeline.synthesize(
                    task['text'], 
                    voice=task.get('voice')
                )
                
                # Save audio
                audio.save(str(task['output_file']))
                
                processed_count += 1
                
                # Progress update
                if processed_count % 10 == 0:
                    print(f"Process {os.getpid()}: Processed {processed_count} tasks")
                
            except Exception as e:
                print(f"Error processing task {task['id']}: {e}")
        
        return processed_count
    
    def create_batch_report(self, output_file="batch_report.html"):
        """Create HTML report of batch processing results"""
        audio_files = list(self.output_dir.glob("*.wav"))
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Batch TTS Processing Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .summary {{ background: #f0f0f0; padding: 15px; border-radius: 5px; }}
                .file-list {{ margin-top: 20px; }}
                .file-item {{ padding: 10px; border-bottom: 1px solid #ddd; }}
                .audio-player {{ margin: 10px 0; }}
            </style>
        </head>
        <body>
            <h1>Batch TTS Processing Report</h1>
            
            <div class="summary">
                <h2>Summary</h2>
                <p><strong>Total files processed:</strong> {len(audio_files)}</p>
                <p><strong>Output directory:</strong> {self.output_dir}</p>
                <p><strong>Generated on:</strong> {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="file-list">
                <h2>Generated Audio Files</h2>
        """
        
        for audio_file in sorted(audio_files):
            file_size = audio_file.stat().st_size
            file_size_mb = file_size / (1024 * 1024)
            
            html_content += f"""
                <div class="file-item">
                    <h3>{audio_file.name}</h3>
                    <p><strong>Size:</strong> {file_size_mb:.2f} MB</p>
                    <div class="audio-player">
                        <audio controls>
                            <source src="{audio_file.name}" type="audio/wav">
                            Your browser does not support the audio element.
                        </audio>
                    </div>
                </div>
            """
        
        html_content += """
            </div>
        </body>
        </html>
        """
        
        # Save report
        report_path = self.output_dir / output_file
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        print(f"Batch report saved to: {report_path}")

# Usage example
if __name__ == "__main__":
    # Create sample CSV
    sample_data = pd.DataFrame({
        'text': [
            "Hello, world! This is the first text.",
            "This is the second text for synthesis.",
            "Machine learning is fascinating and powerful.",
            "Python makes everything easier and more fun.",
            "Voice synthesis technology is advancing rapidly."
        ],
        'voice': ['female-1', 'male-1', 'female-2', 'male-2', 'female-1']
    })
    
    sample_data.to_csv('sample_batch.csv', index=False)
    
    # Process batch
    processor = BatchTTSProcessor(use_gpu=True, num_processes=4)
    processor.process_batch_from_csv('sample_batch.csv', 'text')
    
    # Create report
    processor.create_batch_report()
```

This comprehensive integration guide covers various real-world scenarios where VoiRS Python bindings can be effectively used. Each example demonstrates best practices for performance, error handling, and maintainability.
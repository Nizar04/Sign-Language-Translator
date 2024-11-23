import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from fastapi import FastAPI, WebSocket, BackgroundTasks
from fastapi.staticfiles import StaticFiles
import asyncio
import json
import sqlite3
from datetime import datetime
import threading
import queue
from typing import Dict, List, Optional
import pytorch_lightning as pl
from transformers import T5Tokenizer, T5ForConditionalGeneration
import tkinter as tk
from tkinter import ttk
import uvicorn
import torch
from dataclasses import dataclass
from pathlib import Path

@dataclass
class SignConfig:
    language: str
    model_path: str
    vocab_path: str
    grammar_rules: dict

class HandTracker:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.finger_states = []
        
    def get_landmarks(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        landmarks = []
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                hand_data = []
                for lm in hand_landmarks.landmark:
                    hand_data.extend([lm.x, lm.y, lm.z])
                landmarks.append(hand_data)
                
        return landmarks, results.multi_hand_landmarks

class FingerSpellingModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.lstm = torch.nn.LSTM(63, 256, 2, batch_first=True)
        self.fc1 = torch.nn.Linear(256, 128)
        self.fc2 = torch.nn.Linear(128, 26)
        
    def forward(self, x):
        x, _ = self.lstm(x)
        x = torch.relu(self.fc1(x[:, -1]))
        return torch.softmax(self.fc2(x), dim=1)

class SignLanguageTranslator:
    def __init__(self):
        self.languages = {
            'asl': SignConfig(
                language='American Sign Language',
                model_path='models/asl_model.pt',
                vocab_path='data/asl_vocabulary.json',
                grammar_rules={'word_order': 'SVO'}
            ),
            'bsl': SignConfig(
                language='British Sign Language',
                model_path='models/bsl_model.pt',
                vocab_path='data/bsl_vocabulary.json',
                grammar_rules={'word_order': 'SOV'}
            )
        }
        
        self.current_language = 'asl'
        self.hand_tracker = HandTracker()
        self.finger_spelling_model = FingerSpellingModel()
        self.sentence_model = T5ForConditionalGeneration.from_pretrained('t5-small')
        self.tokenizer = T5Tokenizer.from_pretrained('t5-small')
        
        self.gesture_buffer = []
        self.word_buffer = []
        self.db = Database()
        self.gui = None
        self.active_websockets = set()
        
    async def process_frame(self, frame):
        landmarks, hand_landmarks = self.hand_tracker.get_landmarks(frame)
        if not landmarks:
            return frame, None
            
        gesture_prediction = self.predict_gesture(landmarks)
        finger_spelled = self.predict_finger_spelling(landmarks)
        
        if gesture_prediction:
            self.word_buffer.append(gesture_prediction)
        elif finger_spelled:
            self.word_buffer.append(finger_spelled)
            
        if len(self.word_buffer) >= 3:
            sentence = self.construct_sentence(self.word_buffer)
            self.word_buffer = []
            await self.broadcast_translation(sentence)
            
        return self.draw_debug_info(frame, hand_landmarks), gesture_prediction
        
    def predict_gesture(self, landmarks):
        config = self.languages[self.current_language]
        model = torch.load(config.model_path)
        
        with torch.no_grad():
            features = torch.tensor(landmarks, dtype=torch.float32)
            prediction = model(features)
            
        if torch.max(prediction) > 0.85:
            with open(config.vocab_path) as f:
                vocab = json.load(f)
            return vocab[torch.argmax(prediction).item()]
        return None
        
    def predict_finger_spelling(self, landmarks):
        with torch.no_grad():
            features = torch.tensor(landmarks, dtype=torch.float32)
            prediction = self.finger_spelling_model(features)
            
        if torch.max(prediction) > 0.9:
            return chr(ord('A') + torch.argmax(prediction).item())
        return None
        
    def construct_sentence(self, words):
        input_text = ' '.join(words)
        inputs = self.tokenizer(f"normalize: {input_text}", return_tensors="pt")
        
        outputs = self.sentence_model.generate(
            inputs.input_ids,
            max_length=50,
            num_beams=5,
            temperature=0.7
        )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
    def draw_debug_info(self, frame, hand_landmarks):
        if hand_landmarks:
            for landmarks in hand_landmarks:
                self.hand_tracker.mp_draw.draw_landmarks(
                    frame, landmarks, self.hand_tracker.mp_hands.HAND_CONNECTIONS
                )
        return frame
        
    async def broadcast_translation(self, text):
        for ws in self.active_websockets:
            try:
                await ws.send_text(json.dumps({
                    'type': 'translation',
                    'text': text,
                    'language': self.current_language
                }))
            except:
                self.active_websockets.remove(ws)

class Database:
    def __init__(self):
        self.conn = sqlite3.connect('sign_language.db')
        self.create_tables()
        
    def create_tables(self):
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS translations (
                id INTEGER PRIMARY KEY,
                timestamp TEXT,
                source_language TEXT,
                raw_gestures BLOB,
                translated_text TEXT,
                confidence REAL,
                verified BOOLEAN
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS custom_signs (
                id INTEGER PRIMARY KEY,
                language TEXT,
                sign_data BLOB,
                label TEXT,
                user_id TEXT
            )
        ''')
        self.conn.commit()

class GUI(tk.Tk):
    def __init__(self, translator):
        super().__init__()
        
        self.translator = translator
        self.title("Sign Language Translation System")
        
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(expand=True, fill='both')
        
        self.create_translation_tab()
        self.create_settings_tab()
        self.create_training_tab()
        
    def create_translation_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Translation")
        
        self.translation_text = tk.Text(tab, height=10)
        self.translation_text.pack(pady=10)
        
        language_frame = ttk.LabelFrame(tab, text="Language Selection")
        language_frame.pack(pady=10)
        
        for lang in self.translator.languages:
            ttk.Radiobutton(
                language_frame,
                text=self.translator.languages[lang].language,
                value=lang,
                variable=self.translator.current_language
            ).pack()

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    translator.active_websockets.add(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            if data == "ping": 
                await websocket.send_text("pong")
    except:
        translator.active_websockets.remove(websocket)

def start_server():
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    translator = SignLanguageTranslator()
    
    server_thread = threading.Thread(target=start_server)
    server_thread.daemon = True
    server_thread.start()
    
    gui = GUI(translator)
    gui.mainloop()

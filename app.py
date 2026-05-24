from flask import Flask, render_template, Response, jsonify, request, url_for
import cv2
import numpy as np
import mediapipe as mp
import webbrowser
from threading import Timer
from tensorflow.keras.models import load_model
import os
import sqlite3
import re 

app = Flask(__name__)

# --- DATABASE SETUP ---
def init_db():
    with sqlite3.connect('sign_language.db') as conn:
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS history 
                     (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                      content TEXT, 
                      timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
        conn.commit()

init_db()

# --- LOAD MODELS ---
try:
    model = load_model('THE_FINAL_MODEL.h5')
    print("--- DEFINITIVE MODEL LOADED SUCCESSFULLY ---")
except Exception as e:
    print(f"---!!! CRITICAL ERROR: CANNOT LOAD MODEL !!!---")
    model = None

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
class_names = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','0','1','2','3','4','5','6','7','8','9','NONE']
last_prediction = ""

# --- VIDEO GENERATOR ---
def generate_frames():
    global last_prediction
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    frame_counter = 0
    PREDICTION_INTERVAL = 3
    
    while True:
        success, frame = cap.read()
        if not success: break
        
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        if frame_counter % PREDICTION_INTERVAL == 0:
            if results.multi_hand_landmarks:
                all_landmarks_data = []
                handedness_list = [res.classification[0].label for res in results.multi_handedness]
                
                if 'Left' in handedness_list:
                    hand_landmarks = results.multi_hand_landmarks[handedness_list.index('Left')]
                    wrist = np.array([hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y, hand_landmarks.landmark[0].z])
                    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
                    all_landmarks_data.extend((landmarks - wrist).flatten())
                else: all_landmarks_data.extend([0.0] * 63)

                if 'Right' in handedness_list:
                    hand_landmarks = results.multi_hand_landmarks[handedness_list.index('Right')]
                    wrist = np.array([hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y, hand_landmarks.landmark[0].z])
                    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
                    all_landmarks_data.extend((landmarks - wrist).flatten())
                else: all_landmarks_data.extend([0.0] * 63)
                
                if model is not None:
                    prediction = model.predict(np.array([all_landmarks_data]), verbose=0)
                    predicted_class_index = np.argmax(prediction)
                    if predicted_class_index < len(class_names): 
                        last_prediction = class_names[predicted_class_index]
            else: 
                last_prediction = "NONE"

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        cv2.putText(frame, f'Prediction: {last_prediction}', (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        frame_counter += 1
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    cap.release()

# --- ROUTES ---

# UPDATED INDEX: Sends Stats to Dashboard
@app.route('/')
def index(): 
    count = 0
    try:
        with sqlite3.connect('sign_language.db') as conn:
            c = conn.cursor()
            c.execute("SELECT COUNT(*) FROM history")
            count = c.fetchone()[0]
    except:
        count = 0
    return render_template('index.html', total_translations=count)

@app.route('/sign_recognition')
def sign_recognition(): return render_template('sign_recognition.html')

@app.route('/history')
def history():
    with sqlite3.connect('sign_language.db') as conn:
        c = conn.cursor()
        c.execute("SELECT id, content, timestamp FROM history ORDER BY id DESC")
        data = c.fetchall()
    return render_template('history.html', history_data=data)

@app.route('/converter')
def converter(): return render_template('converter.html')

@app.route('/dictionary')
def dictionary():
    assets_folder = os.path.join('static', 'assets')
    alphabets = []
    words = []
    if os.path.exists(assets_folder):
        for filename in os.listdir(assets_folder):
            if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                alphabets.append(filename)
            elif filename.lower().endswith('.mp4'):
                words.append(filename)
    alphabets.sort()
    words.sort()
    return render_template('dictionary.html', alphabets=alphabets, words=words)

@app.route('/video_feed')
def video_feed(): return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_prediction')
def get_prediction(): return jsonify({'prediction': last_prediction if last_prediction != "NONE" else ""})

@app.route('/save_sentence', methods=['POST'])
def save_sentence():
    data = request.json
    sentence = data.get('sentence')
    if sentence:
        with sqlite3.connect('sign_language.db') as conn:
            c = conn.cursor()
            c.execute("INSERT INTO history (content) VALUES (?)", (sentence,))
            conn.commit()
        return jsonify({"message": "Saved successfully!"})
    return jsonify({"message": "Empty sentence"}), 400

@app.route('/clear_history', methods=['POST'])
def clear_history():
    with sqlite3.connect('sign_language.db') as conn:
        c = conn.cursor()
        c.execute("DELETE FROM history")
        conn.commit()
    return jsonify({"message": "History cleared!"})

# NEW: Delete specific item
@app.route('/delete_item/<int:item_id>', methods=['POST'])
def delete_item(item_id):
    with sqlite3.connect('sign_language.db') as conn:
        c = conn.cursor()
        c.execute("DELETE FROM history WHERE id=?", (item_id,))
        conn.commit()
    return jsonify({"message": "Item deleted!"})

# --- HELPER & ANIMATION ---
def find_file(filename, search_path):
    if not os.path.exists(search_path): return None
    files = os.listdir(search_path)
    for f in files:
        if f.lower() == filename.lower():
            return f
    return None

@app.route('/animation', methods=['POST'])
def animation():
    data = request.json
    text = data.get('text', '').lower()
    text = re.sub(r'[^a-z0-9\s]', '', text) 
    words = text.split()
    media_paths = [] 
    word_names = []
    assets_folder = os.path.join('static', 'assets')
    for word in words:
        found_video = find_file(f"{word}.mp4", assets_folder)
        if found_video:
            media_paths.append(url_for('static', filename=f'assets/{found_video}'))
            word_names.append(word.upper())
        else:
            for letter in word:
                found_img = find_file(f"{letter}.jpg", assets_folder) or find_file(f"{letter}.png", assets_folder) or find_file(f"{letter}.jpeg", assets_folder)
                if found_img:
                    media_paths.append(url_for('static', filename=f'assets/{found_img}'))
                    word_names.append(letter.upper())
                else:
                    found_vid_letter = find_file(f"{letter}.mp4", assets_folder)
                    if found_vid_letter:
                        media_paths.append(url_for('static', filename=f'assets/{found_vid_letter}'))
                        word_names.append(letter.upper())
    return jsonify({'media': media_paths, 'words': word_names})

def open_browser():
    url = 'http://localhost:5000/'
    chrome_path = 'C:/Program Files/Google/Chrome/Application/chrome.exe %s'
    try: webbrowser.get(chrome_path).open(url)
    except: webbrowser.open_new(url)

if __name__ == '__main__':
    Timer(1, open_browser).start()
    app.run(debug=True, use_reloader=False)
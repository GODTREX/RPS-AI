#PREDICTION SUCCESS V2 this works perfectly

import cv2
import random
import mediapipe as mp
import customtkinter as ctk
import threading
import time
from playsound import playsound
from collections import defaultdict
from PIL import Image, ImageTk
import numpy as np
import json
import os
from datetime import datetime
import queue

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Gesture Mapping
gesture_map = {
    0: 'Rock',
    1: 'Paper',
    2: 'Scissors'
}

# Difficulty Levels
DIFFICULTY_LEVELS = {
    'Easy': 0.3,    # 30% chance of making a random move
    'Medium': 0.5,  # 50% chance of making a random move
    'Hard': 0.8     # 80% chance of making a random move
}

# Sound effects
def play_sound(effect):
    try:
        playsound(effect, block=False)
    except:
        pass

# Detect hand gesture
def detect_gesture(hand_landmarks):
    if not hand_landmarks:
        return None

    landmarks = hand_landmarks[0].landmark
    fingers = []

    # Thumb
    fingers.append(landmarks[4].x < landmarks[3].x)
    # Other 4 fingers
    for tip in [8, 12, 16, 20]:
        fingers.append(landmarks[tip].y < landmarks[tip - 2].y)

    if fingers == [False, False, False, False, False]:
        return 0  # Rock
    elif fingers == [True, True, True, True, True]:
        return 1  # Paper
    elif fingers == [False, True, True, False, False]:
        return 2  # Scissors
    else:
        return None

# Game Logic
def get_winner(player_move, ai_move):
    if player_move == ai_move:
        return 'Tie'
    elif (player_move == 'Rock' and ai_move == 'Scissors') or \
         (player_move == 'Scissors' and ai_move == 'Paper') or \
         (player_move == 'Paper' and ai_move == 'Rock'):
        return 'Player'
    else:
        return 'AI'

# Statistics Manager
class StatisticsManager:
    def __init__(self):
        self.stats_file = 'game_stats.json'
        self.load_stats()

    def load_stats(self):
        if os.path.exists(self.stats_file):
            with open(self.stats_file, 'r') as f:
                self.stats = json.load(f)
        else:
            self.stats = {
                'total_games': 0,
                'wins': 0,
                'losses': 0,
                'ties': 0,
                'best_streak': 0,
                'move_history': [],
                'last_played': None
            }

    def save_stats(self):
        with open(self.stats_file, 'w') as f:
            json.dump(self.stats, f)

    def update_stats(self, result, move):
        self.stats['total_games'] += 1
        if result == 'Player':
            self.stats['wins'] += 1
        elif result == 'AI':
            self.stats['losses'] += 1
        else:
            self.stats['ties'] += 1
        
        self.stats['move_history'].append(move)
        self.stats['last_played'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.save_stats()

    def get_win_rate(self):
        if self.stats['total_games'] == 0:
            return 0
        return (self.stats['wins'] / self.stats['total_games']) * 100

# Main App Class
class RPSApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Rock-Paper-Scissors AI Predictive V2")
        self.geometry("1200x800")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # Initialize statistics
        self.stats_manager = StatisticsManager()

        # Game state
        self.player_score = 0
        self.ai_score = 0
        self.tie_score = 0
        self.streak = 0
        self.difficulty = 'Medium'
        self.move_history = []
        self.transition_counts = defaultdict(lambda: defaultdict(int))
        self.countdown_active = False
        self.game_active = False
        self.last_frame = None
        self.last_gesture = None

        # Camera frame queue
        self.frame_queue = queue.Queue(maxsize=4)  # Increased queue size
        self.camera_running = True

        # Create main frame
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Title
        self.label_title = ctk.CTkLabel(
            self.main_frame, 
            text="Rock Paper Scissors (AI Predicts!) V2", 
            font=("Arial", 32, "bold")
        )
        self.label_title.pack(pady=10)

        # Stats frame
        self.stats_frame = ctk.CTkFrame(self.main_frame)
        self.stats_frame.pack(fill="x", padx=20, pady=10)

        self.label_stats = ctk.CTkLabel(
            self.stats_frame,
            text=f"Win Rate: {self.stats_manager.get_win_rate():.1f}% | Total Games: {self.stats_manager.stats['total_games']}",
            font=("Arial", 16)
        )
        self.label_stats.pack(side="left", padx=10)

        # Score frame
        self.score_frame = ctk.CTkFrame(self.main_frame)
        self.score_frame.pack(fill="x", padx=20, pady=10)

        self.label_score = ctk.CTkLabel(
            self.score_frame, 
            text=self.get_score_text(), 
            font=("Arial", 20)
        )
        self.label_score.pack(side="left", padx=10)

        self.label_streak = ctk.CTkLabel(
            self.score_frame,
            text=f"Current Streak: {self.streak}",
            font=("Arial", 20)
        )
        self.label_streak.pack(side="right", padx=10)

        # Difficulty selector
        self.difficulty_frame = ctk.CTkFrame(self.main_frame)
        self.difficulty_frame.pack(fill="x", padx=20, pady=10)

        self.label_difficulty = ctk.CTkLabel(
            self.difficulty_frame,
            text="Difficulty:",
            font=("Arial", 16)
        )
        self.label_difficulty.pack(side="left", padx=10)

        self.difficulty_var = ctk.StringVar(value=self.difficulty)
        self.difficulty_menu = ctk.CTkOptionMenu(
            self.difficulty_frame,
            values=list(DIFFICULTY_LEVELS.keys()),
            variable=self.difficulty_var,
            command=self.change_difficulty
        )
        self.difficulty_menu.pack(side="left", padx=10)

        # Result label
        self.label_result = ctk.CTkLabel(
            self.main_frame, 
            text="Show your hand gesture!", 
            font=("Arial", 24)
        )
        self.label_result.pack(pady=20)

        # Countdown label
        self.label_countdown = ctk.CTkLabel(
            self.main_frame,
            text="",
            font=("Arial", 48, "bold")
        )
        self.label_countdown.pack(pady=10)

        # Buttons frame
        self.button_frame = ctk.CTkFrame(self.main_frame)
        self.button_frame.pack(pady=20)

        self.start_btn = ctk.CTkButton(
            self.button_frame, 
            text="Play Round", 
            command=self.play_round,
            font=("Arial", 16),
            width=200,
            height=40
        )
        self.start_btn.pack(side="left", padx=10)

        self.quit_btn = ctk.CTkButton(
            self.button_frame, 
            text="Quit", 
            command=self.on_closing,
            font=("Arial", 16),
            width=200,
            height=40
        )
        self.quit_btn.pack(side="left", padx=10)

        # Start camera thread
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.label_result.configure(text="Error: Could not open camera!")
            return

        self.camera_thread = threading.Thread(target=self.update_camera, daemon=True)
        self.camera_thread.start()

        # Start UI update thread
        self.after(10, self.update_ui)

    def get_score_text(self):
        return f"Player: {self.player_score} | AI: {self.ai_score} | Ties: {self.tie_score}"

    def change_difficulty(self, choice):
        self.difficulty = choice

    def update_camera(self):
        while self.camera_running:
            try:
                if not self.cap.isOpened():
                    time.sleep(0.1)
                    continue

                ret, frame = self.cap.read()
                if not ret:
                    continue

                frame = cv2.flip(frame, 1)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(frame_rgb)
                gesture_id = None

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                        gesture_id = detect_gesture([hand_landmarks])
                        if gesture_id is not None:
                            cv2.putText(
                                frame,
                                gesture_map[gesture_id],
                                (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (0, 255, 0),
                                2
                            )

                self.last_frame = frame
                self.last_gesture = gesture_map.get(gesture_id) if gesture_id is not None else None

                # Update frame queue
                try:
                    self.frame_queue.put_nowait((frame, gesture_id))
                except queue.Full:
                    try:
                        self.frame_queue.get_nowait()  # Remove old frame
                        self.frame_queue.put_nowait((frame, gesture_id))
                    except (queue.Empty, queue.Full):
                        pass

            except Exception as e:
                print(f"Camera error: {e}")
                time.sleep(0.1)

    def update_ui(self):
        if not self.camera_running:
            return

        try:
            frame, _ = self.frame_queue.get_nowait()
            cv2.imshow("Camera Feed", frame)
        except queue.Empty:
            pass

        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.camera_running = False
            self.destroy()
            return

        self.after(10, self.update_ui)

    def countdown(self):
        try:
            for i in range(3, 0, -1):
                if not self.countdown_active:
                    break
                self.label_countdown.configure(text=str(i))
                self.update()
                time.sleep(0.5)  # Reduced sleep time
            if self.countdown_active:
                self.label_countdown.configure(text="GO!")
                self.update()
                time.sleep(0.2)  # Reduced sleep time
                self.label_countdown.configure(text="")
        except Exception as e:
            print(f"Countdown error: {e}")
        finally:
            self.countdown_active = False

    def predict_next_move(self):
        if len(self.move_history) < 2:
            return random.choice(list(gesture_map.values()))

        # Apply difficulty level
        if random.random() < DIFFICULTY_LEVELS[self.difficulty]:
            return random.choice(list(gesture_map.values()))

        last_move = self.move_history[-1]
        next_moves = self.transition_counts[last_move]
        if not next_moves:
            return random.choice(list(gesture_map.values()))

        predicted_move = max(next_moves, key=next_moves.get)

        # Counter move to beat the predicted move
        counter = {
            'Rock': 'Paper',
            'Paper': 'Scissors',
            'Scissors': 'Rock'
        }
        return counter.get(predicted_move, random.choice(list(gesture_map.values())))

    def play_round(self):
        if self.countdown_active or self.game_active:
            return

        self.countdown_active = True
        self.game_active = True
        self.start_btn.configure(state="disabled")
        
        # Start countdown in a separate thread
        threading.Thread(target=self.countdown, daemon=True).start()
        
        def game_round():
            try:
                # Wait for countdown
                while self.countdown_active:
                    time.sleep(0.05)

                self.label_result.configure(text="Detecting gesture...")
                self.update()

                # Give player time to show gesture
                detection_start = time.time()
                player_move = None
                while time.time() - detection_start < 1.0:  # 1 second to detect gesture
                    if self.last_gesture is not None:
                        player_move = self.last_gesture
                        break
                    time.sleep(0.05)

                if player_move is None:
                    self.label_result.configure(text="Couldn't detect move! Try Again.")
                    play_sound("error.wav")
                    return

                ai_move = self.predict_next_move()

                # Update move history
                if self.move_history:
                    self.transition_counts[self.move_history[-1]][player_move] += 1
                self.move_history.append(player_move)

                winner = get_winner(player_move, ai_move)

                # Update streak and stats
                if winner == 'Player':
                    self.player_score += 1
                    self.streak += 1
                    play_sound("win.wav")
                elif winner == 'AI':
                    self.ai_score += 1
                    self.streak = 0
                    play_sound("lose.wav")
                else:
                    self.tie_score += 1
                    self.streak = 0
                    play_sound("tie.wav")

                # Update statistics
                self.stats_manager.update_stats(winner, player_move)
                if self.streak > self.stats_manager.stats['best_streak']:
                    self.stats_manager.stats['best_streak'] = self.streak
                    self.stats_manager.save_stats()

                self.label_result.configure(text=f"You: {player_move} | AI: {ai_move} | {winner} wins!")
                self.label_score.configure(text=self.get_score_text())
                self.label_streak.configure(text=f"Current Streak: {self.streak}")
                self.label_stats.configure(
                    text=f"Win Rate: {self.stats_manager.get_win_rate():.1f}% | Total Games: {self.stats_manager.stats['total_games']}"
                )

            except Exception as e:
                print(f"Game round error: {e}")
                self.label_result.configure(text="An error occurred! Try again.")
            finally:
                self.game_active = False
                self.start_btn.configure(state="normal")

        # Start game round in a separate thread
        threading.Thread(target=game_round, daemon=True).start()

    def on_closing(self):
        self.camera_running = False
        self.countdown_active = False
        self.game_active = False
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        self.destroy()

if __name__ == "__main__":
    app = RPSApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop() 
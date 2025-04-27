import cv2
import random
import mediapipe as mp
import customtkinter as ctk
import threading
import time
from playsound import playsound
from collections import defaultdict

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

# Main App Class
class RPSApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Rock-Paper-Scissors AI Predictive")
        self.geometry("800x600")
        ctk.set_appearance_mode("dark")

        # Score
        self.player_score = 0
        self.ai_score = 0
        self.tie_score = 0

        # Player move history
        self.move_history = []
        self.transition_counts = defaultdict(lambda: defaultdict(int))

        # Widgets
        self.label_title = ctk.CTkLabel(self, text="Rock Paper Scissors (AI Predicts!)", font=("Arial", 32))
        self.label_title.pack(pady=10)

        self.label_score = ctk.CTkLabel(self, text=self.get_score_text(), font=("Arial", 20))
        self.label_score.pack(pady=5)

        self.label_result = ctk.CTkLabel(self, text="Show your hand gesture!", font=("Arial", 24))
        self.label_result.pack(pady=20)

        self.start_btn = ctk.CTkButton(self, text="Play Round", command=self.play_round)
        self.start_btn.pack(pady=10)

        self.quit_btn = ctk.CTkButton(self, text="Quit", command=self.on_closing)
        self.quit_btn.pack(pady=10)

        # Camera
        self.cap = cv2.VideoCapture(0)
        self.running = True
        threading.Thread(target=self.update_camera, daemon=True).start()

    def get_score_text(self):
        return f"Player: {self.player_score} | AI: {self.ai_score} | Ties: {self.tie_score}"

    def update_camera(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            cv2.imshow("Camera Feed", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def predict_next_move(self):
        if len(self.move_history) < 2:
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
        self.label_result.configure(text="Detecting gesture...")
        self.update()
        time.sleep(1)

        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        player_move = None

        if results.multi_hand_landmarks:
            gesture_id = detect_gesture(results.multi_hand_landmarks)
            if gesture_id is not None:
                player_move = gesture_map.get(gesture_id)

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

        if winner == 'Player':
            self.player_score += 1
            play_sound("win.wav")
        elif winner == 'AI':
            self.ai_score += 1
            play_sound("lose.wav")
        else:
            self.tie_score += 1
            play_sound("tie.wav")

        self.label_result.configure(text=f"You: {player_move} | AI: {ai_move} | {winner} wins!")
        self.label_score.configure(text=self.get_score_text())

    def on_closing(self):
        self.running = False
        self.destroy()

if __name__ == "__main__":
    app = RPSApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()

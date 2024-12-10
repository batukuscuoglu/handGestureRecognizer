import cv2
import mediapipe as mp
import numpy as np
import time
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hand_detector = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawer = mp.solutions.drawing_utils

# Gesture tracking
current_gesture = "No Gesture"
paused = False
selected_item_index = 0
last_gesture = None
last_gesture_time = 0
gesture_stability_threshold = 0.5  # Gesture must persist for 0.5 seconds to be recognized

# Cursor position
cursor_x, cursor_y = 0, 0

# Gesture detection functions
def detect_thumbs_up_pose(landmarks):
    """Detect 'Thumbs Up' gesture."""
    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]
    thumb_mcp = landmarks[2]

    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]

    thumb_up = thumb_tip.y < thumb_ip.y and thumb_tip.y < thumb_mcp.y
    other_fingers_down = all(thumb_tip.y < tip.y for tip in [index_tip, middle_tip, ring_tip, pinky_tip])
    return thumb_up and other_fingers_down

def detect_thumbs_down_pose(landmarks):
    """Detect 'Thumbs Down' gesture."""
    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]
    thumb_mcp = landmarks[2]

    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]

    thumb_down = thumb_tip.y > thumb_ip.y and thumb_tip.y > thumb_mcp.y
    other_fingers_up = all(thumb_tip.y > tip.y for tip in [index_tip, middle_tip, ring_tip, pinky_tip])
    return thumb_down and other_fingers_up

def detect_rock_sign(landmarks):
    """Detect 'Rock Sign' gesture with pinky and index fingers raised."""
    pinky_tip = landmarks[20]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    thumb_tip = landmarks[4]

    pinky_up = pinky_tip.y < landmarks[18].y
    index_up = index_tip.y < landmarks[6].y
    middle_down = middle_tip.y > landmarks[10].y
    ring_down = ring_tip.y > landmarks[14].y
    thumb_down = thumb_tip.y > landmarks[3].y

    return pinky_up and index_up and middle_down and ring_down and thumb_down

def detect_pointing(landmarks):
    """Detect 'Point Gesture' (only index finger extended)."""
    index_tip = landmarks[8]
    index_mcp = landmarks[5]
    other_finger_tips = [landmarks[12], landmarks[16], landmarks[20]]
    other_finger_mcps = [landmarks[9], landmarks[13], landmarks[17]]

    index_extended = index_tip.y < index_mcp.y
    others_curled = all(tip.y > mcp.y for tip, mcp in zip(other_finger_tips, other_finger_mcps))
    return index_extended and others_curled

def detect_stop(landmarks):
    """Detect 'Stop' gesture (open palm) for both hands."""
    finger_tips = [landmarks[8], landmarks[12], landmarks[16], landmarks[20]]
    finger_bases = [landmarks[5], landmarks[9], landmarks[13], landmarks[17]]
    thumb_tip = landmarks[4]
    thumb_mcp = landmarks[2]

    fingers_extended = all(tip.y < base.y for tip, base in zip(finger_tips, finger_bases))
    thumb_aligned = abs(thumb_tip.x - thumb_mcp.x) > 0.1  # Ensure thumb is extended outward

    return fingers_extended and thumb_aligned

# Initialize the GUI
root = tk.Tk()
root.title("Hand Gesture Detection")
root.geometry("1200x600")

# Divide the window into left (video) and right (UI) sections
frame_left = tk.Frame(root, width=800, height=600)
frame_left.grid(row=0, column=0, padx=10, pady=10)

frame_right = tk.Frame(root, width=400, height=600, bg="lightgray")
frame_right.grid(row=0, column=1, padx=10, pady=10)

# Canvas for displaying the video feed
canvas = tk.Canvas(frame_left, width=800, height=600)
canvas.pack()

# Cursor label
cursor_label = tk.Label(canvas, text="◉", font=("Arial", 18), fg="red", bg="black")
cursor_label.place_forget()  # Initially hidden

# Label to display the current detected gesture
gesture_label = tk.Label(frame_right, text="Gesture: No Gesture", font=("Arial", 16), bg="white", height=2)
gesture_label.pack(pady=20)

# Scrolling list of items
item_list = ["Item 1", "Item 2", "Item 3", "Item 4", "Item 5", "Item 6", "Item 7", "Item 8", "Item 9", "Item 10"]
listbox = tk.Listbox(frame_right, height=10, font=("Arial", 14))
for item in item_list:
    listbox.insert(tk.END, item)
listbox.pack(pady=20)
listbox.selection_set(selected_item_index)

# Define actions
def refresh_action():
    global selected_item_index
    gesture_label.config(text="Action: List Reset!")
    print("Resetting list to the top.")
    selected_item_index = 0
    listbox.selection_clear(0, tk.END)
    listbox.selection_set(selected_item_index)

def next_item_action():
    global selected_item_index
    if selected_item_index < len(item_list) - 1:
        selected_item_index += 1
        gesture_label.config(text=f"Action: Moved to {item_list[selected_item_index]}")
        print(f"Moved to {item_list[selected_item_index]}")
        listbox.selection_clear(0, tk.END)
        listbox.selection_set(selected_item_index)

def previous_item_action():
    global selected_item_index
    if selected_item_index > 0:
        selected_item_index -= 1
        gesture_label.config(text=f"Action: Moved to {item_list[selected_item_index]}")
        print(f"Moved to {item_list[selected_item_index]}")
        listbox.selection_clear(0, tk.END)
        listbox.selection_set(selected_item_index)

def stop_action():
    global paused
    if not paused:
        gesture_label.config(text="Action: Paused!")
        print("Paused gesture detection!")
        paused = True
    else:
        gesture_label.config(text="Action: Resumed!")
        print("Resumed gesture detection!")
        paused = False

# Buttons linked to gestures
button_frame = tk.Frame(frame_right, bg="lightgray")
button_frame.pack(pady=20)

# Add buttons
button_1 = tk.Button(button_frame, text="Previous Item (Thumbs Up)", command=previous_item_action, width=20, height=2)
button_1.grid(row=0, column=0, pady=10)

button_2 = tk.Button(button_frame, text="Next Item (Thumbs Down)", command=next_item_action, width=20, height=2)
button_2.grid(row=1, column=0, pady=10)

button_3 = tk.Button(button_frame, text="Refresh (Rock Sign)", command=refresh_action, width=20, height=2)
button_3.grid(row=2, column=0, pady=10)

stop_button = tk.Button(button_frame, text="Pause/Resume (Stop)", command=stop_action, width=20, height=2)
stop_button.grid(row=3, column=0, pady=10)

# Initialize video capture
cap = cv2.VideoCapture(0)

# Process the video feed and update the GUI
def update_video_feed():
    global current_gesture, last_gesture, last_gesture_time, paused, cursor_x, cursor_y
    ret, frame = cap.read()
    if not ret:
        return

    frame = cv2.flip(frame, 1)  # Mirror the video feed
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hand_detector.process(rgb_frame)

    if not paused:
        detected_gesture = "No Gesture"
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawer.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                if detect_pointing(hand_landmarks.landmark):
                    index_tip = hand_landmarks.landmark[8]
                    cursor_x = int(index_tip.x * canvas.winfo_width())
                    cursor_y = int(index_tip.y * canvas.winfo_height())
                    cursor_label.place(x=cursor_x, y=cursor_y)
                    detected_gesture = "Point"

                elif detect_thumbs_up_pose(hand_landmarks.landmark):
                    cursor_label.place_forget()
                    detected_gesture = "Thumbs Up"
                elif detect_thumbs_down_pose(hand_landmarks.landmark):
                    cursor_label.place_forget()
                    detected_gesture = "Thumbs Down"
                elif detect_rock_sign(hand_landmarks.landmark):
                    cursor_label.place_forget()
                    detected_gesture = "Rock Sign"
                elif detect_stop(hand_landmarks.landmark):
                    cursor_label.place_forget()
                    detected_gesture = "Stop"

            # Check gesture stability
            if detected_gesture == last_gesture:
                if time.time() - last_gesture_time > gesture_stability_threshold:
                    current_gesture = detected_gesture
                    last_gesture_time = time.time()

                    if current_gesture == "Thumbs Up":
                        previous_item_action()
                    elif current_gesture == "Thumbs Down":
                        next_item_action()
                    elif current_gesture == "Rock Sign":
                        refresh_action()
                    elif current_gesture == "Stop":
                        stop_action()
            else:
                last_gesture = detected_gesture
                last_gesture_time = time.time()
        else:
            current_gesture = "No Gesture"

        # Update the gesture label
        gesture_label.config(text=f"Gesture: {current_gesture}")

    else:
        gesture_label.config(text="Gesture: Paused")

    # Convert the frame to ImageTk for displaying in Tkinter
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    imgtk = ImageTk.PhotoImage(image=img)
    canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
    canvas.imgtk = imgtk

    # Call this function again after 10ms
    root.after(10, update_video_feed)

# Start the video feed update loop
update_video_feed()

# Run the Tkinter main loop
root.mainloop()

# Release the video capture and destroy OpenCV windows when the program exits
cap.release()
cv2.destroyAllWindows()

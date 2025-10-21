import cv2
import mediapipe as mp
import numpy as np
import math
import time

print("Virtual Mouse script loaded...")

# Initialize Mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Canvas setup
canvas = None
current_color = (0, 0, 255)  # Start with red
brush_size = 5
eraser_size = 20
prev_pos = None
current_tool = "brush"
scroll_index = 0
brush_type = 'line'

undo_stack = []
redo_stack = []

# Colors palette
colors = [
    {"name": "Red", "color": (0, 0, 255)},
    {"name": "Green", "color": (0, 255, 0)},
    {"name": "Blue", "color": (255, 0, 0)},
    {"name": "Yellow", "color": (0, 255, 255)},
    {"name": "Purple", "color": (128, 0, 128)},
    {"name": "Black", "color": (0, 0, 0)},
    {"name": "White", "color": (255, 255, 255)}
]

# Tools
tools = [
    {"name": "Brush", "icon": "B"},
    {"name": "Eraser", "icon": "E"},
    {"name": "Clear", "icon": "C"}
]

def init_canvas(frame):
    return np.zeros_like(frame)

def draw_toolbar(frame, selected_color_idx, selected_tool_idx, scroll_active=False):
    h, w = frame.shape[:2]
    toolbar_w = 100
    toolbar = np.zeros((h, toolbar_w, 3), dtype=np.uint8)
    toolbar[:] = (240, 240, 240)

    if scroll_active:
        cv2.rectangle(toolbar, (0, 0), (toolbar_w, h), (200, 255, 200), 3)

    color_btn_h = 40
    for i, color in enumerate(colors):
        y1 = i * color_btn_h + 10
        y2 = y1 + color_btn_h - 10
        x1, x2 = 10, 40
        rect_color = color["color"]
        cv2.rectangle(toolbar, (x1, y1), (x2, y2), rect_color, -1)
        if i == selected_color_idx:
            cv2.rectangle(toolbar, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (0, 0, 0), 2)

    tool_start_y = len(colors) * color_btn_h + 20
    for i, tool in enumerate(tools):
        y1 = tool_start_y + i * color_btn_h
        y2 = y1 + color_btn_h - 10
        x1, x2 = 10, 80
        cv2.rectangle(toolbar, (x1, y1), (x2, y2), (200, 200, 200), -1)
        cv2.putText(toolbar, tool["icon"], (x1 + 30, y1 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        if i == selected_tool_idx:
            cv2.rectangle(toolbar, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (0, 0, 255), 2)

    frame[:, :toolbar_w] = toolbar
    return frame

def is_pinch(landmarks):
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    distance = math.hypot(thumb_tip.x - index_tip.x, thumb_tip.y - index_tip.y)
    return distance < 0.05

def count_raised_fingers(landmarks):
    tips_ids = [4, 8, 12, 16, 20]
    raised = 0
    for tip_id in tips_ids:
        if landmarks[tip_id].y < landmarks[tip_id - 2].y:
            raised += 1
    return raised

def save_state():
    global undo_stack, canvas
    undo_stack.append(canvas.copy())
    if len(undo_stack) > 10:
        undo_stack.pop(0)

def undo():
    global canvas, undo_stack, redo_stack
    if undo_stack:
        redo_stack.append(canvas.copy())
        canvas = undo_stack.pop()

def redo():
    global canvas, undo_stack, redo_stack
    if redo_stack:
        undo_stack.append(canvas.copy())
        canvas = redo_stack.pop()

def draw_with_brush(canvas, start_point, end_point, color, brush):
    if brush == 'line':
        cv2.line(canvas, start_point, end_point, color, brush_size)
    elif brush == 'dot':
        cv2.circle(canvas, end_point, brush_size, color, -1)
    elif brush == 'spray':
        for _ in range(30):
            offset_x = np.random.randint(-brush_size, brush_size)
            offset_y = np.random.randint(-brush_size, brush_size)
            if offset_x**2 + offset_y**2 <= brush_size**2:
                px = end_point[0] + offset_x
                py = end_point[1] + offset_y
                if 0 <= px < canvas.shape[1] and 0 <= py < canvas.shape[0]:
                    canvas[py, px] = color

def main():
    global canvas, current_color, current_tool, prev_pos, brush_type
    print("Virtual Mouse started...")

    selected_color_idx = 0
    selected_tool_idx = 0
    prev_y = None
    scroll_active = False
    scroll_threshold = 15
    last_scroll_time = 0
    hand_detected_time = None
    can_draw = False

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    ) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame.")
                break

            frame = cv2.flip(frame, 1)
            if canvas is None:
                canvas = init_canvas(frame)

            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            finger_pos = None
            is_pinching = False

            if results.multi_hand_landmarks:
                if hand_detected_time is None:
                    hand_detected_time = cv2.getTickCount()
                    can_draw = False
                else:
                    elapsed_time = (cv2.getTickCount() - hand_detected_time) / cv2.getTickFrequency()
                    if elapsed_time > 2:
                        can_draw = True

                for hand_landmarks in results.multi_hand_landmarks:
                    index_finger = hand_landmarks.landmark[8]
                    finger_pos = (int(index_finger.x * w), int(index_finger.y * h))
                    is_pinching = is_pinch(hand_landmarks.landmark)
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    if finger_pos[0] < 100:
                        scroll_active = True
                        if prev_y is not None:
                            diff = finger_pos[1] - prev_y
                            current_time = cv2.getTickCount()

                            if (current_time - last_scroll_time) / cv2.getTickFrequency() > 0.2:
                                if abs(diff) > scroll_threshold:
                                    if diff < 0:
                                        if selected_color_idx > 0:
                                            selected_color_idx -= 1
                                        elif selected_tool_idx > 0:
                                            selected_tool_idx -= 1
                                    else:
                                        if selected_color_idx < len(colors) - 1:
                                            selected_color_idx += 1
                                        elif selected_tool_idx < len(tools) - 1:
                                            selected_tool_idx += 1

                                    current_color = colors[selected_color_idx]["color"]
                                    current_tool = tools[selected_tool_idx]["name"].lower()
                                    if current_tool == "clear":
                                        canvas = init_canvas(frame)

                                    last_scroll_time = current_time
                                    prev_y = finger_pos[1]
                        else:
                            prev_y = finger_pos[1]
                    else:
                        scroll_active = False
                        prev_y = None

                    raised = count_raised_fingers(hand_landmarks.landmark)
                    if raised == 3:
                        undo()
                    elif raised == 4:
                        redo()
                    elif raised == 5:
                        cv2.imwrite("drawing.png", canvas)

            else:
                hand_detected_time = None
                can_draw = False

            frame = draw_toolbar(frame, selected_color_idx, selected_tool_idx, scroll_active)

            if finger_pos and finger_pos[0] > 100 and can_draw:
                if is_pinching:
                    cv2.circle(frame, finger_pos, 10, (255, 255, 255), 3)
                    save_state()
                    cv2.line(canvas, prev_pos if prev_pos else finger_pos, finger_pos, (0, 0, 0), eraser_size)
                else:
                    cv2.circle(frame, finger_pos, 5, (0, 255, 0), -1)
                    if current_tool == "brush":
                        save_state()
                        draw_with_brush(canvas, prev_pos if prev_pos else finger_pos, finger_pos, current_color, brush_type)
                prev_pos = finger_pos
            else:
                prev_pos = None

            canvas_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
            _, canvas_mask = cv2.threshold(canvas_gray, 1, 255, cv2.THRESH_BINARY_INV)
            canvas_mask = cv2.cvtColor(canvas_mask, cv2.COLOR_GRAY2BGR)
            frame = cv2.bitwise_and(frame, canvas_mask)
            frame = cv2.bitwise_or(frame, canvas)

            cv2.putText(frame, f"Color: {colors[selected_color_idx]['name']}", (110, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, current_color, 2)
            cv2.putText(frame, f"Tool: {current_tool}", (110, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Brush: {brush_type}", (110, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 2)

            if scroll_active:
                cv2.putText(frame, "SCROLL MODE", (110, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if not can_draw and hand_detected_time is not None:
                cv2.putText(frame, "Starting in 2 seconds...", (110, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

            cv2.imshow("Virtual Paint", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('1'):
                brush_type = 'line'
            elif key == ord('2'):
                brush_type = 'dot'
            elif key == ord('3'):
                brush_type = 'spray'
            elif key == 26:  # Ctrl+Z
                undo()
            elif key == 25:  # Ctrl+Y
                redo()
            elif key == 19:  # Ctrl+S
                cv2.imwrite("drawing.png", canvas)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


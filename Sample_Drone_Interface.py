import sys
import time
import cv2
import numpy as np
import NDIlib as ndi
import os
import socket
import threading
import json
import random
import struct
import keyboard
from datetime import datetime
from pathlib import Path

from xlabs_nav.keyframe_io import append_keyframe_png, load_keyframe_manifest
from xlabs_nav.sample_nav_autopilot import compute_nav_p_command, try_load_sample_nav_runtime

# ==============================================================================
# SECTION 1: CONFIGURATION & GLOBAL STATE
# ==============================================================================
# This section holds all the settings and variables that track the drone's status.

# --- Network Configuration ---
HOST = '127.0.0.1'  # Localhost (The machine this script is running on)
PORT = 65432        # The port the Unity Drone communicates with

# --- AI Model Configuration ---
ENABLE_YOLO = False  # Set True to load yolov8n.pt and run detection / 'o' autopilot
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = Path(SCRIPT_DIR)
MISSION_CONFIG_PATH = REPO_ROOT / "config" / "mission_config.yaml"
KEYFRAMES_DIR = Path(SCRIPT_DIR) / "data" / "keyframes"
KEYFRAMES_MANIFEST = KEYFRAMES_DIR / "keyframes.json"
MODEL_CANDIDATES = [
    os.path.join(SCRIPT_DIR, '..', 'yolov8n.pt'),
    os.path.join(SCRIPT_DIR, 'yolov8n.pt'),
    os.path.join(SCRIPT_DIR, 'best.pt'),
]

# --- Drone Control State (The inputs we send to the drone) ---
control_state = {
    "btnAdown": False,      # Button A
    "btnBdown": False,      # Button B
    "btnCdown": False,      # Button C
    "btnARMdown": False,    # Arming Sequence
    "trigger": 0.0,         # Forward Speed (0.0 to 1.0)
    "trigger_down": False,  # Is 'W' pressed?
    "reverse": 0.0,         # Reverse Speed (0.0 to 1.0)
    "reverse_down": False,  # Is 'S' pressed?
    "joy_vertical": 0,      # Altitude Control (-1 to 1)
    "joy_horizontal": 0,    # Strafe Control (-1 to 1)
    "yaw": 0.0,             # Rotation Left/Right
    "pitch": 0.0,           # Tilt Up/Down (Camera)
    "thumb_down": False,    # Thumbstick press
    "joy_click": False,     # Joystick click
    # Internal helpers for smooth ramping
    "joy_up": False, "joy_down": False, 
    "joy_left": False, "joy_right": False,
    "arrow_left": False, "arrow_right": False, 
    "arrow_up": False, "arrow_down": False,
    "autopilot": False,  # YOLO autopilot ('o')
    "nav_autopilot": False,  # Keyframe / ORB nav autopilot ('n'); parallel flag for testing
}

# --- Telemetry Data ---
time_from_unity = 0.0
static_boxes = [
    {"x":-100,"y":-100,"width":100,"height":100,"id":"box1"},
    {"x":100,"y":-100,"width":100,"height":100,"id":"box2"},
    {"x":200,"y":-200,"width":120,"height":140,"id":"box3"}
]

# --- Helper Functions ---
def clamp(minimum, x, maximum):
    """Ensures a value 'x' stays between 'minimum' and 'maximum'."""
    return max(minimum, min(x, maximum))

def resolve_model_path():
    """Returns the first local YOLO model path that exists."""
    for model_path in MODEL_CANDIDATES:
        if os.path.exists(model_path):
            return model_path
    return None

def prepare_frame_for_yolo(frame):
    """Converts NDI BGRA frames to BGR before passing them to YOLO/OpenCV."""
    if len(frame.shape) == 3 and frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame

def run_yolo_inference(model, frame):
    """Runs YOLO on a frame and returns both raw results and an annotated image."""
    if model is None:
        return None, None

    try:
        yolo_frame = prepare_frame_for_yolo(frame)
        results = model(yolo_frame, verbose=False)
        if results:
            return results, results[0].plot()
    except Exception as e:
        print(f"Detection Error: {e}")

    return None, None

def detect_target_from_results(results):
    """
    Scans YOLO results for a 'window' (or the first object it sees).
    Returns the (x, y) center of the target relative to the image size.
    If nothing is found, returns None.
    """
    try:
        if results and len(results[0].boxes) > 0:
            # Take the first detection (highest confidence)
            box = results[0].boxes[0]
            x_center = box.xywh[0][0].item()
            y_center = box.xywh[0][1].item()
            return x_center, y_center
    except Exception as e:
        print(f"Target Detection Error: {e}")
    return None

# ==============================================================================
# SECTION 2: SERVER & CONTROL LOGIC (Background Threads)
# ==============================================================================

def on_key_event(event):
    """
    Callback function triggered whenever a key is pressed or released.
    It updates the global 'control_state' dictionary.
    """
    global control_state
    is_down = (event.event_type == "down")
    key = event.name

    # --- DEBUGGING ADDED HERE ---
    # This will print every single key press to the console
    print(f"[DEBUG] Key: {key} | Action: {event.event_type}") 
    # ----------------------------

    # Mapping keys to control states
    if key == "2": control_state["btnAdown"] = is_down
    if key == "b": control_state["btnBdown"] = is_down
    if key == "c": control_state["btnCdown"] = is_down
    if key == "1": control_state["btnARMdown"] = is_down
    if key == "w": control_state["trigger_down"] = is_down
    if key == "s": control_state["reverse_down"] = is_down
    if key == "e": control_state["joy_up"] = is_down
    if key == "f": control_state["joy_down"] = is_down
    if key == "a": control_state["joy_left"] = is_down
    if key == "d": control_state["joy_right"] = is_down
    if key == "k": control_state["joy_click"] = is_down
    if key == "left": control_state["arrow_left"] = is_down
    if key == "right": control_state["arrow_right"] = is_down
    if key == "up": control_state["arrow_up"] = is_down
    if key == "down": control_state["arrow_down"] = is_down
    if key == "p": control_state["thumb_down"] = is_down

def update_controls():
    """
    Runs continuously to smooth out controls (ramping up/down values).
    Simulates analog stick behavior using digital keys.
    """
    while True:
        # --- AUTOPILOT SAFETY ---
        # If YOLO or nav autopilot is flying, the keyboard logic goes to sleep
        if control_state["autopilot"] or control_state.get("nav_autopilot"):
            time.sleep(1/60)
            continue

        # Smooth Trigger (Gas)
        if control_state["trigger"] > 0 and not control_state["trigger_down"]:
           control_state["trigger"] = clamp(0, control_state["trigger"]-0.1, 1)
        if control_state["trigger"] < 1 and control_state["trigger_down"]:
            control_state["trigger"] = clamp(0, control_state["trigger"]+0.05, 1)
            
        # Smooth Reverse
        if control_state["reverse"] > 0 and not control_state["reverse_down"]:
           control_state["reverse"] = clamp(0, control_state["reverse"]-0.1, 1)
        if control_state["reverse"] < 1 and control_state["reverse_down"]:
            control_state["reverse"] = clamp(0, control_state["reverse"]+0.05, 1)

        # Joystick Logic
        if control_state["joy_up"]: control_state["joy_vertical"] = -1
        elif control_state["joy_down"]: control_state["joy_vertical"] = 1
        else: control_state["joy_vertical"] = 0

        if control_state["joy_left"]: control_state["joy_horizontal"] = -1
        elif control_state["joy_right"]: control_state["joy_horizontal"] = 1
        else: control_state["joy_horizontal"] = 0

        # Yaw (Rotation) Logic with smoothing
        if control_state["arrow_right"]:
            control_state["yaw"] = clamp(-1, control_state["yaw"]+0.05, 1)
        elif control_state["arrow_left"]:
            control_state["yaw"] = clamp(-1, control_state["yaw"]-0.05, 1)
        
        # Pitch (Camera Tilt) Logic with smoothing
        if control_state["arrow_up"]:
            control_state["pitch"] = clamp(-1, control_state["pitch"]-0.05, 1)
        elif control_state["arrow_down"]:
            control_state["pitch"] = clamp(-1, control_state["pitch"]+0.05, 1)
            
        # Reset Pitch/Yaw if 'C' is pressed
        if control_state["btnCdown"]:
            control_state["pitch"] = 0
            control_state["yaw"] = 0
            
        time.sleep(1/60) # Update 60 times per second

def listen_to_unity(conn):
    """Receives JSON data packets from the Unity Drone."""
    try:
        while True:
            raw_msglen = conn.recv(4)
            if not raw_msglen: break
            message_length = struct.unpack('>I', raw_msglen)[0]
            data = b''
            while len(data) < message_length:
                packet = conn.recv(message_length - len(data))
                if not packet: break
                data += packet
            
            try:
                message = json.loads(data)
                global time_from_unity
                time_from_unity = message.get("time", 0)
            except json.JSONDecodeError:
                pass
    except Exception as e:
        print(f"Connection Listener Error: {e}")

def send_to_unity(conn):
    """Packs the control_state into JSON and sends it to Unity."""
    try:
        while True:
            response = {
                "num_of_boxes": 3, "data": static_boxes,
                "time": time_from_unity,
                "btnAdown": control_state["btnAdown"],
                "btnBdown": control_state["btnBdown"],
                "btnCdown": control_state["btnCdown"],
                "btnARMdown": control_state["btnARMdown"],
                "trigger": control_state["trigger"],
                "triggerDown": control_state["trigger_down"],
                "reverse": control_state["reverse"],
                "reverseDown": control_state["reverse_down"],
                "joy_vertical": control_state["joy_vertical"],
                "joy_horizontal": control_state["joy_horizontal"],
                "yaw": control_state["yaw"],
                "pitch": control_state["pitch"],
                "thumbDown": control_state["thumb_down"],
                "joyClick": control_state["joy_click"]
            }
            response_bytes = json.dumps(response).encode('utf-8')
            response_length = len(response_bytes)
            conn.send(response_length.to_bytes(4, byteorder='big'))
            conn.send(response_bytes)
            time.sleep(1/60) # Send at 60Hz
    except (ConnectionResetError, BrokenPipeError):
        print("Unity connection lost (Sender).")

# ==============================================================================
# SECTION 3: VIDEO PLAYER & MAIN EXECUTION
# ==============================================================================

def main():
    # --- 1. Initialize Server ---
    print("--- 1. STARTING CONTROL SERVER ---")
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    try:
        server_socket.bind((HOST, PORT))
        server_socket.listen(1)
        print(f"Waiting for Unity Drone to connect on {HOST}:{PORT}...")
        conn, addr = server_socket.accept()
        print(f"Connected to Drone at {addr}")
    except Exception as e:
        print(f"CRITICAL ERROR: Could not start server. Is the port blocked? {e}")
        return

    # Start Control Threads
    threading.Thread(target=listen_to_unity, args=(conn,), daemon=True).start()
    threading.Thread(target=send_to_unity, args=(conn,), daemon=True).start()
    threading.Thread(target=update_controls, daemon=True).start()
    
    # IMPORTANT: Keyboard hook sometimes requires ADMIN privileges to work correctly
    try:
        keyboard.hook(on_key_event)
        print("Controls Active. Listening for keyboard inputs.")
    except Exception as e:
        print(f"ERROR: Could not hook keyboard. try running as Administrator. Details: {e}")

    # --- 2. Initialize NDI Video ---
    print("\n--- 2. STARTING NDI VIDEO STREAM ---")
    if not ndi.initialize():
        print("ERROR: Could not initialize NDI.")
        return 0

    ndi_find = ndi.find_create_v2()
    if ndi_find is None: return 0

    sources = []
    print("Looking for NDI sources (Drone Camera)...")
    for _ in range(10): # Wait up to 10 iterations
        ndi.find_wait_for_sources(ndi_find, 500)
        sources = ndi.find_get_current_sources(ndi_find)
        if len(sources) > 0: break
            
    if not sources:
        print("WARNING: No NDI sources found. Is Unity running? (Press Ctrl+C to quit)")
    
    # Prefer a source with "Unity" in the name
    selected_source = sources[0] if sources else None
    if sources:
        for s in sources:
            print(f"Found source: {s.ndi_name}")
            if "Unity" in s.ndi_name: selected_source = s
        print(f"Connecting to video: {selected_source.ndi_name}")
    else:
        print("Running in CONTROL ONLY mode (No Video).")

    # --- 2.5 Load AI Model ---
    print("--- 2.5 LOADING AI MODEL ---")
    model = None
    if ENABLE_YOLO:
        try:
            from ultralytics import YOLO

            model_path = resolve_model_path()
            if model_path is None:
                raise FileNotFoundError("Could not find yolov8n.pt or best.pt near Sample_Drone_Interface.py")
            print(f"Loading model: {model_path}...")
            model = YOLO(model_path)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"ERROR: Could not load model. {e}")
            model = None
    else:
        print("YOLO disabled (ENABLE_YOLO=False). No model loaded.")

    # Setup Receiver
    ndi_recv = None
    if selected_source:
        ndi_recv_create = ndi.RecvCreateV3()
        ndi_recv_create.color_format = ndi.RECV_COLOR_FORMAT_BGRX_BGRA
        ndi_recv = ndi.recv_create_v3(ndi_recv_create)
        ndi.recv_connect(ndi_recv, selected_source)
        ndi.find_destroy(ndi_find)

    # --- 3. Main Display Loop ---
    # Setup Output for Recording
    output_dir = "OUTPUT"
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    
    recording = False
    video_writer = None
    window_name = "Drone View (NDI)"
    scale_factor = 0.5
    
    try:
        kf_entries = load_keyframe_manifest(KEYFRAMES_MANIFEST)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"WARNING: Could not load keyframes.json ({e}). Starting with empty keyframe list.")
        kf_entries = []

    nav_mission, nav_km, nav_load_err = try_load_sample_nav_runtime(
        REPO_ROOT, mission_config_path=MISSION_CONFIG_PATH
    )
    if nav_km is None:
        print(f"NOTE: Keyframe nav autopilot unavailable ({nav_load_err}). Hold 'n' will do nothing.")
    else:
        print("Keyframe nav autopilot ready: hold 'n' to servo on active keyframe (parallel to YOLO 'o').")

    print("\n=== SYSTEM READY ===")
    print("Drone Controls: WASD, Arrows, 1 (Arm), B (Land)")
    print("Video Controls: 'r' (Record), +/- (Zoom), 'q' (Quit)")
    print("Keyframes: SPACE saves current camera view (same NDI stream). Do not run record_keyframes.py together.")
    print("Autopilot: hold 'o' if YOLO is enabled; hold 'n' for keyframe nav (ORB error -> P on yaw/pitch/trigger).")
    print("DEBUG MODE: Press any key to see if it registers in the console...")

    # 3.1 Verify OpenCV installation supports GUI
    try:
        # Just test if we can create a window logic (won't actually show until waitKey)
        pass 
    except Exception as e:
        print(f"Warning: OpenCV check failed. {e}")

    last_bgr_for_keyframe = None
    try:
        while True:
            # Release-to-manual must run every loop iteration, not only on NDI video frames.
            # Otherwise nav_autopilot / autopilot can stay True while update_controls() keeps
            # sleeping and keyboard smoothing never runs (feels like "manual is dead").
            if control_state.get("nav_autopilot") and not keyboard.is_pressed("n"):
                print("Nav autopilot disengaged.")
                control_state["nav_autopilot"] = False
                control_state["yaw"] = 0
                control_state["pitch"] = 0
                control_state["joy_horizontal"] = 0
                control_state["joy_vertical"] = 0
                control_state["trigger"] = 0
                control_state["reverse"] = 0

            if control_state["autopilot"] and not keyboard.is_pressed("o"):
                print("Autopilot Disengaged.")
                control_state["autopilot"] = False
                control_state["yaw"] = 0
                control_state["pitch"] = 0
                control_state["trigger"] = 0

            # A. Capture Frame
            if ndi_recv:
                t, v, a, _ = ndi.recv_capture_v2(ndi_recv, 1000)
                
                if t == ndi.FRAME_TYPE_VIDEO:
                    frame = np.copy(v.data) # Copy data to numpy array
                    last_bgr_for_keyframe = prepare_frame_for_yolo(frame)

                    # B. Run YOLO inference and prepare an annotated frame for display/recording
                    yolo_results, annotated_frame = run_yolo_inference(model, frame)
                    output_frame = annotated_frame if annotated_frame is not None else prepare_frame_for_yolo(frame)
                    nav_overlay_lines: list[str] = []

                    # C. Recording Logic
                    if recording:
                        if video_writer is None:
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            fname = os.path.join(output_dir, f"flight_{timestamp}.mp4")
                            # MP4V codec
                            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                            video_writer = cv2.VideoWriter(fname, fourcc, 30.0, (v.xres, v.yres))
                            print(f"Recording started: {fname}")
                        
                        if video_writer and video_writer.isOpened():
                            video_writer.write(output_frame)
                    elif video_writer is not None:
                        video_writer.release()
                        video_writer = None
                        print("Recording stopped.")

                    # --- AUTOPILOT LOGIC ---
                    if keyboard.is_pressed('o') and model is not None:
                        control_state["autopilot"] = True
                        control_state["nav_autopilot"] = False

                        # 1. Look for target
                        target = detect_target_from_results(yolo_results)
                        
                        if target:
                            tx, ty = target
                            h, w, _ = frame.shape
                            cx, cy = w / 2, h / 2
                            
                            # 2. Calculate Errors
                            # X-Axis: if target is to the right, we turn right (positive Yaw)
                            error_x = (tx - cx) / cx # -1 to 1
                            # Y-Axis: if target is above, we aim up (negative Pitch)
                            error_y = (ty - cy) / cy # -1 to 1

                            # 3. Apply Controls (P-Controller)
                            SENSITIVITY = 0.8
                            
                            # Update Yaw
                            control_state["yaw"] = clamp(-1, error_x * SENSITIVITY, 1)
                            
                            # Update Pitch (Inverted logic: target up (y < cy) => needs negative pitch to look up? 
                            # Actually usually: Up arrow = Pitch -1 (Look Up). 
                            # If target is at y=0 (top), ty < cy. error_y is negative. We want Pitch -1. 
                            # So direct mapping works.)
                            control_state["pitch"] = clamp(-1, error_y * SENSITIVITY, 1)

                            # Constant Gas
                            control_state["trigger"] = 0.5
                            
                            # Debug Draw
                            cv2.circle(frame, (int(tx), int(ty)), 10, (0, 255, 0), -1)
                            cv2.line(frame, (int(cx), int(cy)), (int(tx), int(ty)), (0, 255, 0), 2)
                        
                        else:
                            # Lost target? Hover/Stop turning
                            control_state["yaw"] = 0
                            control_state["pitch"] = 0

                    elif keyboard.is_pressed("n") and nav_km is not None and nav_mission is not None:
                        control_state["autopilot"] = False

                        bgr_nav = last_bgr_for_keyframe
                        if bgr_nav is None:
                            bgr_nav = prepare_frame_for_yolo(frame)

                        # apply_servo_command writes yaw/strafe/altitude/trigger into
                        # control_state and sets nav_autopilot=True internally.
                        ve, applied = compute_nav_p_command(
                            bgr_nav, nav_mission, nav_km, control_state
                        )

                        if applied:
                            nav_overlay_lines = [
                                "NAV autopilot (keyframe P)",
                                f"dx_norm={ve.get('dx_norm', 0.0):.4f}  dy_norm={ve.get('dy_norm', 0.0):.4f}",
                                f"scale_err={ve.get('scale_error', 0.0):.4f}  inliers={ve.get('num_inliers', 0)}",
                            ]
                        else:
                            control_state["yaw"] = 0
                            control_state["joy_horizontal"] = 0
                            control_state["joy_vertical"] = 0
                            nav_overlay_lines = [
                                "NAV autopilot (no valid match)",
                                ve.get("reason", "") or "unknown",
                            ]

                    # C. Display Logic
                    display_frame = output_frame.copy()

                    y_hint = 28
                    for line in ("SPACE: save keyframe", f"Saved keyframes: {len(kf_entries)}"):
                        cv2.putText(
                            display_frame,
                            line,
                            (10, y_hint),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.65,
                            (0, 255, 0),
                            2,
                            cv2.LINE_AA,
                        )
                        y_hint += 28
                    for line in nav_overlay_lines:
                        cv2.putText(
                            display_frame,
                            line,
                            (10, y_hint),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 200, 0),
                            2,
                            cv2.LINE_AA,
                        )
                        y_hint += 26
                    
                    # Resize based on zoom
                    if scale_factor != 1.0:
                        h, w = display_frame.shape[:2]
                        display_frame = cv2.resize(display_frame, (int(w*scale_factor), int(h*scale_factor)))

                    # Draw Indicators
                    if recording:
                        cv2.circle(display_frame, (30, 30), 10, (0, 0, 255), -1) # Red Dot
                    
                    try:
                        cv2.imshow(window_name, display_frame)
                    except cv2.error as e:
                        if "not implemented" in str(e):
                            print("\n[CRITICAL ERROR] OpenCV Interface Missing")
                            print("Your installed 'opencv-python' library does not support windows.")
                            print("Please run this command in terminal:")
                            print("   pip uninstall opencv-python-headless -y && pip install opencv-python")
                            print("Exiting...")
                            break
                        else:
                            raise e

                    ndi.recv_free_video_v2(ndi_recv, v)
                
                elif t == ndi.FRAME_TYPE_AUDIO:
                    ndi.recv_free_audio_v2(ndi_recv, a)

            # D. Window Inputs (Video Player Controls)
            try:
                key = cv2.waitKey(1) & 0xFF
            except cv2.error:
                break

            if key == ord('q') or key == 27: # Quit
                break
            elif key == ord(' '):
                if last_bgr_for_keyframe is None:
                    print("Keyframe save skipped (no video frame yet).")
                else:
                    kf_entries, saved_path, kf_err = append_keyframe_png(
                        last_bgr_for_keyframe,
                        manifest_path=KEYFRAMES_MANIFEST,
                        keyframes_dir=KEYFRAMES_DIR,
                        entries=kf_entries,
                    )
                    if kf_err:
                        print(f"Keyframe save failed: {kf_err}")
                    elif saved_path is not None:
                        print(f"Keyframe saved: {saved_path} (total {len(kf_entries)})")
            elif key == ord('r'): # Toggle Record
                recording = not recording
            elif key == ord('+') or key == ord('='): # Zoom In
                scale_factor = min(scale_factor + 0.1, 2.0)
            elif key == ord('-') or key == ord('_'): # Zoom Out
                scale_factor = max(scale_factor - 0.1, 0.1)

    except KeyboardInterrupt:
        pass
    finally:
        print("Shutting down...")
        if video_writer: video_writer.release()
        if ndi_recv: ndi.recv_destroy(ndi_recv)
        ndi.destroy()
        try:
            cv2.destroyAllWindows()
        except:
            pass
        server_socket.close()

if __name__ == "__main__":
    main()
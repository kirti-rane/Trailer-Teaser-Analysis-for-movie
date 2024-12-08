import cv2

def extract_keyframes(video_path, interval=3):
    
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_interval = int(fps * interval)  # Number of frames to skip
    frame_count = 0
    keyframes = []  # Store keyframes as NumPy arrays

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            keyframes.append(frame)  # Append frame to the keyframes list
        
        frame_count += 1

    cap.release()
    print(f"Extracted {len(keyframes)} keyframes from {video_path}")
    return keyframes

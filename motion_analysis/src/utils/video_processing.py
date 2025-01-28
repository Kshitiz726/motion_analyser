import cv2
import numpy as np
import os
from skimage.metrics import structural_similarity as ssim

def capture_frames_from_folder(folder_path, resize_dim=(64, 64)):
    """Capture frames from image files in the folder and resize them."""
    frames = []
    for filename in sorted(os.listdir(folder_path)):  # Sort to maintain order
        if filename.endswith(".jpg"):
            frame_path = os.path.join(folder_path, filename)
            frame = cv2.imread(frame_path)
            if frame is not None:
                frame = cv2.resize(frame, resize_dim)
                frames.append(frame)
    return np.array(frames)

def preprocess_frames(frames):
    """Normalize frame data (scaling pixel values to [0, 1])."""
    return frames / 255.0

def detect_motion_in_frames(frames, ssim_threshold=1):
    """
    Detect motion in the frames using Structural Similarity Index (SSIM).
    If the SSIM between consecutive frames drops below the threshold, motion is detected.
    """
    motion_labels = []
    previous_frame = None

    for i in range(len(frames)):
        current_frame = frames[i]
        
        # Skip the first frame since there's no previous frame to compare it with
        if i == 0:
            motion_labels.append(0)  # No motion in the first frame
            previous_frame = current_frame
            continue
        
        # Ensure frames are in uint8 format before processing
        if current_frame.dtype != np.uint8:
            current_frame = (current_frame * 255).astype(np.uint8)
        if previous_frame.dtype != np.uint8:
            previous_frame = (previous_frame * 255).astype(np.uint8)

        # Convert frames to grayscale
        gray_previous = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
        gray_current = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        gray_previous = cv2.GaussianBlur(gray_previous, (5, 5), 0)
        gray_current = cv2.GaussianBlur(gray_current, (5, 5), 0)

        # Calculate SSIM between consecutive frames
        similarity, _ = ssim(gray_previous, gray_current, full=True)

        # Debug: Print the SSIM for each frame comparison
        # print(f"Frame {i}: SSIM = {similarity}")

        # If SSIM drops below the threshold, motion is detected
        if similarity < ssim_threshold:
            motion_labels.append(1)  # Motion detected
        else:
            motion_labels.append(0)  # No motion detected

        previous_frame = current_frame  # Update the previous frame

    return motion_labels

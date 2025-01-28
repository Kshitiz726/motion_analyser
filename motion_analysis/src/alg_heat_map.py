import numpy as np
import cv2

# Function to capture video frames
def capture_video(video_path=None):
    # Use camera if no video path is provided
    cap = cv2.VideoCapture(video_path if video_path else 0)
    return cap

# Function to preprocess the frames (convert to grayscale, and apply GaussianBlur)
def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    blurred = cv2.GaussianBlur(gray, (21, 21), 0)  # Apply GaussianBlur to reduce noise
    return blurred

# Function to detect motion between two frames
def detect_motion(prev_frame, curr_frame):
    # Calculate the absolute difference between the current and previous frame
    frame_diff = cv2.absdiff(curr_frame, prev_frame)
    
    # Threshold the difference to detect significant motion (simple method)
    _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
    return thresh

# Function to generate heatmap from motion detection
def generate_heatmap(thresh, heatmap_accumulator):
    # Add the detected motion to the heatmap accumulator
    heatmap_accumulator += thresh.astype(np.float32)
    
    # Normalize the heatmap to be in the range [0, 255] for visualization
    heatmap_norm = np.uint8(np.clip(heatmap_accumulator / 255, 0, 255))
    
    # Apply a colormap to create a colored heatmap (using a custom approach here)
    heatmap_colored = np.zeros_like(heatmap_norm, dtype=np.uint8)
    heatmap_colored[:, :] = (heatmap_norm[:, :] * np.array([0, 0, 255])).astype(np.uint8)  # Red color

    return heatmap_colored

# Function to create and stream the heatmap in video format
def stream_heatmap(video_path=None):
    cap = capture_video(video_path)  # Capture video from file or webcam
    ret, prev_frame = cap.read()  # Read the first frame
    
    # Preprocess the first frame
    prev_frame = preprocess_frame(prev_frame)

    # Initialize heatmap accumulator
    heatmap_accumulator = np.zeros_like(prev_frame, dtype=np.float32)

    while cap.isOpened():
        ret, curr_frame = cap.read()
        if not ret:
            break

        # Preprocess the current frame
        curr_frame_processed = preprocess_frame(curr_frame)

        # Detect motion by comparing current frame with the previous frame
        thresh = detect_motion(prev_frame, curr_frame_processed)

        # Generate heatmap based on the detected motion
        heatmap = generate_heatmap(thresh, heatmap_accumulator)

        # Display the current frame and the heatmap
        cv2.imshow("Original Frame", curr_frame)
        cv2.imshow("Heatmap", heatmap)

        # Update the previous frame for the next iteration
        prev_frame = curr_frame_processed

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Provide a video path if needed, else it will use the webcam
    video_path = None  # Set this to the video file path if required, e.g., 'video.mp4'
    stream_heatmap(video_path)

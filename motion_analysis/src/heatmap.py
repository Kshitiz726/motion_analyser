# heatmap.py

import cv2
import numpy as np

# Initialize a heatmap array (accumulates motion intensity)
heatmap = np.zeros((480, 640), dtype=np.float32)  # Adjust size based on your frame resolution

# Function to generate heatmap
def generate_heatmap(frame, fgbg, MIN_CONTOUR_AREA):
    global heatmap

    # Convert the frame to grayscale for motion detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply background subtraction to detect motion
    fgmask = fgbg.apply(gray_frame)

    # Refine the mask to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)  # Remove small noise
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_DILATE, kernel)  # Fill gaps in detected regions

    # Find contours (regions of motion)
    motion_contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Update the heatmap with detected motion
    for motion_contour in motion_contours:
        if cv2.contourArea(motion_contour) >= MIN_CONTOUR_AREA:
            # Get bounding box for each motion contour
            x, y, w, h = cv2.boundingRect(motion_contour)

            # Increment heatmap intensity where motion is detected
            heatmap[y:y+h, x:x+w] += 1

    # Normalize the heatmap to [0, 255] for visualization
    heatmap_normalized = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heatmap_colored = cv2.applyColorMap(heatmap_normalized.astype(np.uint8), cv2.COLORMAP_JET)

    # Overlay the heatmap on the original frame
    overlay = cv2.addWeighted(frame, 0.7, heatmap_colored, 0.3, 0)

    return overlay, heatmap_colored

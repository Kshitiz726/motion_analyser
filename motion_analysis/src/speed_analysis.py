import cv2
import numpy as np

# Minimum contour area to filter out small objects (noise)
MIN_CONTOUR_AREA = 4000  # Increase the contour area threshold
MAX_CONTOUR_AREA = 1000000  # Optionally filter out overly large objects (e.g., your background or wall)

# Initialize webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Forcing DirectShow on Windows
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

# Background subtractor for motion detection
fgbg = cv2.createBackgroundSubtractorMOG2()

# To store previous centers and times for speed calculation
prev_centers = {}  # stores the centers of objects in previous frames
prev_times = {}    # stores the timestamps of objects in previous frames

# Set the scale factor (in meters per pixel) based on real-world knowledge
real_width_m = 2.0  # example: the real width of the object in meters
pixel_width = 100   # example: the width of the object in pixels in the frame
scale_factor = real_width_m / pixel_width

def calculate_speed(center, prev_center, curr_time, prev_time, scale_factor):
    distance = np.linalg.norm(np.array(center) - np.array(prev_center))  # Euclidean distance
    time_diff = curr_time - prev_time  # Time difference between frames
    speed = (distance / time_diff) * scale_factor  # Convert to real-world units (m/s)
    return speed

def generate_speed_analysis_feed():
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame")
            break

        # Convert the frame to grayscale for motion detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply background subtraction to detect motion
        fgmask = fgbg.apply(gray_frame)

        # Refine the mask to remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)  # Remove small noise
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_DILATE, kernel)  # Fill gaps in detected regions

        # Find contours (regions of motion)
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # List of centers of current moving objects
        current_centers = []

        # Iterate through the contours to find moving objects
        for contour in contours:
            if cv2.contourArea(contour) >= MIN_CONTOUR_AREA and cv2.contourArea(contour) <= MAX_CONTOUR_AREA:
                # Get the bounding box for the contour
                x, y, w, h = cv2.boundingRect(contour)
                # Filter out objects with extreme aspect ratios (avoid elongated objects like chains)
                aspect_ratio = w / float(h)
                if aspect_ratio < 0.2 or aspect_ratio > 5.0:  # You can adjust these limits
                    continue

                # Calculate the center of the object
                center = (x + w // 2, y + h // 2)

                # Add the center to the list of current centers
                current_centers.append(center)

                # Get the current time for real-time speed calculation
                curr_time = cv2.getTickCount() / cv2.getTickFrequency()  # Time in seconds

                # Match the current center to the closest previous center based on proximity
                matched = False
                for object_id, prev_center in prev_centers.items():
                    dist = np.linalg.norm(np.array(center) - np.array(prev_center))
                    if dist < 50:  # If the distance between centers is small, consider it the same object
                        prev_time = prev_times[object_id]

                        # Calculate speed
                        speed = calculate_speed(center, prev_center, curr_time, prev_time, scale_factor)

                        # Only display speed if the object is moving fast enough
                        if speed > 0.1:  # Minimum speed threshold to filter out slow objects
                            print(f"Speed for object {object_id}: {speed:.2f} m/s")

                            # Display the speed outside the bounding box (top-left corner)
                            cv2.putText(frame, f"Speed: {speed:.2f} m/s", (x + 10, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                            # Draw the bounding box around the moving object in green
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                        # Update the previous center and time for this object
                        prev_centers[object_id] = np.array(center)
                        prev_times[object_id] = curr_time
                        matched = True
                        break

                # If the object is not matched, treat it as a new object
                if not matched:
                    object_id = len(prev_centers) + 1  # Assign a new ID for the object
                    prev_centers[object_id] = np.array(center)
                    prev_times[object_id] = curr_time
                    print(f"New object detected with ID {object_id}")

                # Draw the bounding box around the object in red (always visible)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red for all objects

        # Display the frame with the bounding boxes and speed information
        cv2.imshow('Speed Analysis Feed', frame)

       

        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting...")
            break

    cap.release()
    cv2.destroyAllWindows()

# Call the function to start the webcam feed and analysis
generate_speed_analysis_feed()

from flask import Flask, request, jsonify,render_template, Response
import cv2
import numpy as np
from heatmap import generate_heatmap

color = '#FFFFFF'
app = Flask(__name__, static_folder='../static', template_folder='../templates')



# Load class names
with open("../models/object_tracking/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]




# Initialize the webcam
cap = cv2.VideoCapture(0)

# Background subtractor for motion detection
fgbg = cv2.createBackgroundSubtractorMOG2()

# Minimum contour area to filter small objects (noise)
MIN_CONTOUR_AREA = 700  # Adjust as needed

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/update_color', methods=['POST'])
def update_color():
    global color
    print("Request Method:", request.method)
    print("Request URL:", request.url)
    print("Request Headers:", request.headers)

    # Checking if there is data in the request
    if not request.data:
        print("No data received")
        return jsonify({'error': 'No data received'}), 400

    try:
        # Attempt to decode JSON data from the request
        data = request.get_json()

        # If no valid JSON, handle it
        if not data:
            print("Invalid JSON received")
            return jsonify({'error': 'Invalid JSON received'}), 400

        # Extract the 'color' from the received JSON
        color = data.get('color')
        if color:
            print(f"Color selected: {color}")
            return jsonify({'color': color}), 200  # Success response
        else:
            print("No color found in data")
            return jsonify({'error': 'No color selected'}), 400  # Error response for no color
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        return jsonify({'error': f'Error processing request: {str(e)}'}), 500  # Generic error handler


def hex_to_hsv(hex_color):
    # Convert hex to RGB
    global rgb
    rgb = np.array([int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)])
    
    # Convert RGB to HSV
    hsv = cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2HSV)[0][0]
    return hsv


def generate_speed_analysis_feed():
    prev_gray = None

    # Constants
    PIXEL_TO_METER = 0.1  # Conversion factor (adjust based on setup)
    MAX_SPEED = 30  # Maximum speed limit in m/s (adjust this)
    BAR_MAX_WIDTH = 640  # Maximum progress bar width (frame width)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize the frame for consistency (optional)
        frame_resized = cv2.resize(frame, (640, 480))

        # Convert the entire frame to grayscale
        frame_gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        frame_gray_colored = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)  # Convert back to BGR for drawing

        # Convert the current frame to grayscale for optical flow
        gray_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

        # Calculate optical flow (if previous frame exists)
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)

            # Calculate displacement magnitude for each pixel
            displacement = np.linalg.norm(flow, axis=2)
            avg_speed = np.mean(displacement)  # Average speed of the entire frame in pixels per frame

            # Convert to pixels per second (assuming 30 FPS)
            avg_speed_per_second = avg_speed * 30  # Adjust according to your frame rate

            # Convert pixels per second to meters per second
            avg_speed_mps = avg_speed_per_second * PIXEL_TO_METER

            # If speed is below 4 m/s, set it to 0
            if avg_speed_mps < 4:
                avg_speed_mps = 0

            # Scale progress bar width relative to max speed
            progress_width = int((avg_speed_mps / MAX_SPEED) * BAR_MAX_WIDTH)
            progress_width = min(progress_width, BAR_MAX_WIDTH)  # Cap at maximum width

            # Frame dimensions
            frame_height, frame_width = frame_resized.shape[:2]

            # Progress bar dimensions and position
            progress_bar_height = 20
            progress_x1 = 0  # Start from the left (center-left horizontally)
            progress_x2 = progress_width
            progress_y1 = frame_height // 2 - (progress_bar_height // 2)
            progress_y2 = progress_y1 + progress_bar_height

            # Color the progress bar (green if moving, red if stationary)
            if avg_speed_mps == 0:
                progress_color = (0, 0, 255)  # Red for no motion
            else:
                progress_color = (0, 255, 0)  # Green for motion

            # Draw the progress bar at the center-left of the frame
            cv2.rectangle(frame_gray_colored, (progress_x1, progress_y1), (progress_x2, progress_y2), progress_color, -1)

            # Text to display the speed
            text = f"Avg Speed: {avg_speed_mps:.2f} m/s"  # Display in m/s
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            thickness = 2

            # Calculate text size and position it just above the progress bar
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            text_x = 10  # Small margin from the left
            text_y = progress_y1 - 10  # 10 pixels above the progress bar

            # Draw the speed text
            cv2.putText(frame_gray_colored, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)

        # Update previous frame and grayscale for the next iteration
        prev_gray = gray_frame

        # Encode the frame as JPEG
        ret, jpeg = cv2.imencode('.jpg', frame_gray_colored)

        if ret:
            # Yield the frame as a byte stream for the web app
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')




def apply_gamma_correction(image, gamma=1.5):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)

def generate_motion_detection_feed():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize the frame to improve performance (optional)
        frame_resized = cv2.resize(frame, (640, 480))
       


        # Convert the frame to grayscale for motion detection
        gray_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

        # Apply background subtraction to detect motion
        fgmask = fgbg.apply(gray_frame)

        # Refine the mask to remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)  # Remove small noise
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_DILATE, kernel)  # Fill gaps in detected regions

        # Find contours (regions of motion)
        motion_contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Initialize a mask to determine which objects have motion
        motion_mask = np.zeros_like(gray_frame)

        # Draw motion contours on the mask
        for motion_contour in motion_contours:
            if cv2.contourArea(motion_contour) >= MIN_CONTOUR_AREA:
                cv2.drawContours(motion_mask, [motion_contour], -1, 255, thickness=cv2.FILLED)

        # Now apply object detection only to the regions with motion
        # For object detection (without relying on motion)
        gray_blur = cv2.GaussianBlur(gray_frame, (15, 15), 0)
        _, object_thresh = cv2.threshold(gray_blur, 50, 255, cv2.THRESH_BINARY_INV)
        object_contours, _ = cv2.findContours(object_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for object_contour in object_contours:
            if cv2.contourArea(object_contour) < MIN_CONTOUR_AREA:
                continue

            # Get bounding box for each object
            x, y, w, h = cv2.boundingRect(object_contour)

            # Extract the region of interest (ROI) for motion
            roi_motion = motion_mask[y:y + h, x:x + w]

            # Check if there's motion in the ROI
            motion_detected = np.any(roi_motion > 0)

            # Set bounding box color: Green (motion) or Red (stationary)
            color = (0, 255, 0) if motion_detected else (0, 0, 255)

            # Draw the bounding box
            cv2.rectangle(frame_resized, (x, y), (x + w, y + h), color, 2)

        # Encode the frame as JPEG
        ret, jpeg = cv2.imencode('.jpg', frame_resized)

        if ret:
            # Yield the frame as a byte stream for the web app
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')


def generate_color_detection_feed():
    global color
    
    # Ensure color is properly set before using it
    if not color:
        color = "#FFFFFF"  # Default to white if no color is selected

    # Convert hex color to HSV
    selected_color_hsv = hex_to_hsv(color)  # This dynamically updates based on the selected color
    
    print(f"Selected color (HSV) in feed: {selected_color_hsv}")  # Debug print for selected color

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to HSV for color detection
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

       # Define the tolerance range for detecting the color black
        lower_bound = np.array([0, 0, 0])  # Lower bound for black in HSV
        upper_bound = np.array([180, 255, 50])  # Upper bound for black in HSV  # Upper bound (with tolerance)

        # Debugging the bounds and mask
        print(f"Lower Bound: {lower_bound}")
        print(f"Upper Bound: {upper_bound}")
        
        # Create a mask to detect the selected color
        mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)

        # Debug print to check if mask has any white regions (color detection)
        print(f"Mask sum: {np.sum(mask)}")
       
        # Apply the mask to the frame
        color_detected = cv2.bitwise_and(frame, frame, mask=mask)

        # Encode the frame as JPEG
        ret, jpeg = cv2.imencode('.jpg', color_detected)
        if ret:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')



def generate_edge_detection_feed():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame to improve performance
        frame_resized = cv2.resize(frame, (640, 480))

        # Convert the frame to grayscale for edge detection
        gray_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

        # Apply Canny edge detection
        edges = cv2.Canny(gray_frame, 100, 200)

        # Invert the edges to make edges white and the rest black
        edges_inverted = cv2.bitwise_not(edges)

        # Convert the inverted edges to BGR for consistency (optional)
        edges_colored = cv2.cvtColor(edges_inverted, cv2.COLOR_GRAY2BGR)

        # Encode the frame as JPEG
        ret, jpeg = cv2.imencode('.jpg', edges_colored)

        if ret:
            # Yield the frame as a byte stream for the web app
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')



def generate_heatmap_feed():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Generate the heatmap (only the heatmap output is needed)
        _, heatmap_colored = generate_heatmap(frame, fgbg, MIN_CONTOUR_AREA)

        # Resize the heatmap to match the desired display size if necessary
        if frame.shape[:2] != heatmap_colored.shape[:2]:
            heatmap_colored = cv2.resize(heatmap_colored, (frame.shape[1], frame.shape[0]))

        # Encode the heatmap frame as JPEG
        ret, jpeg = cv2.imencode('.jpg', heatmap_colored)

        if ret:
            # Yield the heatmap frame as a byte stream for the web app
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')


@app.route('/speed_analysis_feed')
def speed_analysis_feed():
    # Generate or get the speed analysis feed (similar to how you generate object classification feed)
    return Response(generate_speed_analysis_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/motion_detection_feed')
def motion_detection_feed():
    # Logic to serve the Motion Detection feed
    return Response(generate_motion_detection_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')



@app.route('/heatmap_feed')
def heatmap_feed():
    # Logic to serve the Heatmap feed
    return Response(generate_heatmap_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/color_detection_feed')
def color_detection_feed():
    return Response(generate_color_detection_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/edge_detection_feed')
def edge_detection_feed():
    return Response(generate_edge_detection_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)

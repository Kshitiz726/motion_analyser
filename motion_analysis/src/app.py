from flask import Flask, request, jsonify,render_template, Response
import cv2
import numpy as np
from heatmap import generate_heatmap
import colorsys
import sys
sys.path.append(r'D:/Minor')
from mdl import get_model
color = '#FFFFFF'
app = Flask(__name__, static_folder='../static', template_folder='../templates')




COLOR_RANGES = {
    "black": {"lower": [0, 0, 0], "upper": [180, 255, 50]},
    "white": {"lower": [0, 0, 200], "upper": [180, 50, 255]},
    "red": {"lower": [0, 120, 70], "upper": [10, 255, 255]},
    "green": {"lower": [35, 100, 100], "upper": [85, 255, 255]},
    "blue": {"lower": [100, 150, 50], "upper": [140, 255, 255]},
    "orange": {"lower": [5, 150, 150], "upper": [15, 255, 255]},
    "yellow": {"lower": [20, 100, 100], "upper": [40, 255, 255]},
    "pink": {"lower": [160, 100, 100], "upper": [180, 255, 255]},
    "purple": {"lower": [140, 100, 100], "upper": [160, 255, 255]},
    "brown": {"lower": [10, 100, 20], "upper": [20, 255, 200]},
    "gray": {"lower": [0, 0, 50], "upper": [180, 50, 150]},
    "cyan": {"lower": [85, 100, 100], "upper": [95, 255, 255]},
    
}

selected_color = "black"
lower_bound = np.array(COLOR_RANGES[selected_color]["lower"])
upper_bound = np.array(COLOR_RANGES[selected_color]["upper"])





cap = cv2.VideoCapture(0)


fgbg = cv2.createBackgroundSubtractorMOG2()


MIN_CONTOUR_AREA = 700  



previous_positions = {}
motion_timers = {}
MOTION_THRESHOLD = 7  
MOTION_DURATION = 10   

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/update_color', methods=['POST'])
def update_color():
    global lower_bound, upper_bound, selected_color
    
   
    data = request.get_json()
    color = data.get('color', 'black')  
    
   
    if color in COLOR_RANGES:
        selected_color = color
        lower_bound = np.array(COLOR_RANGES[color]["lower"])
        upper_bound = np.array(COLOR_RANGES[color]["upper"])
        return jsonify({"color": color, "status": "updated"}), 200
    else:
        return jsonify({"error": "Invalid color"}), 400





def generate_speed_analysis_feed():
    prev_gray = None

    # Constants
    PIXEL_TO_METER = 0.1
    MAX_SPEED = 30  
    BAR_MAX_WIDTH = 640  

    while True:
        ret, frame = cap.read()
        if not ret:
            break

       
        frame_resized = cv2.resize(frame, (640, 480))

        # entire frame to grayscale
        frame_gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        frame_gray_colored = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)  # Convert back to BGR for drawing

        #  current frame to grayscale for optical flow
        gray_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

        # Calculate optical flow 
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)

            # Calculate displacement magnitude for each pixel
            displacement = np.linalg.norm(flow, axis=2)
            avg_speed = np.mean(displacement)  

          
            fps = cap.get(cv2.CAP_PROP_FPS)
            avg_speed_per_second = avg_speed * fps  

            
            avg_speed_mps = avg_speed_per_second * PIXEL_TO_METER

            
            if avg_speed_mps < 4:
                avg_speed_mps = 0

            
            progress_width = int((avg_speed_mps / MAX_SPEED) * BAR_MAX_WIDTH)
            progress_width = min(progress_width, BAR_MAX_WIDTH)  # Cap at maximum width

            # Frame dimensions
            frame_height, frame_width = frame_resized.shape[:2]

            
            progress_bar_height = 20
            progress_x1 = 0  
            progress_x2 = progress_width
            progress_y1 = frame_height // 2 - (progress_bar_height // 2)
            progress_y2 = progress_y1 + progress_bar_height

           
            if avg_speed_mps == 0:
                progress_color = (0, 0, 255)  
            else:
                progress_color = (0, 255, 0)  

           
            cv2.rectangle(frame_gray_colored, (progress_x1, progress_y1), (progress_x2, progress_y2), progress_color, -1)

            
            text = f"Avg Speed: {avg_speed_mps:.2f} m/s"  
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            thickness = 2

          
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            text_x = 10  
            text_y = progress_y1 - 10  

            
            cv2.putText(frame_gray_colored, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)

        
        prev_gray = gray_frame

        
        ret, jpeg = cv2.imencode('.jpg', frame_gray_colored)

        if ret:
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')




def apply_gamma_correction(image, gamma=1.5):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)

def generate_motion_detection_feed():
    
   
    model = get_model()  
    
    previous_positions = {}
    motion_timers = {}
    
    
    MOTION_THRESHOLD = 9 
    MOTION_DURATION = 9   

    while True:
        ret, frame = cap.read()
        if not ret:
            break

       
        results = model.track(frame, persist=True, verbose=False)

        current_positions = {}

        for result in results:
            for track in result.boxes.data:
                x1, y1, x2, y2, track_id, conf = map(int, track[:6])
                centroid_x = int((x1 + x2) / 2)
                centroid_y = int((y1 + y2) / 2)

               
                current_positions[track_id] = (centroid_x, centroid_y, x1, y1, x2, y2)

        # Draw bounding boxes
        for track_id, (cx, cy, x1, y1, x2, y2) in current_positions.items():
            color = (0, 0, 255)  

            # Check if object moved
            if track_id in previous_positions:
                prev_cx, prev_cy = previous_positions[track_id][:2]
                distance = np.sqrt((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2)

                if distance > MOTION_THRESHOLD:
                    motion_timers[track_id] = MOTION_DURATION  

            # Keep object green for a few frames after detecting motion
            if track_id in motion_timers and motion_timers[track_id] > 0:
                color = (0, 255, 0)  
                motion_timers[track_id] -= 1  

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

        # Update previous positions
        previous_positions = current_positions.copy()

        # Encode the frame as JPEG
        ret, jpeg = cv2.imencode('.jpg', frame)

        if ret:
           
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')


def generate_color_detection_feed():
    global lower_bound, upper_bound
    
   
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
       
        mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)
        color_detected = cv2.bitwise_and(frame, frame, mask=mask)
        
        
        ret, jpeg = cv2.imencode('.jpg', color_detected)
        if ret:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')







def generate_edge_detection_feed():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

       
        frame_resized = cv2.resize(frame, (640, 480))

        
        gray_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

        
        edges = cv2.Canny(gray_frame, 100, 200)

       
        edges_inverted = cv2.bitwise_not(edges)

        
        edges_colored = cv2.cvtColor(edges_inverted, cv2.COLOR_GRAY2BGR)

        
        ret, jpeg = cv2.imencode('.jpg', edges_colored)

        if ret:
            
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
    return Response(generate_motion_detection_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')



@app.route('/heatmap_feed')
def heatmap_feed():
    
    return Response(generate_heatmap_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/color_detection_feed')
def color_detection_feed():
    return Response(generate_color_detection_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/edge_detection_feed')
def edge_detection_feed():
    return Response(generate_edge_detection_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)

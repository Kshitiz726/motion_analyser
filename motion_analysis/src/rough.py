import cv2
import numpy as np

# Initialize color variable and default to black
color = "#000000"  # Default color (black)
frame_shape = None  # To store the shape of the frame globally

def hex_to_hsv(hex_color):
    """Converts hex color to HSV format."""
    rgb = np.array([int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)])
    hsv = cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2HSV)[0][0]
    return hsv

def mouse_callback(event, x, y, flags, param):
    """Handles mouse events to select colors from the palette."""
    global color
    frame_shape, color_list = param  # Unpack the frame shape and color list from param
    
    if event == cv2.EVENT_LBUTTONDOWN:
        # Check if the click is within the palette region
        color_index = x // 50  # Calculate which color was clicked
        if 0 <= color_index < len(color_list):
            color = color_list[color_index]  # Update the selected color
            print(f"Selected color: {color}")

def generate_color_detection_feed(color):
    """Generates real-time color detection feed with a color palette."""
    global frame_shape
    selected_color_hsv = hex_to_hsv(color)
    
    print(f"Selected color (HSV): {selected_color_hsv}")  # Debug print to check HSV conversion
    
    cap = cv2.VideoCapture(0)  # Access the webcam
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Update frame_shape with current frame dimensions
        frame_shape = frame.shape

        # Convert to HSV for color detection
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Adjust the color range based on the selected color's HSV
        hue_tolerance = 10  # Adjust tolerance for hue range
        sat_tolerance = 30  # Adjust tolerance for saturation
        val_tolerance = 90  # Adjust tolerance for value

        # Handling colors like white which have low saturation
        if selected_color_hsv[1] < 50:  # If saturation is low (indicating white or light colors)
            sat_tolerance = 255  # Allow higher tolerance for saturation

        # Calculate lower and upper bounds for the selected color
        lower_bound = np.array([selected_color_hsv[0] - hue_tolerance, 
                                selected_color_hsv[1] - sat_tolerance, 
                                selected_color_hsv[2] - val_tolerance])  # Lower bound
        upper_bound = np.array([selected_color_hsv[0] + hue_tolerance, 
                                selected_color_hsv[1] + sat_tolerance, 
                                selected_color_hsv[2] + val_tolerance])  # Upper bound

        # Debug prints to check HSV range and mask
       

        # Create a mask to detect the selected color
        mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)

        # Debug: Show the mask to understand what is detected
        cv2.imshow('Mask', mask)

        # Apply the mask to the frame
        color_detected = cv2.bitwise_and(frame, frame, mask=mask)

        # Display the color detected feed
        cv2.imshow('Color Detected', color_detected)

        # Display the current selected color on the frame
        cv2.putText(frame, f"Selected color: {color}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('Frame with Color Info', frame)

        # Ask for new color input every frame if 'n' is pressed
        if cv2.waitKey(1) & 0xFF == ord('n'):
            color_input = input("Enter a new color (HEX format e.g., #FF5733): ")
            color = color_input.strip()
            selected_color_hsv = hex_to_hsv(color)
            print(f"New color selected: {color} (HSV: {selected_color_hsv})")

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Run the color detection feed with default color (black)
generate_color_detection_feed(color)

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from cnn_model import create_cnn_model



model = create_cnn_model() 
model.load_state_dict(torch.load('../model/motion_detection_model.pt'))
model.eval()

cap = cv2.VideoCapture(0)


fgbg = cv2.createBackgroundSubtractorMOG2()

MIN_CONTOUR_AREA = 700   

prev_frame = None


preprocess = transforms.Compose([
    transforms.ToPILImage(),          
    transforms.Resize((224, 224)),    
    transforms.ToTensor(),           
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize based on pre-trained model
])

def detect_motion(frame):
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_tensor = preprocess(frame_rgb)
    frame_tensor = frame_tensor.unsqueeze(0) 
    
    with torch.no_grad():
        output = model(frame_tensor)  
    
    #for binary classiication
    motion = torch.sigmoid(output) > 0.5  
    
    return motion.item()  


def calculate_optical_flow(prev_gray, curr_gray):
    
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return flow


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_resized = cv2.resize(frame, (640, 480))
    gray_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

    fgmask = fgbg.apply(gray_frame) 

    #noise removal
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_DILATE, kernel)

    #find contours 
    motion_contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    motion_mask = np.zeros_like(gray_frame)

    for motion_contour in motion_contours:
        if cv2.contourArea(motion_contour) >= MIN_CONTOUR_AREA:
            cv2.drawContours(motion_mask, [motion_contour], -1, 255, thickness=cv2.FILLED)

    #is there any motion?
    motion_detected = detect_motion(frame_resized)

    if motion_detected:
        # If motion is detected by the CNN model, apply optical flow to detect the region of motion
        if prev_frame is not None:
            flow = calculate_optical_flow(prev_frame, gray_frame)
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

            # Threshold the magnitude of the flow to detect motion regions
            flow_mask = np.zeros_like(gray_frame)
            flow_mask[mag > 2] = 255 

            flow_contours, _ = cv2.findContours(flow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            #draw boxed around motion
            for flow_contour in flow_contours:
                if cv2.contourArea(flow_contour) >= MIN_CONTOUR_AREA:
                    cv2.drawContours(frame_resized, [flow_contour], -1, (0, 255, 255), 2)  # Yellow bounding box

    
    prev_frame = gray_frame

    #Display the motion detected
    cv2.imshow('Motion Detection', frame_resized)

   
    
cap.release()
cv2.destroyAllWindows()

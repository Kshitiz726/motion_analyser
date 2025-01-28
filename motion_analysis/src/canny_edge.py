import cv2
import numpy as np

# Step 1: Convert image to grayscale
def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Step 2: Gaussian Blurring
def gaussian_blur(image, kernel_size=5, sigma=1.4):
    # Create Gaussian kernel
    kernel = cv2.getGaussianKernel(kernel_size, sigma)
    kernel = kernel @ kernel.T  # Make it 2D
    return cv2.filter2D(image, -1, kernel)

# Step 3: Sobel Gradient Calculation
def sobel_gradients(image):
    # Sobel kernels for X and Y direction
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    grad_x = cv2.filter2D(image, -1, sobel_x)
    grad_y = cv2.filter2D(image, -1, sobel_y)

    # Compute the magnitude and angle of gradients
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    angle = np.arctan2(grad_y, grad_x) * 180 / np.pi  # In degrees

    return magnitude, angle

# Step 4: Non-Maximum Suppression
def non_maximum_suppression(magnitude, angle):
    height, width = magnitude.shape
    suppressed_image = np.zeros_like(magnitude)

    for i in range(1, height-1):
        for j in range(1, width-1):
            angle_value = angle[i, j]
            # 0 to 180 degree range
            if (0 <= angle_value < 22.5) or (157.5 <= angle_value <= 180):
                neighbor1 = magnitude[i, j+1]
                neighbor2 = magnitude[i, j-1]
            elif (22.5 <= angle_value < 67.5):
                neighbor1 = magnitude[i+1, j-1]
                neighbor2 = magnitude[i-1, j+1]
            elif (67.5 <= angle_value < 112.5):
                neighbor1 = magnitude[i+1, j]
                neighbor2 = magnitude[i-1, j]
            else:
                neighbor1 = magnitude[i-1, j-1]
                neighbor2 = magnitude[i+1, j+1]

            # If the current pixel is not a local maximum, suppress it
            if magnitude[i, j] >= neighbor1 and magnitude[i, j] >= neighbor2:
                suppressed_image[i, j] = magnitude[i, j]
            else:
                suppressed_image[i, j] = 0

    return suppressed_image

# Step 5: Hysteresis Thresholding
def hysteresis_thresholding(image, low_threshold, high_threshold):
    output_image = np.zeros_like(image)
    strong_edges = np.where(image >= high_threshold)
    weak_edges = np.where((image >= low_threshold) & (image < high_threshold))

    output_image[strong_edges] = 255  # Strong edges
    output_image[weak_edges] = 75    # Weak edges

    # Final pass: connect weak edges to strong ones
    for i in range(1, image.shape[0]-1):
        for j in range(1, image.shape[1]-1):
            if output_image[i, j] == 75:
                # Check neighbors for strong edge
                if np.any(output_image[i-1:i+2, j-1:j+2] == 255):
                    output_image[i, j] = 255

    return output_image

# Canny Edge Detection from Scratch
def canny_edge_detection(image, low_threshold=50, high_threshold=150):
    # Step 1: Convert to grayscale
    gray_image = grayscale(image)

    # Step 2: Gaussian blur to smooth the image
    blurred_image = gaussian_blur(gray_image)

    # Step 3: Compute gradients using Sobel
    magnitude, angle = sobel_gradients(blurred_image)

    # Step 4: Apply non-maximum suppression
    suppressed_image = non_maximum_suppression(magnitude, angle)

    # Step 5: Apply hysteresis thresholding
    final_edges = hysteresis_thresholding(suppressed_image, low_threshold, high_threshold)

    return final_edges

# Real-time Edge Detection with webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform Canny edge detection from scratch
    edges = canny_edge_detection(frame)

    # Convert edges to uint8 for proper display
    edges_display = np.uint8(edges)

    # Show the original frame and the edges
    cv2.imshow('Original Frame', frame)
    cv2.imshow('Canny Edge Detection', edges_display)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture object and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()

import cv2
import numpy as np
import os

def create_output_folder():
    """Create output folder if it doesn't exist"""
    if not os.path.exists('high_contrast_images'):
        os.makedirs('high_contrast_images')

def manual_thresholding(image):
    """Apply manual thresholding to enhance contrast"""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Manually chosen threshold value (adjust as needed)
    _, thresholded = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    return thresholded

def manual_linear_transformation(image):
    """Apply manual linear transformation to enhance contrast"""
    # Manually chosen parameters (adjust as needed)
    a = 1.5  # Contrast control
    b = -50  # Brightness control
    
    # Apply linear transformation: T(r) = a*r + b
    transformed = np.clip(a * image.astype(np.float32) + b, 0, 255).astype(np.uint8)
    
    return transformed

def auto_linear_transformation(image):
    """Apply automatic linear transformation based on histogram"""
    # Convert to grayscale for histogram analysis
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calculate histogram percentiles to determine parameters
    low_percentile = np.percentile(gray, 5)   # 5th percentile
    high_percentile = np.percentile(gray, 95)  # 95th percentile
    
    # Calculate parameters to stretch the histogram
    a = 255.0 / (high_percentile - low_percentile) if high_percentile > low_percentile else 1.0
    b = -a * low_percentile
    
    # Apply linear transformation to each channel
    result = np.zeros_like(image, dtype=np.float32)
    for i in range(3):
        result[:, :, i] = a * image[:, :, i].astype(np.float32) + b
    
    # Clip values to valid range and convert back to uint8
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    return result

def histogram_equalization(image):
    """Apply histogram equalization using HSV color space"""
    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Apply histogram equalization to the Value channel only
    hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
    
    # Convert back to BGR
    equalized = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return equalized

def main():
    # Create output folder
    create_output_folder()
    
    # Load the input image
    input_path = 'scanned_images/warped_linear.jpg'
    if not os.path.exists(input_path):
        print(f"Error: Input image not found at {input_path}")
        return
    
    image = cv2.imread(input_path)
    if image is None:
        print("Error: Failed to load image")
        return
    
    # Apply contrast enhancement techniques
    thresholded = manual_thresholding(image)
    manual_linear = manual_linear_transformation(image)
    auto_linear = auto_linear_transformation(image)
    equalized = histogram_equalization(image)
    
    # Save the results
    cv2.imshow("Thresholded", thresholded)
    cv2.imshow("Manual Linear Transformation", manual_linear)
    cv2.imshow("Auto Linear Transformation", auto_linear)
    cv2.imshow("Histogram Equalization", equalized)

    cv2.imwrite('high_contrast_images/thresholded.jpg', thresholded)
    cv2.imwrite('high_contrast_images/manual_linear.jpg', manual_linear)
    cv2.imwrite('high_contrast_images/auto_linear.jpg', auto_linear)
    cv2.imwrite('high_contrast_images/histogram_equalized.jpg', equalized)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("Contrast enhancement completed. Results saved in 'high_contrast_images' folder.")

if __name__ == "__main__":
    main()
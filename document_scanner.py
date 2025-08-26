import cv2
import numpy as np
import os

def select_points(image, num_points=4):
    """
    Allow user to select points on the image using OpenCV
    Returns: List of selected points
    """
    points = []
    image_copy = image.copy()
    
    def click_event(event, x, y, flags, params):
        nonlocal points, image_copy
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            # Make the point larger and more visible
            cv2.circle(image_copy, (x, y), 12, (0, 255, 0), -1)  # Larger green circle
            cv2.circle(image_copy, (x, y), 15, (0, 0, 255), 2)   # Red outline
            cv2.putText(image_copy, str(len(points)), (x+20, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)  # Larger white text
            cv2.imshow("Select 4 Corners (Press 'q' when done)", image_copy)
    
    cv2.imshow("Select 4 Corners (Press 'q' when done)", image_copy)
    cv2.setMouseCallback("Select 4 Corners (Press 'q' when done)", click_event)
    
    print("Click on the four corners of the document in order:")
    print("1. Top-left\n2. Top-right\n3. Bottom-right\n4. Bottom-left")
    print("Press 'q' when you have selected all 4 points")
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or len(points) >= num_points:
            break
    
    cv2.destroyAllWindows()
    return points[:num_points]

def order_points(pts):
    """
    Order points in clockwise order starting from top-left
    """
    rect = np.zeros((4, 2), dtype="float32")
    
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    
    return rect

def compute_homography(src_points, dst_points):
    """
    Compute homography matrix from source to destination points
    """
    # Create matrix A
    A = []
    for i in range(4):
        x, y = src_points[i]
        u, v = dst_points[i]
        A.append([x, y, 1, 0, 0, 0, -u*x, -u*y])
        A.append([0, 0, 0, x, y, 1, -v*x, -v*y])
    
    A = np.array(A)
    
    # Create vector b
    b = dst_points.flatten()
    
    # Solve for homography parameters
    h = np.linalg.lstsq(A, b, rcond=None)[0]
    
    # Reshape into 3x3 matrix
    H = np.array([[h[0], h[1], h[2]],
                  [h[3], h[4], h[5]],
                  [h[6], h[7], 1]])
    
    return H

def apply_homography(H, x, y):
    """
    Apply homography transformation to a point
    """
    point = np.array([x, y, 1])
    transformed = H @ point
    transformed /= transformed[2]  # Normalize by homogeneous coordinate
    return transformed[0], transformed[1]

def manual_warp_perspective(image, H, output_size, interpolation='nearest'):
    """
    Apply perspective transformation manually
    """
    h, w = output_size
    warped = np.zeros((h, w), dtype=image.dtype)
    
    # Compute inverse homography for backward mapping
    H_inv = np.linalg.inv(H)
    
    for y_out in range(h):
        for x_out in range(w):
            # Apply inverse homography to find source coordinates
            x_src, y_src = apply_homography(H_inv, x_out, y_out)
            
            if interpolation == 'nearest':
                # Nearest neighbor interpolation
                x_nearest = int(round(x_src))
                y_nearest = int(round(y_src))
                
                # Check if within source image bounds
                if 0 <= x_nearest < image.shape[1] and 0 <= y_nearest < image.shape[0]:
                    warped[y_out, x_out] = image[y_nearest, x_nearest]
            
            elif interpolation == 'bilinear':
                # Bilinear interpolation
                x_floor = int(np.floor(x_src))
                y_floor = int(np.floor(y_src))
                x_ceil = min(x_floor + 1, image.shape[1] - 1)
                y_ceil = min(y_floor + 1, image.shape[0] - 1)
                
                # Check if within source image bounds
                if 0 <= x_floor < image.shape[1] and 0 <= y_floor < image.shape[0]:
                    # Calculate interpolation weights
                    dx = x_src - x_floor
                    dy = y_src - y_floor
                    
                    # Get the four neighboring pixels
                    top_left = image[y_floor, x_floor]
                    top_right = image[y_floor, x_ceil]
                    bottom_left = image[y_ceil, x_floor]
                    bottom_right = image[y_ceil, x_ceil]
                    
                    # Interpolate
                    top = (1 - dx) * top_left + dx * top_right
                    bottom = (1 - dx) * bottom_left + dx * bottom_right
                    warped[y_out, x_out] = (1 - dy) * top + dy * bottom
    
    return warped

def perspective_transform(image, pts):
    """
    Apply perspective transformation to get a top-down view
    Returns images with nearest neighbor and bilinear interpolation
    """
    # Order the points
    rect = order_points(np.array(pts, dtype="float32"))
    (tl, tr, br, bl) = rect
    
    # Compute the width of the new image
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    # Compute the height of the new image
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    # Destination points
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    
    # Compute the homography matrix
    H = compute_homography(rect, dst)
    
    # Apply the transformation with different interpolation methods
    warped_nearest = manual_warp_perspective(image, H, (maxHeight, maxWidth), interpolation='nearest')
    warped_linear = manual_warp_perspective(image, H, (maxHeight, maxWidth), interpolation='bilinear')
    
    return warped_nearest, warped_linear

def display_comparison(original, points, warped_nearest, warped_linear):
    """
    Display comparison using OpenCV windows
    """
    # Draw points on original image (make them more visible)
    original_with_points = original.copy()
    for i, point in enumerate(points):
        cv2.circle(original_with_points, point, 15, (0, 255, 0), -1)  # Larger green circle
        cv2.circle(original_with_points, point, 18, (0, 0, 255), 3)   # Red outline
        cv2.putText(original_with_points, str(i+1), (point[0]+25, point[1]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4)  # Larger white text
    
    # Create zoomed regions from the CENTER of the document
    h, w = warped_nearest.shape
    center_h, center_w = h // 2, w // 2
    zoom_size = min(h, w) // 6  # Size of zoomed area
    
    # Define zoom region around center
    zoom_region = (
        slice(center_h - zoom_size, center_h + zoom_size),
        slice(center_w - zoom_size, center_w + zoom_size)
    )
    
    # Extract zoomed regions
    zoom_nearest = warped_nearest[zoom_region]
    zoom_linear = warped_linear[zoom_region]
    
    # Resize for better display (make zoom larger)
    zoom_nearest = cv2.resize(zoom_nearest, (400, 400), interpolation=cv2.INTER_NEAREST)
    zoom_linear = cv2.resize(zoom_linear, (400, 400), interpolation=cv2.INTER_NEAREST)
    
    # Add text labels to zoom images
    cv2.putText(zoom_nearest, "Nearest Neighbor", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(zoom_linear, "Bilinear", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Display images
    cv2.imshow("Original Image with Points", original_with_points)
    cv2.imshow("Nearest Neighbor Interpolation", warped_nearest)
    cv2.imshow("Bilinear Interpolation", warped_linear)
    cv2.imshow("Zoom - Center (Nearest Neighbor)", zoom_nearest)
    cv2.imshow("Zoom - Center (Bilinear)", zoom_linear)
    
    print("Press any key to close windows...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def save_results(original, points, warped_nearest, warped_linear):
    """
    Save all results to files
    """
    # Draw points on original image (make them more visible)
    original_with_points = original.copy()
    for i, point in enumerate(points):
        cv2.circle(original_with_points, point, 15, (0, 255, 0), -1)  # Larger green circle
        cv2.circle(original_with_points, point, 18, (0, 0, 255), 3)   # Red outline
        cv2.putText(original_with_points, str(i+1), (point[0]+25, point[1]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4)  # Larger white text
    
    # Create zoomed regions from the CENTER of the document
    h, w = warped_nearest.shape
    center_h, center_w = h // 2, w // 2
    zoom_size = min(h, w) // 6  # Size of zoomed area
    
    # Define zoom region around center
    zoom_region = (
        slice(center_h - zoom_size, center_h + zoom_size),
        slice(center_w - zoom_size, center_w + zoom_size)
    )
    
    # Extract zoomed regions
    zoom_nearest = warped_nearest[zoom_region]
    zoom_linear = warped_linear[zoom_region]
    
    # Resize zoomed regions
    zoom_nearest = cv2.resize(zoom_nearest, (400, 400), interpolation=cv2.INTER_NEAREST)
    zoom_linear = cv2.resize(zoom_linear, (400, 400), interpolation=cv2.INTER_NEAREST)
    
    # Add text labels to zoom images
    cv2.putText(zoom_nearest, "Nearest Neighbor", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(zoom_linear, "Bilinear", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Save images
    cv2.imwrite("original_with_points.jpg", original_with_points)
    cv2.imwrite("warped_nearest.jpg", warped_nearest)
    cv2.imwrite("warped_linear.jpg", warped_linear)
    cv2.imwrite("zoom_center_nearest.jpg", zoom_nearest)
    cv2.imwrite("zoom_center_linear.jpg", zoom_linear)
    
    print("Results saved as:")
    print("- original_with_points.jpg")
    print("- warped_nearest.jpg")
    print("- warped_linear.jpg")
    print("- zoom_center_nearest.jpg")
    print("- zoom_center_linear.jpg")
    
def main():
    # Load the image
    image_path = "document.jpg"  # Replace with your image path
    
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found!")
        print("Please make sure your image is in the same folder and update the image_path variable.")
        return
    
    image = cv2.imread(image_path)
    
    if image is None:
        print("Error: Could not load image")
        return
    
    # Convert to intensity image by averaging the first 3 components
    if len(image.shape) == 3:
        intensity_image = np.mean(image[:, :, :3], axis=2).astype(np.uint8)
    else:
        intensity_image = image
    
    # Select the four corners of the document
    points = select_points(image)
    
    if len(points) != 4:
        print("Error: Exactly 4 points need to be selected")
        return
    
    # Apply perspective transformation with different interpolation methods
    warped_nearest, warped_linear = perspective_transform(intensity_image, points)
    
    # Display results
    display_comparison(image, points, warped_nearest, warped_linear)
    
    # Save results
    save_results(image, points, warped_nearest, warped_linear)

if __name__ == "__main__":
    main()
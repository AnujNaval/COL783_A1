import cv2
import numpy as np
import os

def create_folders():
    """Create necessary folders if they don't exist"""
    if not os.path.exists('watermarked_images'):
        os.makedirs('watermarked_images')
    if not os.path.exists('iitd_logos'):
        os.makedirs('iitd_logos')

def process_logo(logo_path, target_width):
    """Process the logo: create mask, resize, and apply transparency"""
    # Read the logo image
    logo = cv2.imread(logo_path)
    if logo is None:
        print(f"Error: Could not load logo from {logo_path}")
        return None, None, None
    
    # Convert to grayscale manually using the standard formula: 0.299*R + 0.587*G + 0.114*B
    height, width = logo.shape[:2]
    gray_logo = np.zeros((height, width), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            b = logo[y, x, 0]
            g = logo[y, x, 1]
            r = logo[y, x, 2]
            gray_value = 0.299 * r + 0.587 * g + 0.114 * b
            gray_logo[y, x] = int(gray_value)
    
    # Create mask manually (equivalent to THRESH_BINARY_INV)
    # THRESH_BINARY_INV: 0 if pixel > threshold, max_value otherwise
    mask = np.zeros((height, width), dtype=np.uint8)
    threshold_value = 200
    max_value = 255
    
    for y in range(height):
        for x in range(width):
            if gray_logo[y, x] > threshold_value:
                mask[y, x] = 0  # White background becomes black in mask
            else:
                mask[y, x] = max_value  # Logo area becomes white in mask
    
    # Calculate new dimensions while maintaining aspect ratio
    aspect_ratio = height / width
    new_height = int(target_width * aspect_ratio)
    
    # Resize the logo and mask
    resized_logo = cv2.resize(logo, (target_width, new_height), interpolation=cv2.INTER_AREA)
    resized_mask = cv2.resize(mask, (target_width, new_height), interpolation=cv2.INTER_AREA)
    
    # Create a version with transparency (4-channel image)
    logo_with_alpha = cv2.merge([resized_logo[:, :, 0], 
                                 resized_logo[:, :, 1], 
                                 resized_logo[:, :, 2], 
                                 resized_mask])
    
    return resized_logo, resized_mask, logo_with_alpha

def apply_watermark(document, logo, mask):
    """Apply watermark to document with 50% transparency"""
    # Create a copy of the document to work with
    watermarked = document.copy()
    
    # Get dimensions
    doc_h, doc_w = document.shape[:2]
    logo_h, logo_w = logo.shape[:2]
    
    # Calculate position (bottom right corner)
    y_start = doc_h - logo_h
    y_end = doc_h
    x_start = doc_w - logo_w
    x_end = doc_w
    
    # Ensure the logo fits in the document
    if y_start < 0 or x_start < 0:
        print("Logo is too large for the document")
        return document
    
    # Extract the region of interest
    roi = document[y_start:y_end, x_start:x_end]
    
    # Apply the watermark with 50% transparency only to non-white areas
    # Formula: output = (1 - α) * document + α * logo, where α = 0.5 for logo regions
    for y in range(logo_h):
        for x in range(logo_w):
            if mask[y, x] > 0:  # Non-white area in logo
                for c in range(3):  # For each color channel
                    watermarked[y_start+y, x_start+x, c] = (
                        0.5 * roi[y, x, c] + 0.5 * logo[y, x, c]
                    )
    cv2.imshow("Watermarked Document", watermarked)

    return watermarked

def main():
    # Create necessary folders
    create_folders()
    
    # Load the document image
    doc_path = 'high_contrast_images/auto_linear.jpg'
    if not os.path.exists(doc_path):
        print(f"Error: Document image not found at {doc_path}")
        return
    
    document = cv2.imread(doc_path)
    if document is None:
        print("Error: Failed to load document image")
        return
    
    # Calculate logo number based on instructions
    # For this example, I'll use logo 19 as specified
    logo_num = 19
    logo_path = f'iitd_logos/iitlogo-{logo_num}.jpg'
    
    if not os.path.exists(logo_path):
        print(f"Error: Logo image not found at {logo_path}")
        return
    
    # Calculate target width for logo (20% of document width)
    target_width = int(document.shape[1] * 0.2)
    
    # Process the logo
    logo, mask, logo_with_alpha = process_logo(logo_path, target_width)
    
    if logo is None:
        return
    
    # Apply watermark to document
    watermarked = apply_watermark(document, logo, mask)
    
    # Save results
    cv2.imwrite('watermarked_images/watermarked_document.jpg', watermarked)
    cv2.imwrite('iitd_logos/resized_logo.jpg', logo)
    cv2.imwrite('iitd_logos/logo_mask.jpg', mask)
    cv2.imwrite('iitd_logos/logo_with_alpha.png', logo_with_alpha)
    
    print("Watermarking completed. Results saved in:")
    print("- watermarked_images/watermarked_document.jpg")
    print("- iitd_logos/resized_logo.jpg")
    print("- iitd_logos/logo_mask.jpg")
    print("- iitd_logos/logo_with_alpha.png")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
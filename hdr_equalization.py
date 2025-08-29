import cv2
import numpy as np
import os

def hdr_histogram_equalization(hdr_image, target_min=0, target_max=255):
    """
    Perform histogram equalization on HDR image without quantization.
    
    Args:
        hdr_image: HDR image as numpy array (float32, values can be > 1.0)
        target_min: Minimum value of target range
        target_max: Maximum value of target range
    
    Returns:
        equalized_image: Histogram equalized image
        luminance_original: Original luminance values
        luminance_equalized: Equalized luminance values
    """
    
    # Step 1: Convert RGB to luminance using standard weights
    luminance = 0.299 * hdr_image[:,:,2] + 0.587 * hdr_image[:,:,1] + 0.114 * hdr_image[:,:,0]
    
    # Step 2: Handle edge cases (avoid log(0) issues)
    epsilon = 1e-8
    luminance = np.maximum(luminance, epsilon)
    
    # Step 3: Flatten luminance for sorting while keeping track of positions
    height, width = luminance.shape
    flat_luminance = luminance.flatten()
    
    # Step 4: Create array of (intensity, original_index) pairs
    indexed_intensities = [(flat_luminance[i], i) for i in range(len(flat_luminance))]
    
    # Step 5: Sort by intensity values
    indexed_intensities.sort(key=lambda x: x[0])
    
    # Step 6: Compute empirical CDF without histogram bins
    n_pixels = len(flat_luminance)
    equalized_flat = np.zeros_like(flat_luminance)
    
    for rank, (intensity, original_idx) in enumerate(indexed_intensities):
        cumulative_prob = rank / (n_pixels - 1)
        equalized_value = target_min + cumulative_prob * (target_max - target_min)
        equalized_flat[original_idx] = equalized_value
    
    # Step 7: Reshape back to 2D
    luminance_equalized = equalized_flat.reshape(height, width)
    
    # Step 8: Preserve color ratios while applying luminance transformation
    equalized_image = np.zeros_like(hdr_image)
    for i in range(height):
        for j in range(width):
            original_lum = luminance[i, j]
            new_lum = luminance_equalized[i, j]
            
            if original_lum > epsilon:
                scale_factor = new_lum / original_lum
                equalized_image[i, j] = hdr_image[i, j] * scale_factor
            else:
                equalized_image[i, j] = [new_lum, new_lum, new_lum]
    
    return equalized_image, luminance, luminance_equalized

def load_hdr_image(file_path):
    print(f"Attempting to load: {file_path}")
    
    try:
        hdr_img = cv2.imread(file_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
        if hdr_img is not None:
            print("✓ Loaded with cv2.imread (ANYDEPTH | COLOR)")
            return hdr_img
    except Exception as e:
        print(f"Method 1 failed: {e}")
    
    try:
        hdr_img = cv2.imread(file_path, cv2.IMREAD_ANYDEPTH)
        if hdr_img is not None:
            print("✓ Loaded with cv2.imread (ANYDEPTH only)")
            return hdr_img
    except Exception as e:
        print(f"Method 2 failed: {e}")
    
    try:
        hdr_img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        if hdr_img is not None:
            print("✓ Loaded with cv2.imread (UNCHANGED)")
            return hdr_img
    except Exception as e:
        print(f"Method 3 failed: {e}")
    
    return None

def find_hdr_files(directory):
    hdr_extensions = ['.hdr', '.exr', '.pic', '.rgbe']
    hdr_files = []
    
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist")
        return []
    
    for file in os.listdir(directory):
        if any(file.lower().endswith(ext) for ext in hdr_extensions):
            hdr_files.append(os.path.join(directory, file))
    
    return hdr_files

def load_and_process_hdr():
    target_file = "hdr_images/memorial.hdr"
    
    if not os.path.exists(target_file):
        print(f"Target file {target_file} not found.")
        print("Searching for HDR files in hdr_images directory...")
        
        hdr_files = find_hdr_files("hdr_images")
        if not hdr_files:
            print("No HDR files found in hdr_images directory")
            print("Please check that HDR files exist with extensions: .hdr, .exr, .pic, .rgbe")
            return
        
        print(f"Found HDR files: {hdr_files}")
        target_file = hdr_files[0]
        print(f"Using: {target_file}")
    
    hdr_img = load_hdr_image(target_file)
    
    if hdr_img is None:
        print(f"Error: Could not load {target_file}")
        return
    
    if len(hdr_img.shape) == 2:
        print("Converting grayscale HDR to 3-channel")
        hdr_img = cv2.cvtColor(hdr_img, cv2.COLOR_GRAY2BGR)
    elif hdr_img.shape[2] == 4:
        print("Converting RGBA to RGB")
        hdr_img = hdr_img[:, :, :3]
    
    print(f"Image shape: {hdr_img.shape}")
    print(f"Image data type: {hdr_img.dtype}")
    print(f"Intensity range: [{np.min(hdr_img):.6f}, {np.max(hdr_img):.6f}]")
    
    if hdr_img.dtype != np.float32:
        hdr_img = hdr_img.astype(np.float32)
    
    print("\nApplying HDR histogram equalization...")
    equalized_img, original_lum, equalized_lum = hdr_histogram_equalization(
        hdr_img, target_min=0, target_max=255
    )
    
    equalized_8bit = np.clip(equalized_img, 0, 255).astype(np.uint8)
    
    log_original = np.log(hdr_img + 1e-8)
    log_normalized = (log_original - np.min(log_original)) / (np.max(log_original) - np.min(log_original))
    original_8bit = (log_normalized * 255).astype(np.uint8)
    
    output_dir = "hdr_output"
    os.makedirs(output_dir, exist_ok=True)
    
    cv2.imwrite(os.path.join(output_dir, "original_tonemapped.jpg"), original_8bit)
    cv2.imwrite(os.path.join(output_dir, "histogram_equalized.jpg"), equalized_8bit)
    
    print("\nResults saved in hdr_output folder:")
    print(f"- {os.path.join(output_dir, 'original_tonemapped.jpg')} (tone-mapped original)")
    print(f"- {os.path.join(output_dir, 'histogram_equalized.jpg')} (histogram equalized)")
    
    print(f"\nOriginal luminance range: [{np.min(original_lum):.6f}, {np.max(original_lum):.6f}]")
    print(f"Equalized luminance range: [{np.min(equalized_lum):.6f}, {np.max(equalized_lum):.6f}]")
    print(f"Target range achieved: [0, 255]")
    
    unique_original = len(np.unique(original_lum))
    unique_equalized = len(np.unique(equalized_lum))
    print(f"\nUnique luminance values - Original: {unique_original}, Equalized: {unique_equalized}")
    
    return equalized_img, original_lum, equalized_lum

def analyze_distribution(luminance_values, title="Luminance Distribution"):
    print(f"\n{title}:")
    print(f"  Mean: {np.mean(luminance_values):.6f}")
    print(f"  Std: {np.std(luminance_values):.6f}")
    print(f"  Min: {np.min(luminance_values):.6f}")
    print(f"  Max: {np.max(luminance_values):.6f}")
    print(f"  Median: {np.median(luminance_values):.6f}")
    
    percentiles = [10, 25, 75, 90, 95, 99]
    print("  Percentiles:", end=" ")
    for p in percentiles:
        print(f"{p}%: {np.percentile(luminance_values, p):.6f}", end=" | ")
    print()

if __name__ == "__main__":
    print("HDR Histogram Equalization Algorithm")
    print("=" * 40)
    
    result = load_and_process_hdr()
    
    if result is not None:
        equalized_img, original_lum, equalized_lum = result
        
        analyze_distribution(original_lum, "Original Luminance Distribution")
        analyze_distribution(equalized_lum, "Equalized Luminance Distribution")
        
        print("\nHistogram equalization completed successfully!")
        print("The algorithm works by:")
        print("1. Computing luminance from RGB channels")
        print("2. Sorting all pixel intensities (no quantization)")
        print("3. Mapping each pixel's rank to uniform distribution")
        print("4. Preserving color ratios during transformation")
        print("\nThis approach handles arbitrary real-valued intensities")
        print("without requiring histogram bins or quantization.")

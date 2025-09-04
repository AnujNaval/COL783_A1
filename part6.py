import cv2
import numpy as np
import matplotlib.pyplot as plt

def gamma_correction(image, gamma):
    # Normalize to [0,1]
    norm = image.astype(np.float32) / 255.0
    corrected = np.power(norm, gamma)
    return np.clip(corrected * 255, 0, 255).astype(np.uint8)

def disk_kernel(radius):
    """Create normalized disk kernel with given radius"""
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    mask = x**2 + y**2 <= radius**2
    kernel = mask.astype(np.float32)
    kernel /= kernel.sum()
    return kernel

def custom_convolution(image, kernel):
    img_float = image.astype(np.float32)
    k_h, k_w = kernel.shape
    pad_h, pad_w = k_h // 2, k_w // 2
    padded = cv2.copyMakeBorder(img_float, pad_h, pad_h, pad_w, pad_w, 
                                cv2.BORDER_REPLICATE)
    output = np.zeros_like(img_float, dtype=np.float32)

    for i in range(img_float.shape[0]):
        for j in range(img_float.shape[1]):
            region = padded[i:i+k_h, j:j+k_w]
            output[i, j] = np.sum(region * kernel)
    return np.clip(output, 0, 255).astype(np.uint8)


# Load grayscale image
image = cv2.imread("part_5_6_7_resource_images/scenery_2.jpg", cv2.IMREAD_GRAYSCALE)

# Parameters
gamma = 0.2   # gamma << 1 to enhance dark regions
radius = 5
kernel = disk_kernel(radius)

# Method 1: Gamma before convolution
gamma_before = gamma_correction(image, gamma)
result1 = custom_convolution(gamma_before, kernel)

# Method 2: Gamma after convolution
blurred = custom_convolution(image, kernel)
result2 = gamma_correction(blurred, gamma)

# Show results
plt.figure(figsize=(15,6))

plt.subplot(1,3,1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.axis("off")

plt.subplot(1,3,2)
plt.title("Gamma before Convolution")
plt.imshow(result1, cmap='gray')
plt.axis("off")

plt.subplot(1,3,3)
plt.title("Gamma after Convolution")
plt.imshow(result2, cmap='gray')
plt.axis("off")

plt.show()

"""
Expected Results

Gamma before convolution: Enhances local contrast first -> blur spreads modified intensities.

Gamma after convolution: Looks like an out-of-focus camera (blur first, then tone mapping).

Both look different -> but the second is more realistic for actual imaging."""
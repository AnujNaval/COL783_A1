import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
def custom_convolution(image, kernel, c=0):
    """Apply custom convolution with floating-point kernel and optional bias c."""
    # Ensure image is float for computation
    img_float = image.astype(np.float32)

    # Kernel size
    k_h, k_w = kernel.shape
    pad_h, pad_w = k_h // 2, k_w // 2

    # Pad image
    padded = cv2.copyMakeBorder(img_float, pad_h, pad_h, pad_w, pad_w, 
                                cv2.BORDER_REPLICATE)

    # Output image
    output = np.zeros_like(img_float, dtype=np.float32)

    # Convolution operation
    for i in range(img_float.shape[0]):
        for j in range(img_float.shape[1]):
            region = padded[i:i+k_h, j:j+k_w]
            value = np.sum(region * kernel) + c
            output[i, j] = value

    # Clip to 8-bit range
    output = np.clip(output, 0, 255).astype(np.uint8)
    return output


def gaussian_kernel_2d(ksize, sigma):
    """Generate a normalized 2D Gaussian kernel."""
    ax = np.linspace(-(ksize // 2), ksize // 2, ksize)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    kernel /= np.sum(kernel)
    return kernel.astype(np.float32)


def gaussian_kernel_1d(ksize, sigma):
    """Generate a normalized 1D Gaussian kernel."""
    ax = np.linspace(-(ksize // 2), ksize // 2, ksize)
    kernel = np.exp(-(ax**2) / (2. * sigma**2))
    kernel /= np.sum(kernel)
    return kernel.astype(np.float32)

# Load grayscale image
image = cv2.imread("part_5_6_7_resource_images/cat.jpg", cv2.IMREAD_GRAYSCALE)

# Define Laplacian kernel (3x3)
laplacian_kernel = np.array([[0, 1, 0],
                             [1, -4, 1],
                             [0, 1, 0]], dtype=np.float32)

# Apply convolution with c=128
laplacian_result = custom_convolution(image, laplacian_kernel, c=128)

# Show results
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.axis("off")

plt.subplot(1,2,2)
plt.title("Laplacian (with c=128)")
plt.imshow(laplacian_result, cmap='gray')
plt.axis("off")

plt.show()

# Parameters
sigma = 5.0
ksize = int(6*sigma + 1)   # Rule of thumb for kernel size

# --------- Method 1: Naive 2D Gaussian Convolution ----------
gauss2d = gaussian_kernel_2d(ksize, sigma)

start_time = time.time()
gaussian_result_2d = custom_convolution(image, gauss2d)
time_2d = time.time() - start_time


# --------- Method 2: Separable Gaussian Convolution ----------
gauss1d = gaussian_kernel_1d(ksize, sigma)

start_time = time.time()
# First convolve rows
temp = custom_convolution(image, gauss1d.reshape(1, -1))
# Then convolve cols
gaussian_result_sep = custom_convolution(temp, gauss1d.reshape(-1, 1))
time_sep = time.time() - start_time


# --------- Show Results ----------
plt.figure(figsize=(15,6))

plt.subplot(1,3,1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.axis("off")

plt.subplot(1,3,2)
plt.title(f"Gaussian 2D (time={time_2d:.3f}s)")
plt.imshow(gaussian_result_2d, cmap='gray')
plt.axis("off")

plt.subplot(1,3,3)
plt.title(f"Gaussian Separable (time={time_sep:.3f}s)")
plt.imshow(gaussian_result_sep, cmap='gray')
plt.axis("off")

plt.show()

print(f"Naive 2D Gaussian time: {time_2d:.4f} s")
print(f"Separable Gaussian time: {time_sep:.4f} s")


#-------------------------------------part(C)-------------------------------------#


# Gaussian kernel (moderate sigma)
sigma = 3.0
ksize = int(6*sigma + 1)
gaussian_kernel = gaussian_kernel_2d(ksize, sigma)


# -------- Method 1: l * (g * f) --------
g_f = custom_convolution(image, gaussian_kernel)
lgf = custom_convolution(g_f, laplacian_kernel, c=128)

# -------- Method 2: g * (l * f) --------
l_f = custom_convolution(image, laplacian_kernel, c=128)
glf = custom_convolution(l_f, gaussian_kernel)

# -------- Method 3: (l * g) * f --------
lg = custom_convolution(gaussian_kernel, laplacian_kernel)  # LoG kernel
lgf2 = custom_convolution(image, lg, c=128)


# -------- Show Results --------
plt.figure(figsize=(15,6))

plt.subplot(1,3,1)
plt.title("l * (g * f)")
plt.imshow(lgf, cmap='gray')
plt.axis("off")

plt.subplot(1,3,2)
plt.title("g * (l * f)")
plt.imshow(glf, cmap='gray')
plt.axis("off")

plt.subplot(1,3,3)
plt.title("(l * g) * f")
plt.imshow(lgf2, cmap='gray')
plt.axis("off")

plt.show()

"""
Why results differ in practice

1. Finite precision & rounding: floating-point operations, clipping, and integer rounding accumulate differently depending on the order.

2. Kernel truncation: Gaussian is truncated to finite size → mathematically infinite, so results vary depending on how/when it is applied.

3. Border effects: padding influences the order of convolution differently.

4. Discrete sampling: convolution order is equivalent in theory (convolution is associative/commutative), but in sampled, finite images it isn’t exact.
"""
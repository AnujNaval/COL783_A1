import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- Step 1: Load image and convert to HSI ---
def rgb_to_hsi(img):
    # Convert RGB [0,255] to [0,1]
    img = img.astype(np.float32) / 255.0
    R, G, B = img[:,:,0], img[:,:,1], img[:,:,2]

    # Intensity
    I = (R + G + B) / 3.0

    # Saturation
    min_val = np.minimum(np.minimum(R, G), B)
    S = 1 - (3 / (R + G + B + 1e-6)) * min_val

    # Hue
    num = 0.5 * ((R - G) + (R - B))
    den = np.sqrt((R - G)**2 + (R - B) * (G - B)) + 1e-6
    theta = np.arccos(num / den)

    H = np.where(B <= G, theta, 2*np.pi - theta)
    H = H / (2*np.pi)  # normalize [0,1]

    return np.stack([H, S, I], axis=-1)

def hsi_to_rgb(hsi):
    H, S, I = hsi[:,:,0]*2*np.pi, hsi[:,:,1], hsi[:,:,2]
    R, G, B = np.zeros_like(H), np.zeros_like(H), np.zeros_like(H)

    # Sector 0: 0 <= H < 2π/3
    idx = (H >= 0) & (H < 2*np.pi/3)
    B[idx] = I[idx]*(1-S[idx])
    R[idx] = I[idx]*(1 + (S[idx]*np.cos(H[idx]))/(np.cos(np.pi/3 - H[idx]) + 1e-6))
    G[idx] = 3*I[idx] - (R[idx] + B[idx])

    # Sector 1: 2π/3 <= H < 4π/3
    idx = (H >= 2*np.pi/3) & (H < 4*np.pi/3)
    H_tmp = H[idx] - 2*np.pi/3
    R[idx] = I[idx]*(1-S[idx])
    G[idx] = I[idx]*(1 + (S[idx]*np.cos(H_tmp))/(np.cos(np.pi/3 - H_tmp) + 1e-6))
    B[idx] = 3*I[idx] - (R[idx] + G[idx])

    # Sector 2: 4π/3 <= H < 2π
    idx = (H >= 4*np.pi/3) & (H < 2*np.pi)
    H_tmp = H[idx] - 4*np.pi/3
    G[idx] = I[idx]*(1-S[idx])
    B[idx] = I[idx]*(1 + (S[idx]*np.cos(H_tmp))/(np.cos(np.pi/3 - H_tmp) + 1e-6))
    R[idx] = 3*I[idx] - (G[idx] + B[idx])

    rgb = np.stack([R,G,B], axis=-1)
    return np.clip(rgb*255, 0, 255).astype(np.uint8)

# Load image
img = cv2.imread("part_5_6_7_resource_images/fruits.jpg")  # replace with your image
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

hsi = rgb_to_hsi(img)

# Display H, S, I channels
fig, axs = plt.subplots(1,3, figsize=(12,4))
axs[0].imshow(hsi[:,:,0], cmap='gray'); axs[0].set_title("Hue")
axs[1].imshow(hsi[:,:,1], cmap='gray'); axs[1].set_title("Saturation")
axs[2].imshow(hsi[:,:,2], cmap='gray'); axs[2].set_title("Intensity")
plt.show()

# --- Step 2: Colour slicing mask ---
seed = (100, 100)   # choose seed pixel coordinates
c_s = hsi[seed[1], seed[0], :]  # HSI vector

dH, dS, dI = 0.05, 0.3, 0.3   # thresholds
mask = ((np.abs(hsi[:,:,0] - c_s[0]) < dH) &
        (np.abs(hsi[:,:,1] - c_s[1]) < dS) &
        (np.abs(hsi[:,:,2] - c_s[2]) < dI))

# Show mask
plt.imshow(mask, cmap='gray')
plt.title("Mask of selected object")
plt.show()

# --- Step 3: Colour transformation ---
c_t = np.array([0.8, 0.9, 0.7])  # target HSI colour

transformed = hsi.copy()
for k in range(3):
    if k == 0:  # Hue: additive shift
        transformed[:,:,k][mask] = (hsi[:,:,k][mask] + (c_t[k] - c_s[k])) % 1.0
    elif k == 1:  # Saturation: multiplicative
        transformed[:,:,k][mask] = hsi[:,:,k][mask] * (c_t[k]/(c_s[k]+1e-6))
    else:  # Intensity: additive
        transformed[:,:,k][mask] = np.clip(hsi[:,:,k][mask] + (c_t[k] - c_s[k]), 0, 1)

# Convert back to RGB
img_new = hsi_to_rgb(transformed)

# Show results
plt.subplot(1,2,1); plt.imshow(img); plt.title("Original")
plt.subplot(1,2,2); plt.imshow(img_new); plt.title("Recoloured Object")
plt.show()

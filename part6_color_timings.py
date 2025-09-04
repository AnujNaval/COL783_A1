import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

# ---------- kernels & basic ops ----------
def disk_kernel(radius: int) -> np.ndarray:
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    mask = (x**2 + y**2) <= radius**2
    k = mask.astype(np.float32)
    k /= k.sum()
    return k

def custom_convolution(gray: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Gray image (uint8), float kernel -> uint8 (no bias)."""
    img_f = gray.astype(np.float32)
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    pad = cv2.copyMakeBorder(img_f, ph, ph, pw, pw, cv2.BORDER_REPLICATE)
    out = np.zeros_like(img_f, np.float32)
    for i in range(img_f.shape[0]):
        for j in range(img_f.shape[1]):
            region = pad[i:i+kh, j:j+kw]
            out[i, j] = np.sum(region * kernel)
    return np.clip(out, 0, 255).astype(np.uint8)

def gamma_u8(channel_u8: np.ndarray, gamma: float) -> np.ndarray:
    """Gamma on an 8-bit channel."""
    x = channel_u8.astype(np.float32) / 255.0
    y = np.power(x, gamma)
    return np.clip(y * 255.0, 0, 255).astype(np.uint8)

# ---------- luminance-space helpers ----------
def blur_on_luminance_bgr(bgr: np.ndarray, kernel: np.ndarray, space="lab") -> np.ndarray:
    if space.lower() == "hsv":
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        H, S, V = cv2.split(hsv)
        V_blur = custom_convolution(V, kernel)
        hsv_out = cv2.merge([H, S, V_blur])
        return cv2.cvtColor(hsv_out, cv2.COLOR_HSV2BGR)
    else:  # Lab
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        L, a, b = cv2.split(lab)
        L_blur = custom_convolution(L, kernel)
        lab_out = cv2.merge([L_blur, a, b])
        return cv2.cvtColor(lab_out, cv2.COLOR_LAB2BGR)

def gamma_on_luminance_bgr(bgr: np.ndarray, gamma: float, space="lab") -> np.ndarray:
    if space.lower() == "hsv":
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        H, S, V = cv2.split(hsv)
        Vg = gamma_u8(V, gamma)
        hsv_out = cv2.merge([H, S, Vg])
        return cv2.cvtColor(hsv_out, cv2.COLOR_HSV2BGR)
    else:  # Lab
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        L, a, b = cv2.split(lab)
        Lg = gamma_u8(L, gamma)
        lab_out = cv2.merge([Lg, a, b])
        return cv2.cvtColor(lab_out, cv2.COLOR_LAB2BGR)

def pipeline(bgr: np.ndarray, gamma: float, kernel: np.ndarray, space="lab", order="gamma_before") -> np.ndarray:
    """
    order:
      - 'gamma_before': T(w *before? no) => w * (T(f))  (gamma then blur)
      - 'gamma_after' : T(w * f)          (blur then gamma)
    """
    if order == "gamma_before":
        x = gamma_on_luminance_bgr(bgr, gamma, space=space)
        y = blur_on_luminance_bgr(x, kernel, space=space)
        return y
    else:
        x = blur_on_luminance_bgr(bgr, kernel, space=space)
        y = gamma_on_luminance_bgr(x, gamma, space=space)
        return y
# ---------- demo ----------
# Load your color image
img_path = "part_5_6_7_resource_images/scenery_2.jpg"  # replace with your own image
bgr = cv2.imread(img_path)
if bgr is None:
    raise FileNotFoundError(f"Could not read {img_path}")

gamma = 0.25
radius = 6
kernel = disk_kernel(radius)

results = {}
timings = {}

for space in ["hsv", "lab"]:
    for order in ["gamma_before", "gamma_after"]:
        key = f"{space}_{order}"
        start = time.time()
        out = pipeline(bgr, gamma, kernel, space=space, order=order)
        t = time.time() - start
        results[key] = out
        timings[key] = t

# Display results and timings
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

axes[0].set_title("Original")
axes[0].imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
axes[0].axis("off")

titles = ["HSV: gamma → blur", "HSV: blur → gamma", "Lab: gamma → blur",
          "Lab: blur → gamma", "Difference (Lab pipel.)"]
plot_idxs = [1, 2, 3, 4, 5]

# for idx, title in zip(plot_idxs, titles):
#     axes[idx].set_title(title)
#     if "Difference" in title:
#         img1 = results["lab_gamma_before"]
#         img2 = results["lab_gamma_after"]
#         diff = cv2.absdiff(img1, img2)
#         axes[idx].imshow(cv2.cvtColor(diff, cv2.COLOR_BGR2RGB))
#     else:
#         key = title.lower().replace(": ", "_").replace(" ", "_")
#         axes[idx].imshow(cv2.cvtColor(results[key], cv2.COLOR_BGR2RGB))
#     axes[idx].axis("off")
# Mapping titles to results dict keys
title_to_key = {
    "HSV: gamma → blur": "hsv_gamma_before",
    "HSV: blur → gamma": "hsv_gamma_after",
    "Lab: gamma → blur": "lab_gamma_before",
    "Lab: blur → gamma": "lab_gamma_after",
}

for idx, title in zip(plot_idxs, titles):
    axes[idx].set_title(title)
    if "Difference" in title:
        img1 = results["lab_gamma_before"]
        img2 = results["lab_gamma_after"]
        diff = cv2.absdiff(img1, img2)
        # diff_vis = cv2.convertScaleAbs(diff, alpha=10)  # multiply differences by 10
        axes[idx].imshow(cv2.cvtColor(diff, cv2.COLOR_BGR2RGB))
    else:
        key = title_to_key[title]
        axes[idx].imshow(cv2.cvtColor(results[key], cv2.COLOR_BGR2RGB))
    axes[idx].axis("off")


plt.tight_layout()
plt.show()

# Print timing summary
print("Timing Summary (seconds):")
for key in timings:
    print(f"  {key}: {timings[key]:.4f} s")

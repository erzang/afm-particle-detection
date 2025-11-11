import cv2
import numpy as np
from igor2 import binarywave
from skimage.morphology import white_tophat, disk
import matplotlib.pyplot as plt


def load_ibw_heightmap(path_to_ibw):
    """Load AFM height map (HtT channel) from an Igor Binary Wave file."""
    with open(path_to_ibw, "rb") as f:
        wave = binarywave.load(f)

    data = wave["wave"]["wData"]
    height_map = data[:, :, 0]

    scale_x, scale_y, scale_z = wave["wave"]["wave_header"]["sfA"][:3]

    height_nm = (height_map * scale_z * 1e9).astype(np.float32)
    height_nm = np.rot90(height_nm, k=1)

    return height_nm


def wth_flatten(height_nm, radius):
    """Flatten AFM height map using a morphological white top-hat."""
    selem = disk(radius)
    return white_tophat(height_nm.astype(np.float32), selem)


def filter_particles(binary_mask):
    """Remove edge-touching and zero-area contours."""
    mask_uint8 = (binary_mask > 0).astype(np.uint8) * 255
    h, w = mask_uint8.shape

    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered = np.zeros_like(mask_uint8)

    for cnt in contours:
        xs = cnt[:, 0, 0]
        ys = cnt[:, 0, 1]
        if xs.min() <= 0 or xs.max() >= w - 1 or ys.min() <= 0 or ys.max() >= h - 1:
            continue
        if cv2.contourArea(cnt) <= 0:
            continue
        cv2.drawContours(filtered, [cnt], -1, 255, -1)

    return filtered


def compare_bounds(mask1, img1, mask2, img2, label1="Original", label2="WTH"):
    filtered1 = filter_particles(mask1)
    filtered2 = filter_particles(mask2)

    contours1, _ = cv2.findContours(filtered1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours2, _ = cv2.findContours(filtered2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    vis1 = cv2.cvtColor(
        cv2.normalize(img1, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
        cv2.COLOR_GRAY2BGR
    )
    vis2 = cv2.cvtColor(
        cv2.normalize(img2, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
        cv2.COLOR_GRAY2BGR
    )
    cv2.drawContours(vis1, contours1, -1, (255, 0, 0), 1)
    cv2.drawContours(vis2, contours2, -1, (255, 0, 0), 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(vis1)
    plt.title(f"{label1} (Count: {len(contours1)})")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(vis2)
    plt.title(f"{label2} (Count: {len(contours2)})")
    plt.axis("off")
    plt.show()


# Example usage (replace path)
if __name__ == "__main__":
    height_nm = load_ibw_heightmap("path/to/file.ibw")

    # Flatten using WTH
    flattened = wth_flatten(height_nm, radius=45)

    # Thresholding
    img_orig = cv2.normalize(height_nm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    img_flat = cv2.normalize(flattened, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    mask_orig = cv2.adaptiveThreshold(img_orig, 255,
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY,
                                      71, -2)
    mask_flat = cv2.adaptiveThreshold(img_flat, 255,
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY,
                                      71, -2)

    compare_bounds(mask_orig, height_nm, mask_flat, flattened)
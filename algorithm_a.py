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

    # Scale factors (meters per pixel)
    scale_x, scale_y, scale_z = wave["wave"]["wave_header"]["sfA"][:3]

    # Convert to nanometers and rotate for consistent orientation
    height_nm = (height_map * scale_z * 1e9).astype(np.float32)
    height_nm = np.rot90(height_nm, k=1)

    return height_nm


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


def visualize_bounds(binary_mask, original):
    """Overlay contour bounds on a normalized version of the AFM height map."""
    filtered = filter_particles(binary_mask)
    contours, _ = cv2.findContours(filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    vis = cv2.cvtColor(
        cv2.normalize(original, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
        cv2.COLOR_GRAY2BGR
    )
    cv2.drawContours(vis, contours, -1, (255, 0, 0), 1)

    plt.figure(figsize=(6, 6))
    plt.imshow(vis)
    plt.title(f"Particle Bounds (Count: {len(contours)})")
    plt.axis("off")
    plt.show()


# Example usage (replace path)
if __name__ == "__main__":
    height_nm = load_ibw_heightmap("path/to/file.ibw")

    img_8bit = cv2.normalize(height_nm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    thresh = cv2.adaptiveThreshold(
        img_8bit, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        71, -2
    )

    visualize_bounds(thresh, height_nm)
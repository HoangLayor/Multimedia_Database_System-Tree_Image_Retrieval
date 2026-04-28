import cv2
import numpy as np

def _calculate_hw_ratio(w: int, h: int) -> float:
    return h / w if w > 0 else 0.0

def _calculate_canopy_density(mask: np.ndarray, x: int, y: int, w: int, h: int) -> float:
    tree_pixels = cv2.countNonZero(mask[y:y+h, x:x+w])
    return tree_pixels / (w * h) if (w * h) > 0 else 0.0

def _calculate_trunk_width_ratio(mask: np.ndarray, x: int, y: int, w: int, h: int) -> float:
    bottom_h = max(1, int(h * 0.1))
    bottom_part = mask[y+h-bottom_h:y+h, x:x+w]
    trunk_pixels = cv2.countNonZero(bottom_part)
    return (trunk_pixels / bottom_h) / w if (bottom_h * w) > 0 else 0.0

def _calculate_canopy_symmetry(mask: np.ndarray, x: int, y: int, w: int, h: int) -> float:
    half_w = w // 2
    left_half = mask[y:y+h, x:x+half_w]
    right_half = mask[y:y+h, x+w-half_w:x+w]
    
    left_area = cv2.countNonZero(left_half)
    right_area = cv2.countNonZero(right_half)
    total_area = left_area + right_area
    return 1.0 - (abs(left_area - right_area) / total_area) if total_area > 0 else 0.0

def _calculate_branch_angle_variance(gray: np.ndarray, mask: np.ndarray) -> float:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    
    significant_angles = angle[(mag > 50) & (mask > 0)]
    if len(significant_angles) > 0:
        # Normalize variance by maximum possible (180^2 / 12 for uniform distribution, roughly)
        # Here we use a simpler normalization by 32400 (180^2)
        return float(np.var(significant_angles) / 32400.0)
    return 0.0

def extract_geometry(img: np.ndarray) -> list[float]:
    """
    Extract 5 geometric features using a modular approach.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    
    coords = cv2.findNonZero(mask)
    if coords is None:
        return [0.0] * 5
    
    x, y, w, h = cv2.boundingRect(coords)
    
    return [
        float(_calculate_hw_ratio(w, h)),
        float(_calculate_canopy_density(mask, x, y, w, h)),
        float(_calculate_trunk_width_ratio(mask, x, y, w, h)),
        float(_calculate_canopy_symmetry(mask, x, y, w, h)),
        float(_calculate_branch_angle_variance(gray, mask))
    ]

import cv2
import numpy as np

def _calculate_primary_metrics(masked_pixels: np.ndarray) -> tuple[float, float]:
    """Returns (mean_h, mean_s) normalized."""
    mean_h = np.mean(masked_pixels[:, 0]) / 180.0
    mean_s = np.mean(masked_pixels[:, 1]) / 255.0
    return mean_h, mean_s

def _calculate_leaf_color_variance(img: np.ndarray, mask: np.ndarray) -> float:
    green_channel = img[:, :, 1][mask > 0]
    return np.std(green_channel) / 255.0

def _calculate_brown_gray_ratio(masked_pixels: np.ndarray) -> float:
    h = masked_pixels[:, 0]
    s = masked_pixels[:, 1]
    
    brown_mask = (h >= 5) & (h <= 30) & (s >= 20) & (s <= 150)
    gray_mask = (s < 40)
    return np.sum(brown_mask | gray_mask) / len(masked_pixels)

def _calculate_hue_histogram(hsv: np.ndarray, mask: np.ndarray, bins: int = 5) -> list[float]:
    hist = cv2.calcHist([hsv], [0], mask, [bins], [0, 180])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist.flatten().tolist()

def extract_color(img: np.ndarray) -> list[float]:
    """
    Extract 9 color features using a modular approach.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    
    if cv2.countNonZero(mask) == 0:
        return [0.0] * 9
        
    masked_pixels = hsv[mask > 0]
    mean_h, mean_s = _calculate_primary_metrics(masked_pixels)
    leaf_var = _calculate_leaf_color_variance(img, mask)
    bg_ratio = _calculate_brown_gray_ratio(masked_pixels)
    hue_hist = _calculate_hue_histogram(hsv, mask)

    return [
        float(mean_h),
        float(mean_s),
        float(leaf_var),
        float(bg_ratio)
    ] + [float(x) for x in hue_hist]

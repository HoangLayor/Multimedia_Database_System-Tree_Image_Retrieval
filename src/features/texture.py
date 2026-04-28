import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from scipy.stats import entropy

def _calculate_lbp_coarseness(gray: np.ndarray, mask: np.ndarray) -> float:
    lbp = local_binary_pattern(gray, 8, 1, method='uniform')
    lbp_values = lbp[mask > 0]
    return np.var(lbp_values) / 10.0

def _calculate_contour_complexity(mask: np.ndarray) -> float:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    if area > 0:
        complexity = (perimeter ** 2) / (4 * np.pi * area)
        return min(10.0, complexity) / 10.0
    return 0.0

def _calculate_edge_density(gray: np.ndarray, mask: np.ndarray) -> float:
    edges = cv2.Canny(gray, 100, 200)
    tree_edges = cv2.bitwise_and(edges, edges, mask=mask)
    edge_count = cv2.countNonZero(tree_edges)
    tree_area = cv2.countNonZero(mask)
    return edge_count / tree_area if tree_area > 0 else 0.0

def _calculate_texture_entropy(gray: np.ndarray, mask: np.ndarray) -> float:
    roi_gray = gray[mask > 0]
    hist, _ = np.histogram(roi_gray, bins=256, range=(0, 255), density=True)
    hist = hist[hist > 0]
    return entropy(hist, base=2) / 8.0

def extract_texture(img: np.ndarray) -> list[float]:
    """
    Extract 4 texture features using a modular approach.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    
    if cv2.countNonZero(mask) == 0:
        return [0.0] * 4
        
    return [
        float(_calculate_lbp_coarseness(gray, mask)),
        float(_calculate_contour_complexity(mask)),
        float(_calculate_edge_density(gray, mask)),
        float(_calculate_texture_entropy(gray, mask))
    ]

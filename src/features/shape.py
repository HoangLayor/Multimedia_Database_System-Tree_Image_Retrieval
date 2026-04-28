import cv2
import numpy as np

def _calculate_hu_moments(cnt: np.ndarray, n: int = 3) -> list[float]:
    moments = cv2.moments(cnt)
    hu = cv2.HuMoments(moments).flatten()
    hu_feats = []
    for i in range(n):
        val = hu[i]
        if val != 0:
            val = -1.0 * np.sign(val) * np.log10(np.abs(val))
        hu_feats.append(val / 20.0)
    return hu_feats

def _calculate_solidity(cnt: np.ndarray) -> float:
    area = cv2.contourArea(cnt)
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    return area / hull_area if hull_area > 0 else 0.0

def _calculate_circularity(cnt: np.ndarray) -> float:
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0.0
    return min(1.0, max(0.0, circularity))

def extract_shape(img: np.ndarray) -> list[float]:
    """
    Extract 5 shape features using a modular approach.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return [0.0] * 5
        
    cnt = max(contours, key=cv2.contourArea)
    
    hu_feats = _calculate_hu_moments(cnt, 3)
    solidity = _calculate_solidity(cnt)
    circularity = _calculate_circularity(cnt)

    return [
        float(hu_feats[0]),
        float(hu_feats[1]),
        float(hu_feats[2]),
        float(solidity),
        float(circularity)
    ]

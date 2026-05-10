import cv2
import numpy as np

def _calculate_hu_moments(cnt: np.ndarray, n: int = 7) -> list[float]:
    """
    Kéo xuất n Momen Hu (Hu Moments) đầu tiên từ viền Contour của cây.
    Momen Hu là tập hợp độ đo hình dáng bất biến (không thay đổi dù cây bị xoay, 
    lật, thay đổi kích thước). Đặc biệt hữu ích để nhận dạng hình thái chung (Shape).
    """
    moments = cv2.moments(cnt)
    hu = cv2.HuMoments(moments).flatten()
    hu_feats = []
    for i in range(min(n, len(hu))):
        val = hu[i]
        if val != 0:
            # Transform log để nén khoảng dao động của giá trị
            val = -1.0 * np.sign(val) * np.log10(np.abs(val))
        hu_feats.append(val / 20.0)  # /20.0 để chuẩn hoá tương đối
    return hu_feats

def _calculate_solidity(cnt: np.ndarray) -> float:
    """
    Tính Độ đặc (Solidity) của cây.
    Là tỷ lệ giữa Diện tích bề mặt cây / Diện tích bao lồi (Convex Hull).
    Solidity < 1 cho thấy lá và cành tua tủa chia nhiều hướng, tạo hốc trống.
    Solidity gần 1 là cây có dáng ôm tròn hoặc chùm đặc (bụi cây khép kín).
    """
    area = cv2.contourArea(cnt)
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    return area / hull_area if hull_area > 0 else 0.0

def _calculate_circularity(cnt: np.ndarray) -> float:
    """
    Tính Độ tròn trịa (Circularity) của cây.
    Dựa trên công thức 4 * Pi * Diện tích / (Chu vi ^ 2).
    Circularity = 1 nếu cái cây tròn xoe như quả bóng.
    Circularity < 1 chỉ ra hình dáng cây kéo dài, lệch, hoặc có nhiều góc cạnh.
    """
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0.0
    return min(1.0, max(0.0, circularity))

def extract_shape(img: np.ndarray) -> list[float]:
    """
    Trích xuất tổng hợp 5 đặc trưng Hình dạng (Shape Features).
    Dùng Contour làm đối tượng phân tích chính thay vì Mask diện rộng.
    
    Quy trình: 
    Tìm viền (Contour) lớn nhất và tính 3 Momen Hu, tính Độ đặc (Solidity) và Độ tròn (Circularity).
    
    Args:
        img (np.ndarray): Ảnh đầu vào chứa cây (định dạng BGR).
        
    Returns:
        list[float]: 5 giá trị đặc trưng hình dáng dạng chuẩn hoá gồm [Hu1, Hu2, Hu3, Solidity, Circularity].
    """
    from .utils import get_tree_mask
    mask = get_tree_mask(img)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return [0.0] * 9  # 7 Hu + Solidity + Circularity
        
    cnt = max(contours, key=cv2.contourArea)
    
    hu_feats = _calculate_hu_moments(cnt, 7)
    solidity = _calculate_solidity(cnt)
    circularity = _calculate_circularity(cnt)

    return [float(h) for h in hu_feats] + [
        float(solidity),
        float(circularity)
    ]

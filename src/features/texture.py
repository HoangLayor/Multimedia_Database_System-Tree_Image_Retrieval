import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from scipy.stats import entropy

def _calculate_lbp_coarseness(gray: np.ndarray, mask: np.ndarray) -> float:
    """
    Tính toán độ thô ráp / sần sùi (Coarseness) của tán lá bằng Local Binary Pattern (LBP).
    Qua đó phản ánh cách các lớp lá đan xen nhau tạo nên cấu trúc hạt (hạt to hay mịn màng).
    Giá trị variance của điểm LBP càng cao, kết cấu bề mặt cây càng xù xì, gồ ghề.
    """
    lbp = local_binary_pattern(gray, 8, 1, method='uniform')
    lbp_values = lbp[mask > 0]
    return np.var(lbp_values) / 10.0

def _calculate_contour_complexity(mask: np.ndarray) -> float:
    """
    Tính Độ phức tạp của viền cây (Contour Complexity).
    Bằng (Chu vi ^ 2) / (4 * Pi * Diện tích).
    Đại diện cho số lượng vết răng cưa (viền gai gốc) hay độ nhấp nhô của tán lá quanh viền cây.
    Giá trị phức tạp cao thường thấy ở cây dương xỉ hay thông, lởm chởm lá ở vùng ranh giới.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    if area > 0:
        complexity = (perimeter ** 2) / (4 * np.pi * area)
        return min(10.0, complexity) / 10.0  # Chặn max 10 rồi chia 10 để normalize
    return 0.0

def _calculate_edge_density(gray: np.ndarray, mask: np.ndarray) -> float:
    """
    Tính mật độ cạnh cạnh viền bên trong tán lá (Edge Density).
    Dùng bộ quét cạnh Canny để dò tìm các đường nét cứng của kẽ lá lớn, cành đâm ngang, v.v...
    Tỉ lệ diện tích rìa/cạnh trên tổng diện tích vùng mask = Mật độ.
    """
    edges = cv2.Canny(gray, 100, 200)
    tree_edges = cv2.bitwise_and(edges, edges, mask=mask)
    edge_count = cv2.countNonZero(tree_edges)
    tree_area = cv2.countNonZero(mask)
    return edge_count / tree_area if tree_area > 0 else 0.0

def _calculate_texture_entropy(gray: np.ndarray, mask: np.ndarray) -> float:
    """
    Tính Entropy bề mặt (Texture Entropy).
    Dùng mức phân bổ histogram ánh sáng xám (grayscale) để đo mức độ hỗn loạn của chất liệu lá.
    Ảnh sắc nét nhiều chi tiết xám trắng xen kẽ nhau -> Entropy cao.
    Lá nhẵn thín đồng màu -> Entropy thấp cực đoan.
    """
    roi_gray = gray[mask > 0]
    hist, _ = np.histogram(roi_gray, bins=256, range=(0, 255), density=True)
    hist = hist[hist > 0]
    return entropy(hist, base=2) / 8.0

def extract_texture(img: np.ndarray) -> list[float]:
    """
    Trích xuất tổng cộng 4 đặc trưng Bề mặt (Texture Features) của tán lá.
    Sử dụng các thuật toán phân tích hoạ tiết như LBP (Local Binary Patterns) và Canny.
    
    Args:
        img (np.ndarray): Ảnh đầu vào chứa cây (định dạng BGR).
        
    Returns:
        list[float]: 4 giá trị đại diện cho [Độ LBP thô ráp, Phức tạp viền, Mật độ cạnh, Entropy].
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    from .utils import get_tree_mask
    mask = get_tree_mask(img)
    
    if cv2.countNonZero(mask) == 0:
        return [0.0] * 4
        
    return [
        float(_calculate_lbp_coarseness(gray, mask)),
        float(_calculate_contour_complexity(mask)),
        float(_calculate_edge_density(gray, mask)),
        float(_calculate_texture_entropy(gray, mask))
    ]

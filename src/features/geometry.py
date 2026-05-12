import cv2
import numpy as np

def _calculate_hw_ratio(w: int, h: int) -> float:
    """
    Tính tỷ lệ Chiều cao trên Chiều rộng (Height / Width Ratio).
    Giúp phân biệt các loại cây có dáng cao gầy (ví dụ: thông) với cây thấp bè (ví dụ: đa, bàng).
    """
    return h / w if w > 0 else 0.0

def _calculate_canopy_density(mask: np.ndarray, x: int, y: int, w: int, h: int) -> float:
    """
    Tính mật độ tán lá (Canopy Density).
    Đo bằng tỷ lệ giữa diện tích pixel thực tế của cây so với diện tích của khung bao khép kín (Bounding Box).
    Cây có tán lá khép kín, rậm rạp sẽ có mật độ cao hơn.
    """
    tree_pixels = cv2.countNonZero(mask[y:y+h, x:x+w])
    return tree_pixels / (w * h) if (w * h) > 0 else 0.0

def _calculate_trunk_diameter(mask: np.ndarray, x: int, y: int, w: int, h: int) -> float:
    """
    Tính đường kính thân cây (Trunk Diameter).
    Ước lượng dựa trên chiều rộng trung bình của phần gốc cây (10% chiều cao dưới cùng).
    Chỉ số này thường tỷ lệ thuận với tuổi thọ của cây.
    """
    bottom_h = max(1, int(h * 0.1))
    bottom_part = mask[y+h-bottom_h:y+h, x:x+w]
    
    # Tính chiều rộng trung bình của các hàng trong phần đáy
    widths = []
    for row in bottom_part:
        pixels = cv2.countNonZero(row)
        if pixels > 0:
            widths.append(pixels)
    
    if not widths:
        return 0.0
        
    avg_width = sum(widths) / len(widths)
    # Chuẩn hóa đường kính thân so với tổng chiều rộng khung bao
    return avg_width / w if w > 0 else 0.0

def _calculate_canopy_symmetry(mask: np.ndarray, x: int, y: int, w: int, h: int) -> float:
    """
    Tính độ đối xứng của tán lá (Canopy Symmetry).
    Chia đôi khung bao theo trục dọc và so sánh sự chênh lệch diện tích lá giữa nửa trái và nửa phải.
    Giá trị càng gần 1.0 nghĩa là nhánh cây mọc đều ra 2 bên, độ đối xứng càng cao.
    """
    half_w = w // 2
    left_half = mask[y:y+h, x:x+half_w]
    right_half = mask[y:y+h, x+w-half_w:x+w]
    
    left_area = cv2.countNonZero(left_half)
    right_area = cv2.countNonZero(right_half)
    total_area = left_area + right_area
    return 1.0 - (abs(left_area - right_area) / total_area) if total_area > 0 else 0.0

def _calculate_branch_angle_variance(gray: np.ndarray, mask: np.ndarray) -> float:
    """
    Tính độ phân tán góc của các nhánh / cành cây (Branch Angle Variance) bằng cách dùng bộ lọc Sobel 
    để tìm hướng gradient của biên cạnh. Mức biến thiên góc phân bố sẽ chỉ ra độ lởm chởm và hướng mọc cành.
    """
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    
    significant_angles = angle[(mag > 50) & (mask > 0)]
    if len(significant_angles) > 0:
        # Chuẩn hóa phương sai dưa trên giá trị tối đa ước tính (180^2 = 32400)
        return float(np.var(significant_angles) / 32400.0)
    return 0.0

def extract_geometry(img: np.ndarray) -> list[float]:
    """
    Trích xuất tổng cộng 5 đặc trưng Hình học (Geometry Features) của cây.
    Dựa trên việc tìm kiếm vùng bao (Bounding Box) làm khung tham chiếu ban đầu.
    
    Args:
        img (np.ndarray): Ảnh đầu vào chứa cây (định dạng BGR).
        
    Returns:
        list[float]: 5 giá trị đặc trưng dạng số thực, lần lượt là: Tỉ lệ độ cao/rộng, 
                     Mật độ tán, Độ rộng cành, Độ đối xứng tán và Phân tán hướng cành.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    from .utils import get_tree_mask
    mask = get_tree_mask(img)
    
    coords = cv2.findNonZero(mask)
    if coords is None:
        return [0.0] * 5
    
    x, y, w, h = cv2.boundingRect(coords)
    
    return [
        float(_calculate_hw_ratio(w, h)),
        float(_calculate_canopy_density(mask, x, y, w, h)),
        float(_calculate_trunk_diameter(mask, x, y, w, h)),
        float(_calculate_canopy_symmetry(mask, x, y, w, h)),
        float(_calculate_branch_angle_variance(gray, mask))
    ]

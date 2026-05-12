import cv2
import numpy as np

def _calculate_primary_metrics(masked_pixels: np.ndarray) -> tuple[float, float]:
    """
    Tính toán giá trị trung bình của kênh Hue (Màu sắc) và Saturation (Độ bão hòa).
    
    Args:
        masked_pixels (np.ndarray): Mảng các điểm ảnh (pixels) của cây trong không gian màu HSV.
        
    Returns:
        tuple[float, float]: Cặp giá trị (mean_h, mean_s) đã được chuẩn hóa (0.0 - 1.0).
                             - mean_h chia cho 180.0
                             - mean_s chia cho 255.0
    """
    # Tính trung bình trên mảng chiều số 0 (Kênh Hue) và chuẩn hoá theo thang OpenCV (0-179)
    mean_h = np.mean(masked_pixels[:, 0]) / 180.0
    # Tính trung bình trên mảng chiều số 1 (Kênh Saturation) và chuẩn hoá theo độ sâu màu 255
    mean_s = np.mean(masked_pixels[:, 1]) / 255.0
    return mean_h, mean_s

def _calculate_leaf_color_variance(img: np.ndarray, mask: np.ndarray) -> float:
    """
    Đo lường độ biến thiên (phân tán) của màu xanh lá trên tán cây.
    Giúp nhận biết tán lá có màu đồng nhất hay có nhiều sắc độ sáng/tối lốm đốm.
    
    Args:
        img (np.ndarray): Ảnh gốc đầu vào không gian màu BGR.
        mask (np.ndarray): Mặt nạ nhị phân (chỉ lấy vùng chứa cây, bỏ nền).
        
    Returns:
        float: Độ lệch chuẩn (standard deviation) của kênh màu xanh (Green) đã chuẩn hóa (0.0 - 1.0).
    """
    # Lấy kênh Green (index 1 trong hệ màu BGR) và chỉ giữ lại các pixel vùng mask (giá trị > 0)
    green_channel = img[:, :, 1][mask > 0]
    # Tính độ lệch chuẩn (standard deviation) để tìm khoảng dao động màu, sau đó chuẩn hóa
    return np.std(green_channel) / 255.0

def _calculate_brown_gray_ratio(masked_pixels: np.ndarray) -> float:
    """
    Tính tỷ lệ các điểm ảnh thuộc mảng màu Nâu (biểu thị thân, cành cây) 
    và Xám (biểu thị bóng râm, lá úa, cành khô).
    
    Args:
        masked_pixels (np.ndarray): Mảng các điểm ảnh của cây trong không gian màu HSV.
        
    Returns:
        float: Tỷ lệ phần trăm (0.0 - 1.0) của pixel màu nâu và xám trên tổng số pixel của cây.
    """
    # Tách riêng kênh Hue và Saturation trên tập pixel đã được lọc bởi mask
    h = masked_pixels[:, 0]
    s = masked_pixels[:, 1]
    
    # Theo kinh nghiệm (Heuristic), màu nâu thường có Hue nằm trong [5, 30] 
    # và Saturation dao động nhẹ [20, 150] (tránh lẫn màu cam sáng chói)
    brown_mask = (h >= 5) & (h <= 30) & (s >= 20) & (s <= 150)
    
    # Màu xám (bóng, râm, khô héo) thường có Saturation (bão hoà) rất thấp (< 40)
    gray_mask = (s < 40)
    
    # Tính tỷ lệ pixel thuộc nhóm Nâu/Xám trên tổng diện tích (len) của cây
    return np.sum(brown_mask | gray_mask) / len(masked_pixels)

def _calculate_hue_histogram(hsv: np.ndarray, mask: np.ndarray, bins: int = 5) -> list[float]:
    """
    Trích xuất biểu đồ phân bố màu sắc (Histogram) của kênh Hue chia thành các dải (bins).
    Giúp nắm bắt ngay các dải màu chính mà cây đang sở hữu (xanh lục, vàng, đỏ, ...).
    
    Args:
        hsv (np.ndarray): Ảnh đã chuyển đổi sang không gian màu HSV.
        mask (np.ndarray): Mặt nạ nhị phân để chỉ lấy vùng cây.
        bins (int): Số lượng dải màu (bins) muốn chia cho kênh Hue (mặc định là 5).
    Returns:
        list[float]: Danh sách tần suất xuất hiện của các dải màu (đã chuẩn hóa trong khoảng 0.0 - 1.0).
    """
    # Tính histogram cho biểu đồ cường độ Hue (kênh 0) thông qua vùng mask giới hạn, chia đều thành số bins (ví dụ 5)
    hist = cv2.calcHist([hsv], [0], mask, [bins], [0, 180])
    # Chuẩn hoá Min-Max cho biểu đồ về dải 0.0 -> 1.0 (nhằm loại bỏ yếu tố kích thước vật lí ảnh)
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    # Duỗi mảng 2D NumPy thành Python List tiêu chuẩn 1D
    return hist.flatten().tolist()

def extract_color(img: np.ndarray) -> list[float]:
    """
    Trích xuất tổng cộng 9 đặc trưng màu sắc (Color Features) từ ảnh chứa cây.
    
    Quy trình phân tích màu sắc:
    1. Chuyển đổi ảnh sang không gian màu HSV và tạo Mặt nạ (Mask) để tách vùng có cây khỏi nền đen.
    2. Nếu không tìm thấy cây trong Mask, trả về mảng 9 giá trị 0.0 mặc định.
    3. Lần lượt gọi các hàm thuật toán con để tính toán:
       - 2 giá trị Mean cơ bản: Hue trung bình, Saturation trung bình.
       - 1 biến thiên kênh màu: Biến thiên (lệch chuẩn) màu xanh lá trên tán cây.
       - 1 tỷ lệ các bộ phận: Tỷ lệ nhóm nhánh/thân cây/lá úa (Màu Nâu & Xám).
       - 5 đặc trưng phân bố màu: Biểu đồ Histogram chia 5 cụm (bins) trên kênh màu Hue.
       
    Args:
        img (np.ndarray): Ảnh OpenCV (BGR) của đoạn cắt chứa cái cây cần trích xuất đặc trưng.
        
    Returns:
        list[float]: Danh sách 9 giá trị đặc trưng màu sắc đã được chuẩn hóa.
    """
    # 1. Chuyển đổi hệ màu để dễ dàng tách nền và nắm bắt ánh sáng/mực màu rời rạc
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    from .utils import get_tree_mask
    mask = get_tree_mask(img)
    
    # 2. Xử lý ngoại lệ: Nếu ảnh đầu vào đen thui xì không có cây (mask trống)
    if cv2.countNonZero(mask) == 0:
        return [0.0] * 9
        
    # Lấy toàn bộ cường độ pixel của cây loại bỏ hoàn toàn background
    masked_pixels = hsv[mask > 0]
    
    # 3. Tính toán các thuộc tính chi tiết thông qua các module con tương ứng (phân tách trách nhiệm)
    mean_h, mean_s = _calculate_primary_metrics(masked_pixels)    # Phân tích Giá trị trung bình
    leaf_var = _calculate_leaf_color_variance(img, mask)            # Đọc kênh Xanh lá độc lập
    bg_ratio = _calculate_brown_gray_ratio(masked_pixels)           # Tỉ lệ Gỗ & Bóng tối
    hue_hist = _calculate_hue_histogram(hsv, mask)                  # Biến đổi thành dãy tần suất màu
    
    # 4. Gộp nhóm kết quả cho vào 1 list số thực duy nhất bảo toàn trật tự đúng với `extractor.py`    
    return [
        float(mean_h),
        float(mean_s),
        float(leaf_var),
        float(bg_ratio)
    ] + [float(x) for x in hue_hist]

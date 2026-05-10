import cv2
import numpy as np

def get_tree_mask(img: np.ndarray) -> np.ndarray:
    """
    Tạo mặt nạ (mask) để tách cây khỏi nền một cách thông minh.
    Hỗ trợ cả ảnh có nền đen (dataset chuẩn) và ảnh tự nhiên (Otsu adaptive).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Kiểm tra 4 góc ảnh để xem có phải nền đen không
    corners = [gray[0,0], gray[0,-1], gray[-1,0], gray[-1,-1]]
    if np.mean(corners) < 15:
        # Giả định nền đen: dùng Simple Threshold
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    else:
        # Ảnh tự nhiên: Dùng Otsu
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        # Thông thường cây tối hơn nền trời
        _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Kiểm tra lại: nếu diện tích trắng chiếm đa số (>70%), có thể bị ngược
        if cv2.countNonZero(mask) > (mask.size * 0.7):
            mask = cv2.bitwise_not(mask)
            
    return mask

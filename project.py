import cv2
import numpy as np
import matplotlib.pyplot as plt


def detect_road_lines(image_path):
    image = cv2.imread(image_path)
    result_image = np.copy(image)
    height, width = image.shape[:2]

    # تبدیل به فضاهای رنگی مختلف برای استخراج ویژگی بهتر
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # ایجاد ماسک‌های مختلف برای شناسایی بهتر خطوط سفید در شرایط نوری مختلف
    _, gray_thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    
    lower_white_hsv = np.array([0, 0, 200])
    upper_white_hsv = np.array([180, 30, 255])
    hsv_white_mask = cv2.inRange(hsv, lower_white_hsv, upper_white_hsv)
    
    combined_mask = cv2.bitwise_or(gray_thresh, hsv_white_mask)
    
    # ایجاد یک ذوزنقه برای تمرکز روی پایین تصویر
    roi_vertices = np.array([
        [(0, height), 
         (width * 0.35, height * 0.65), 
         (width * 0.60, height * 0.65), 
         (width, height)]
    ], dtype=np.int32)
    
    # ایجاد ماسک ناحیه مورد نظر
    roi_mask = np.zeros_like(combined_mask)
    cv2.fillPoly(roi_mask, roi_vertices, 255)
    
    # اعمال ماسک ناحیه مورد نظر
    masked_image = cv2.bitwise_and(combined_mask, roi_mask)
    
    # کاهش نویز و بهبود تصویر
    kernel = np.ones((3, 3), np.uint8)
    
    # حذف نویزهای کوچک
    opening = cv2.morphologyEx(masked_image, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # بستن شکاف‌های خطوط
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # یافتن کانتورها برای شناسایی نشانه‌های جاده
    contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # فیلتر کانتورها بر اساس مساحت و شکل
    min_area = 20 
    road_marking_contours = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                # نشانه‌های جاده معمولا کشیده هستند (دایره‌ای نیستند)
                if circularity < 0.6:
                    road_marking_contours.append(contour)

    # نمایش نتایج
    cv2.drawContours(result_image, road_marking_contours, -1, (0, 255, 0), 2)
    result_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
    
    return result_rgb
def show_results(original_path, processed_image):
    original = cv2.imread(original_path)
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(14, 7))
    
    plt.subplot(1, 2, 1)
    plt.imshow(original_rgb)
    plt.title('original', fontsize=14,)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(processed_image)
    plt.title('Road Lane Line Detection', fontsize=14)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
def show_processing_steps(image_path):
    image = cv2.imread(image_path)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    _, gray_thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    lower_white_hsv = np.array([0, 0, 200])
    upper_white_hsv = np.array([180, 30, 255])
    hsv_white_mask = cv2.inRange(hsv, lower_white_hsv, upper_white_hsv)
    
    combined_mask = cv2.bitwise_or(gray_thresh, hsv_white_mask)
    
    # تعریف ناحیه مورد نظر به صورت ذوزنقه‌ای
    height, width = image.shape[:2]
    roi_vertices = np.array([
        [(0, height),
         (width * 0.55, height * 0.85),
         (width * 0.85, height * 0.85),
         (width, height)]
    ], dtype=np.int32)
    
    roi_mask = np.zeros_like(combined_mask)
    cv2.fillPoly(roi_mask, roi_vertices, 255)
    
    masked_image = cv2.bitwise_and(combined_mask, roi_mask)
    
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(masked_image, cv2.MORPH_OPEN, kernel, iterations=1)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    image_path = "6.jpg"
    
    result = detect_road_lines(image_path)
    
    if result is not None:
        show_results(image_path, result)

        
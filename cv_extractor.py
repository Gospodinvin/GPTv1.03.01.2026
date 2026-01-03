import cv2
import numpy as np

def compute_quality(crop, num_candles):
    base_from_candles = min(num_candles / 50.0, 1.0)
    edges = cv2.Canny(crop, 30, 100)
    edge_density = edges.mean() / 255.0
    contrast = crop.std() / 255.0
    technical = (edge_density + contrast) / 2
    return round(0.6 * base_from_candles + 0.4 * technical, 2)

def dynamic_crop(img):
    edges = cv2.Canny(img, 30, 100)
    projection = np.sum(edges, axis=1)
    threshold = np.max(projection) * 0.03 if np.max(projection) > 0 else 0
    non_zero_rows = np.where(projection > threshold)[0]
    if len(non_zero_rows) == 0:
        return img
    top = max(0, non_zero_rows[0] - 40)
    bottom = min(img.shape[0], non_zero_rows[-1] + 40)
    w = img.shape[1]
    return img[top:bottom, int(w*0.02):int(w*0.98)]

def extract_candles(image_bytes, max_candles=60):
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Не удалось декодировать изображение")
    h, w = img.shape

    # Удаляем очевидные панели
    initial_crop = img[int(h*0.08):int(h*0.92), int(w*0.02):int(w*0.98)]

    # Сильное улучшение контраста
    blurred = cv2.GaussianBlur(initial_crop, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))  # Увеличен clipLimit
    enhanced = clahe.apply(blurred)

    # Динамический кроп вокруг свечей
    crop = dynamic_crop(enhanced)

    # Более чувствительная детекция краёв
    edges = cv2.Canny(crop, 30, 100)

    # Вертикальный kernel для тонких и высоких свечей
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 25))
    verticals = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    verticals = cv2.dilate(verticals, kernel, iterations=2)

    contours, _ = cv2.findContours(verticals, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ch, cw = crop.shape
    raw_candles = []

    for c in contours:
        x, y, w_, h_ = cv2.boundingRect(c)
        # Смягчённые фильтры для тонких свечей
        if h_ < ch * 0.04: continue  # слишком короткие
        if w_ > cw * 0.12: continue   # слишком широкие (горизонтальные линии/шум)
        if h_ / max(w_, 1) < 2.0: continue  # слишком жирные
        raw_candles.append({
            "x": x,
            "open": (y + h_ * 0.25) / ch,
            "close": (y + h_ * 0.75) / ch,
            "high": y / ch,
            "low": (y + h_) / ch,
        })

    # Критично: сортировка слева направо!
    raw_candles.sort(key=lambda c: c["x"])
    candles = [{k: v for k, v in c.items() if k != "x"} for c in raw_candles]

    quality = compute_quality(crop, len(candles))
    return candles[-max_candles:], quality

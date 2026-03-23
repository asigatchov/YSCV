import cv2
import numpy as np
import time

# Use all available threads (explicit, matches yscv rayon default).
# Set to 1 for single-threaded comparison, or 0 for OpenCV default.
cv2.setNumThreads(cv2.getNumberOfCPUs())

def bench(name, func, warmup=1, iters=100):
    for _ in range(warmup):
        func()
    best = float('inf')
    for _ in range(iters):
        t0 = time.perf_counter()
        func()
        t1 = time.perf_counter()
        elapsed = (t1 - t0) * 1000.0
        if elapsed < best:
            best = elapsed
    print(f"{name}: {best:.3f} ms")

# Inputs
img_480_640_3 = np.zeros((480, 640, 3), dtype=np.uint8)
img_480_640_1 = np.zeros((480, 640, 1), dtype=np.uint8)
img_1080_1920_3 = np.zeros((1080, 1920, 3), dtype=np.uint8)

kernel_3x3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

print(f"OpenCV version: {cv2.__version__}")
print(f"NumPy version: {np.__version__}")
print()

# 1. Grayscale
bench("grayscale 480x640x3", lambda: cv2.cvtColor(img_480_640_3, cv2.COLOR_BGR2GRAY))

# 2. Dilate
bench("dilate 3x3 480x640x1", lambda: cv2.dilate(img_480_640_1, kernel_3x3))

# 3. Erode
bench("erode 3x3 480x640x1", lambda: cv2.erode(img_480_640_1, kernel_3x3))

# 4. Gaussian blur
bench("gaussian_blur 3x3 480x640x1", lambda: cv2.GaussianBlur(img_480_640_1, (3, 3), 0))

# 5. Box blur
bench("box_blur 3x3 480x640x1", lambda: cv2.blur(img_480_640_1, (3, 3)))

# 6. Sobel magnitude
def sobel_mag():
    sx = cv2.Sobel(img_480_640_1, cv2.CV_16S, 1, 0, ksize=3)
    sy = cv2.Sobel(img_480_640_1, cv2.CV_16S, 0, 1, ksize=3)
    mag = cv2.convertScaleAbs(sx) + cv2.convertScaleAbs(sy)
    return mag

bench("sobel 3x3 480x640x1", sobel_mag)

# 7. Median blur
bench("median_blur 3x3 480x640x1", lambda: cv2.medianBlur(img_480_640_1, 3))

# 8. Canny
bench("canny 480x640x1", lambda: cv2.Canny(img_480_640_1, 30, 100))

# 9. Resize bilinear 1080x1920x3 -> 720x1280
bench("resize_bilinear 1080x1920x3->720x1280", lambda: cv2.resize(img_1080_1920_3, (1280, 720), interpolation=cv2.INTER_LINEAR))

# 10. Resize bilinear 480x640x1 -> 240x320
bench("resize_bilinear 480x640x1->240x320", lambda: cv2.resize(img_480_640_1, (320, 240), interpolation=cv2.INTER_LINEAR))

# --- f32 ops ---
print()
print("--- f32 ops ---")

img_f32_1 = np.zeros((480, 640, 1), dtype=np.float32)
img_f32_3 = np.zeros((480, 640, 3), dtype=np.float32)

# f32 Gaussian blur
bench("f32_gaussian_blur 3x3 480x640x1", lambda: cv2.GaussianBlur(img_f32_1, (3, 3), 0))

# f32 Box blur
bench("f32_box_blur 3x3 480x640x1", lambda: cv2.blur(img_f32_1, (3, 3)))

# f32 Sobel
def f32_sobel_mag():
    sx = cv2.Sobel(img_f32_1, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(img_f32_1, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(sx, sy)
bench("f32_sobel 3x3 480x640x1", f32_sobel_mag)

# f32 Resize
bench("f32_resize 480x640x3->240x320", lambda: cv2.resize(img_f32_3, (320, 240), interpolation=cv2.INTER_LINEAR))

# f32 Dilate
bench("f32_dilate 3x3 480x640x1", lambda: cv2.dilate(img_f32_1, kernel_3x3))

# f32 Threshold
bench("f32_threshold 480x640x1", lambda: cv2.threshold(img_f32_1, 0.5, 1.0, cv2.THRESH_BINARY))

# f32 Grayscale
bench("f32_grayscale 480x640x3", lambda: cv2.cvtColor(img_f32_3, cv2.COLOR_BGR2GRAY))

# --- imgproc feature / transform ops ---
print()
print("--- imgproc feature / transform ops ---")

# FAST corners
fast_detector = cv2.FastFeatureDetector_create(20)
bench("fast_corners 480x640", lambda: fast_detector.detect(img_480_640_1))

# ORB features
orb_detector = cv2.ORB_create(500)
bench("orb_detect 480x640", lambda: orb_detector.detectAndCompute(img_480_640_1, None))

# Histogram
bench("histogram 480x640", lambda: cv2.calcHist([img_480_640_1], [0], None, [256], [0, 256]))

# CLAHE
clahe_obj = cv2.createCLAHE(clipLimit=40.0, tileGridSize=(8, 8))
bench("clahe 480x640", lambda: clahe_obj.apply(img_480_640_1.reshape(480, 640)))

# Distance transform
# cv2.distanceTransform requires a binary image (CV_8U)
dt_input = np.zeros((480, 640), dtype=np.uint8)
bench("distance_transform 480x640", lambda: cv2.distanceTransform(dt_input, cv2.DIST_L2, 3))

# Warp perspective
M_persp = np.array([[0.99, -0.01, 5.0], [0.01, 0.99, 3.0], [0.0, 0.0, 1.0]], dtype=np.float32)
bench("warp_perspective 480x640", lambda: cv2.warpPerspective(img_480_640_1, M_persp, (640, 480)))

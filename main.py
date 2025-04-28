import os
import cv2
import argparse
import numpy as np


# Fixed perspective transformation coordinates 
def correct_perspective(img):
    "Performs a fixed perspective transformation on an image to correct perspective distortion"
    src_pts = np.float32([[10, 16], [233, 5], [247, 230], [31, 240]])
    dst_pts = np.float32([[0, 0], [255, 0], [255, 255], [0, 255]])
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(
        img, matrix, (256, 256), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )
    return warped


#  Gamma adjustment
def adjust_gamma(img, gamma=0.85):
    "Gamma correction adjusts the overall brightness of the image, reducing the risk of overexposure in highlight areas."
    inv_gamma = 1.0 / gamma
    table = np.array(
        [(i / 255.0) ** inv_gamma * 255 for i in range(256)]
    ).astype("uint8")
    return cv2.LUT(img, table)


# Automatic black circle detection (heuristic algorithm) 
def detect_black_circle_auto(img):
    "Auto-detects the top-right black circle using Hough Circles."  
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    roi = gray[0:h // 2, w // 3:]
    roi_blur = cv2.GaussianBlur(roi, (7, 7), 0)

    circles = cv2.HoughCircles(
        roi_blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
        param1=50, param2=20, minRadius=10, maxRadius=35
    )

    mask = np.zeros_like(gray)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            x_global = x + w // 3
            y_global = y
            if gray[y_global, x_global] < 60 and r < 40:
                cv2.circle(mask, (x_global, y_global), r + 2, 255, -1)
                break

    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    return mask


#  Gray world color balance (heuristic correction) 
def gray_world_balance(img):
    "Executes gray world assumption-based color normalization for cast reduction."
    b, g, r = cv2.split(img.astype('float32'))
    avg_gray = (np.mean(b) + np.mean(g) + np.mean(r)) / 3
    b = np.clip(b * (avg_gray / np.mean(b)), 0, 255)
    g = np.clip(g * (avg_gray / np.mean(g)), 0, 255)
    r = np.clip(r * (avg_gray / np.mean(r)), 0, 255)
    return cv2.merge((b, g, r)).astype('uint8')


#  Soft Denoising (Luma/Chroma Separate Denoising Strategy)
def soft_denoise_pipeline(img):
    "YCrCb-space denoising (separate luma/chroma processing)"
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    cr = cv2.GaussianBlur(cr, (7, 7), sigmaX=5)
    cb = cv2.GaussianBlur(cb, (7, 7), sigmaX=5)
    y = cv2.medianBlur(y, 3)
    merged = cv2.merge((y, cr, cb))
    result = cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)
    return result


# Image sharpening 
def sharpen_image(img):
    "Detail enhancement using conservative sharpening (noise-aware)"
    kernel = np.array([[0, -0.5, 0],
                       [-0.5, 3, -0.5],
                       [0, -0.5, 0]])
    return cv2.filter2D(img, -1, kernel)


# pipeline
def process_pipeline(img):
    "Whole pipeline including all functions."
    img = correct_perspective(img)
    mask = detect_black_circle_auto(img)
    if np.any(mask):
        img = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)

    img = soft_denoise_pipeline(img)
    img = adjust_gamma(img, gamma=0.85)
    img = gray_world_balance(img)

    img = cv2.fastNlMeansDenoisingColored(
        img, None, h=4, hColor=4, templateWindowSize=7, searchWindowSize=21
    )

    img = sharpen_image(img)
    return img


# Main
def main(input_dir):
    "Save the processed images to Results"
    output_dir = 'Results'
    os.makedirs(output_dir, exist_ok=True)

    for filename in sorted(os.listdir(input_dir)):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(input_dir, filename)
            img = cv2.imread(img_path)

            if img is None:
                print(f"Unable to read image: {filename}")
                continue

            processed = process_pipeline(img)
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, processed)
            print(f"Processed: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image enhancement pipeline")
    parser.add_argument("input_dir", type=str, help="Path to input image folder")
    args = parser.parse_args()
    main(args.input_dir)
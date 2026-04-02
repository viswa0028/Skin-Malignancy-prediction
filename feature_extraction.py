import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, "ISIC2018_Task3_Training_Input")
METADATA_CSV = os.path.join(BASE_DIR, "ISIC2018_binary_labels.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "skin_tone_features2.csv")
RESIZE_DIM = (512, 512)
try:
    metadata = pd.read_csv(METADATA_CSV)
except Exception as e:
    raise SystemExit(f"Failed to read metadata CSV at {METADATA_CSV}: {e}")

results = []


def extract_skin_mask(img):
    """
    Returns a binary mask where skin pixels are 1 and others are 0
    """
    # Convert image to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Skin range in HSV (these limits can be tuned)
    lower = np.array([0, 20, 70], dtype=np.uint8)
    upper = np.array([50, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower, upper)
    return mask


for idx, row in tqdm(metadata.iterrows(), total=len(metadata)):
    image_id = row.get("image")
    if pd.isna(image_id) or str(image_id).strip() == "":
        continue
    image_id = str(image_id)
    img_path = os.path.join(IMAGE_DIR, image_id + ".jpg")

    # skip if image not found
    if not os.path.exists(img_path):
        continue

    # Read image
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        # try common alternative extensions
        for ext in (".png", ".jpeg", ".bmp"):
            alt = os.path.join(IMAGE_DIR, image_id + ext)
            img = cv2.imread(alt, cv2.IMREAD_COLOR)
            if img is not None:
                img_path = alt
                break
    if img is None:
        continue

    # Optional resizing for speed
    img = cv2.resize(img, RESIZE_DIM)

    # ensure 3-channel BGR (some images may be grayscale)
    if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Extract skin mask
    skin_mask = extract_skin_mask(img)

    # If there are no skin pixels, skip
    skin_pixels = img[skin_mask > 0]
    if len(skin_pixels) == 0:
        continue

    # --- COLOR FEATURES ---

    # Average color
    avg_color = np.mean(skin_pixels, axis=0)

    # Median color
    med_color = np.median(skin_pixels, axis=0)

    # Standard deviation
    std_color = np.std(skin_pixels, axis=0)

    # Convert to HSV and LAB
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # Skin pixels in HSV/LAB
    hsv_skin = hsv_img[skin_mask > 0]
    lab_skin = lab_img[skin_mask > 0]

    # Brightness (V channel)
    brightness_mean = np.mean(hsv_skin[:, 2])

    # Hue & Saturation
    hue_mean = np.mean(hsv_skin[:, 0])
    saturation_mean = np.mean(hsv_skin[:, 1])

    # Lightness (L in LAB)
    lightness_mean = np.mean(lab_skin[:, 0])

    # Skin pixel ratio
    skin_ratio = np.sum(skin_mask > 0) / (img.shape[0]*img.shape[1])

    # Edge density (texture info)
    edges = cv2.Canny(img, 100, 200)
    edge_density = np.sum(edges > 0) / edges.size

    # Save features
    results.append({
        "image": image_id,
        "avg_R": avg_color[2],
        "avg_G": avg_color[1],
        "avg_B": avg_color[0],
        "med_R": med_color[2],
        "med_G": med_color[1],
        "med_B": med_color[0],
        "std_R": std_color[2],
        "std_G": std_color[1],
        "std_B": std_color[0],
        "brightness_mean": brightness_mean,
        "hue_mean": hue_mean,
        "saturation_mean": saturation_mean,
        "lightness_mean": lightness_mean,
        "skin_pixel_ratio": skin_ratio,
        "edge_density": edge_density
    })

features_df = pd.DataFrame(results)
# Merge features with original metadata to retain 'malignant' or other original columns
final_df = pd.merge(metadata, features_df, on="image", how="inner")
final_df.to_csv(OUTPUT_CSV, index=False)

print("Skin tone features extracted & saved to:", OUTPUT_CSV)

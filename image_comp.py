import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import imagehash
from PIL import Image

def calculate_color_score(img1, img2):
    hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
    hist1 = cv2.calcHist([hsv1], [0, 1], None, [50, 60], [0, 180, 0, 256])
    hist2 = cv2.calcHist([hsv2], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist1, hist1)
    cv2.normalize(hist2, hist2)
    score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return round(score * 100, 2)

def calculate_ssim_score(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.resize(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), (gray1.shape[1], gray1.shape[0]))
    score, _ = ssim(gray1, gray2, full=True)
    return round(score * 100, 2)

def calculate_feature_match_score(img1, img2):
    orb = cv2.ORB_create(nfeatures=1000)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    if des1 is None or des2 is None:
        return 0.0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    good_matches = [m for m in matches if m.distance < 50]
    score = len(good_matches) / max(len(matches), 1)
    return round(score * 100, 2)

def calculate_edge_similarity(img1, img2):
    gray1 = cv2.Canny(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), 50, 150)
    gray2 = cv2.Canny(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), 50, 150)
    gray2 = cv2.resize(gray2, (gray1.shape[1], gray1.shape[0]))
    overlap = np.sum(gray1 == gray2)
    total = gray1.shape[0] * gray1.shape[1]
    return round((overlap / total) * 100, 2)

def calculate_hash_similarity(img1, img2):
    img1 = Image.fromarray(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    img2 = Image.fromarray(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    hash1 = imagehash.average_hash(img1)
    hash2 = imagehash.average_hash(img2)
    return round((1 - (hash1 - hash2) / len(hash1.hash)**2) * 100, 2)

def compare_images(img_path1, img_path2):
    img1 = cv2.imread(img_path1)
    img2 = cv2.imread(img_path2)

    if img1 is None or img2 is None:
        print("❌ Error: One or both images could not be loaded.")
        return

    print("\n🔍 Comparing images...\n")

    color_score = calculate_color_score(img1, img2)
    ssim_score = calculate_ssim_score(img1, img2)
    feature_score = calculate_feature_match_score(img1, img2)
    edge_score = calculate_edge_similarity(img1, img2)
    hash_score = calculate_hash_similarity(img1, img2)

    weights = {
        "color": 0.075,
        "ssim": 0.3,
        "feature": 0.025, 
        "edge": 0.3,
        "hash": 0.3
    }

    overall_score = round(
        color_score * weights["color"] +
        ssim_score * weights["ssim"] +
        feature_score * weights["feature"] +
        edge_score * weights["edge"] +
        hash_score * weights["hash"], 2
    )

    print(f"🎨 Color Similarity        : {color_score}%")
    print(f"🧱 Structural Similarity   : {ssim_score}%")
    print(f"🧬 Feature Matching        : {feature_score}%")
    print(f"🔳 Edge Similarity         : {edge_score}%")
    print(f"🧠 Hash (Perceptual) Match : {hash_score}%")
    print(f"\n🎯 Overall Similarity Score: {overall_score}%")

# Example usage portion::--::--::______::
img_1=r"C:\Users\vsaga\Downloads\ai-generated-cat-and-dog-together-with-happy-expressions-ai-generated-free-photo.jpg"
img_2=r"C:\Users\vsaga\Downloads\ai-generated-cat-and-dog-together-with-happy-expressions-ai-generated-photo.jpg"
img_3=r"C:\Users\vsaga\Downloads\ai-generated-cat-and-dog-together-with-happy-expressions-on-yellow-background-ai-generated-free-photo.jpg"
img_4=r"C:\Users\vsaga\Downloads\dog-and-cat-together-pets-spring-or-summer-nature-generative-ai-photo.jpg"
img_5=r"C:\Users\vsaga\.vscode\prompto\close-up-dog-field_662214-129629.jpg"
img_6=r"C:\Users\vsaga\.vscode\prompto\669726ef5242a23882952518_663fc2a1da49d30b9a44e774_image_3cN5ZzSm_1715403464233_raw.jpeg"
img_7=r"C:\Users\vsaga\.vscode\prompto\I-made-visual-dog-puns-with-an-AI-art-generator-63881c119c954-png__880.jpg"
img_8=r"C:\Users\vsaga\.vscode\prompto\images.jpeg"
compare_images(img_7, img_3) #replace with your image paths
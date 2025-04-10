
# Change the import order to put torch first
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Import torch before any other local imports
import torch
import cv2
import supervision as sv

# Create project directories
HOME = os.path.join(os.getcwd(), "sam_project")
os.makedirs(HOME, exist_ok=True)
os.makedirs(os.path.join(HOME, "weights"), exist_ok=True)
os.makedirs(os.path.join(HOME, "data"), exist_ok=True)

print("Project directory:", HOME)

# Install required packages if not already installed
def install_package(package_name):
    try:
        __import__(package_name)
        print(f"{package_name} is already installed.")
    except ImportError:
        print(f"Installing {package_name}...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])

# Install segment-anything from GitHub if not already installed
try:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    print("segment-anything is already installed.")
except ImportError:
    print("Installing segment-anything...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", 'git+https://github.com/facebookresearch/segment-anything.git'])
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# Install other dependencies
packages = ["jupyter_bbox_widget", "roboflow", "dataclasses-json", "supervision"]
for package in packages:
    install_package(package)

# Download SAM model weights
CHECKPOINT_PATH = os.path.join(HOME, "weights", "sam_vit_h_4b8939.pth")

if not os.path.isfile(CHECKPOINT_PATH):
    print("Downloading SAM model weights...")
    import urllib.request
    urllib.request.urlretrieve(
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        CHECKPOINT_PATH
    )
    print("Model weights downloaded.")
else:
    print("Model weights already downloaded.")

# Download sample images
image_urls = {
    "times_square.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e9/Times_Sq_Feb_2017_4.jpg/640px-Times_Sq_Feb_2017_4.jpg",
    "liberty_island.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6b/Liberty_Island_photo_Don_Ramey_Logan.jpg/640px-Liberty_Island_photo_Don_Ramey_Logan.jpg",
    "dubai.jpg": "https://upload.wikimedia.org/wikipedia/commons/4/43/Downtown_Dubai%2C_Dubai%2C_United_Arab_Emirates.jpg"

}

for image_name, url in image_urls.items():
    image_path = os.path.join(HOME, "data", image_name)
    if not os.path.isfile(image_path):
        print(f"Downloading {image_name}...")
        import urllib.request
        urllib.request.urlretrieve(url, image_path)
        print(f"{image_name} downloaded.")
    else:
        print(f"{image_name} already downloaded.")

# Set up the device (GPU if available, otherwise CPU)
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# Load the SAM model
MODEL_TYPE = "vit_h"
print("Loading SAM model...")
sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)
mask_generator = SamAutomaticMaskGenerator(sam)
print("SAM model loaded.")

# Process images
def process_image(image_path, title):
    print(f"Processing {image_path}...")
    
    # Load and convert image
    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    # Generate masks
    print("Generating segmentation masks...")
    sam_result = mask_generator.generate(image_rgb)
    print(f"Generated {len(sam_result)} masks.")
    
    # Convert SAM results to Supervision detection format
    mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
    detections = sv.Detections.from_sam(sam_result=sam_result)
    
    # Create annotated image
    annotated_image = mask_annotator.annotate(scene=image_bgr.copy(), detections=detections)
    
    # Display results
    plt.figure(figsize=(18, 8))
    
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    plt.title(f"Original Image: {title}")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
    plt.title(f"Segmented Image: {title}")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return sam_result

# Process the sample images
IMAGE_PATH1 = os.path.join(HOME, "data", "times_square.jpg")
IMAGE_PATH2 = os.path.join(HOME, "data", "liberty_island.jpg")
IMAGE_PATH3 = os.path.join(HOME, "data", "dubai_resized.jpg")

sam_result1 = process_image(IMAGE_PATH1, "Times Square")
sam_result2 = process_image(IMAGE_PATH2, "Liberty Island")
#sam_result3 = process_image(IMAGE_PATH3, "Dubai Island")

print("All processing complete!")

# Optional: Print information about the segmentation results
print("\nSegmentation Result Structure:")
print(sam_result1[0].keys())
print(f"\nTotal segments in Times Square image: {len(sam_result1)}")
print(f"Total segments in Liberty Island image: {len(sam_result2)}")
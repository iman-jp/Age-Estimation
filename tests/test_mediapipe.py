
from masking.mediapipe_init import media_pipe
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# --- Process and visualize ---
def draw_landmarks(image_path):
    # --- 1. Ask the user for their preference ---
    face_landmarker,pose_landmarker = media_pipe()
    print("\n--- Display Options ---")
    print("1: Only the changed version")
    print("2: Both side-by-side")
    choice = input("Enter your choice (1 or 2): ").strip()

    # Load image with PIL
    pil_image = Image.open(image_path).convert('RGB')
    image_np = np.array(pil_image)

    # Create MediaPipe image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_np)

    # Run detection
    face_result = face_landmarker.detect(mp_image)
    pose_result = pose_landmarker.detect(mp_image)

    annotated = image_np.copy()
    h, w = annotated.shape[:2]
    
    # --- 2. Draw Face Landmarks (Green) ---
    if face_result.face_landmarks:
        for lm in face_result.face_landmarks[0]:
            x, y = int(lm.x * w), int(lm.y * h)
            # Small 2x2 green dots
            annotated[max(0,y-1):min(h,y+1), max(0,x-1):min(w,x+1)] = [0, 255, 0]

    # --- 3. Draw Pose Landmarks (Red) ---
    if pose_result.pose_landmarks:
        print("\nProcessing Pose Landmarks...")
        for idx, lm in enumerate(pose_result.pose_landmarks[0]):
            if lm.visibility > 0.8:
                # 1. Convert normalized (0-1) to pixel coordinates first
                x, y = int(lm.x * w), int(lm.y * h)
                
                # 2. NOW print those pixel coordinates!
                print(f"✅ Landmark {idx} is visible!")
                print(f"   Pixel Loc: X -> {x}, Y -> {y} (Normalized: {lm.x:.4f}, {lm.y:.4f})")
                
                # 3. Draw the red square on the image
                y_min, y_max = max(0, y - 3), min(h, y + 3)
                x_min, x_max = max(0, x - 3), min(w, x + 3)
                annotated[y_min:y_max, x_min:x_max] = [255, 0, 0]
            else:
                print(f"❌ Skipping {idx} (Visibility: {lm.visibility:.2f})")

    # --- 4. Plotting based on user choice ---
    if choice == '2':
        # Side-by-side mode
        fig, axes = plt.subplots(1, 2, figsize=(15, 8))
        
        axes[0].imshow(image_np)
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        axes[1].imshow(annotated)
        axes[1].set_title('Face (green) + Pose (red)')
        axes[1].axis('off')
    else:
        # Single version mode (Default)
        plt.figure(figsize=(10, 8))
        plt.imshow(annotated)
        plt.title('Processed Landmarks')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# --- Test ---
# draw_landmarks(r'..\utkface_images\105_1_0_20170112213507183.jpg')
draw_landmarks(r'C:\Users\imanj\Desktop\Age-Estimation\utkface_images\26_0_0_20170117144510833.jpg')

from mediapipe_init import media_pipe
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image



def draw_landmarks(image_path):
    face_landmarker,pose_landmarker = media_pipe()
    print("\n--Display Options---")
    print("1: Only the changed version")
    print("2: Both side-by-side")
    choice = input("Enter your choice (1 or 2): ").strip()

    
    pil_image = Image.open(image_path).convert('RGB')
    image_np = np.array(pil_image)

    
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_np)

    
    face_result = face_landmarker.detect(mp_image)
    pose_result = pose_landmarker.detect(mp_image)

    annotated = image_np.copy()
    h, w = annotated.shape[:2]
    
    
    if face_result.face_landmarks:
        for lm in face_result.face_landmarks[0]:
            x, y = int(lm.x * w), int(lm.y * h)
            annotated[max(0,y-1):min(h,y+1), max(0,x-1):min(w,x+1)] = [0, 255, 0]

    
    if pose_result.pose_landmarks:
        print("\nProcessing Pose Landmarks...")
        for idx, lm in enumerate(pose_result.pose_landmarks[0]):
            if lm.visibility > 0.8:
                
                x, y = int(lm.x * w), int(lm.y * h)
                
               
                print(f" Landmark {idx} is visible!")
                print(f"   Pixel Loc: X -> {x}, Y -> {y} (Normalized: {lm.x:.4f}, {lm.y:.4f})")
                
                
                y_min, y_max = max(0, y - 3), min(h, y + 3)
                x_min, x_max = max(0, x - 3), min(w, x + 3)
                annotated[y_min:y_max, x_min:x_max] = [255, 0, 0]
            else:
                print(f" Skipping {idx} (Visibility: {lm.visibility:.2f})")

    
    if choice == '2':
        
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


# draw_landmarks(r'..\utkface_images\105_1_0_20170112213507183.jpg')
# draw_landmarks(r'\Desktop\Age-Estimation\utkface_images\26_0_0_20170117144510833.jpg')
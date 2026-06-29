from mediapipe_init import media_pipe
from body_part import BodyPartMask
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

# --- Face mesh landmark indices for specific regions ---




def get_region_mask(indices:list,w:int,h:int,landmarks):
    """
    return a array of the given indices
    :param indices : the section we want to extract as a list of num
    :param w : wide of the matplot
    :param h : height of the matplot
    :return : array of the given segment
    """
    mask = Image.new('L', (w, h), 0)
    draw = ImageDraw.Draw(mask)
    points = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices]
    draw.polygon(points, fill=255)
    return np.array(mask)

def apply_mask(image, mask):
    result = image.copy()
    result[mask == 0] = 0
    return result


def extract_face_regions(image_path):
    # Load image
    face_landmarker,pose_landmarker = media_pipe()
    pil_image = Image.open(image_path).convert('RGB')
    image_np = np.array(pil_image)
    h, w = image_np.shape[:2]

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_np)
    face_result = face_landmarker.detect(mp_image)

    if not face_result.face_landmarks:
        print("No face detected")
        return

    landmarks = face_result.face_landmarks[0]

    

    
    print( BodyPartMask.LEFT_EYE)
    print( BodyPartMask.LEFT_EYE.value)
    # Extract each region
    left_eye_mask   = get_region_mask(BodyPartMask.LEFT_EYE.value,w,h,landmarks)
    right_eye_mask  = get_region_mask(BodyPartMask.RIGHT_EYE.value,w,h,landmarks)
    nose_mask       = get_region_mask(BodyPartMask.NOSE.value,w,h,landmarks)
    lips_mask       = get_region_mask(BodyPartMask.LIPS.value,w,h,landmarks)

    left_eye_img    = apply_mask(image_np, left_eye_mask)
    right_eye_img   = apply_mask(image_np, right_eye_mask)
    nose_img        = apply_mask(image_np, nose_mask)
    lips_img        = apply_mask(image_np, lips_mask)

    # Plot
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    titles = ['Original', 'Left Eye', 'Right Eye', 'Nose', 'Lips']
    images = [image_np, left_eye_img, right_eye_img, nose_img, lips_img]

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')

    plt.tight_layout()
    plt.show()

extract_face_regions(r'..\utkface_images\105_1_0_20170112213507183.jpg')
# extract_face_regions(r'C:\Users\imanj\Desktop\Age-Estimation\utkface_images\26_0_0_20170117144510833.jpg')
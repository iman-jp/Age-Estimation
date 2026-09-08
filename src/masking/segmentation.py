from masking.mediapipe_init import media_pipe
from masking.body_part import BodyPartMask
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import cv2




def fall_back(pose_result, face_landmarker, image_np, w: int, h: int):
    if not pose_result or not pose_result.pose_landmarks:
        return None,None

    pose_landmarks = pose_result.pose_landmarks[0]
            
    #key facial landmarks in MediaPipe Pose
    head_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    head_xs = [pose_landmarks[i].x * w for i in head_indices if pose_landmarks[i].visibility > 0.3]
    head_ys = [pose_landmarks[i].y * h for i in head_indices if pose_landmarks[i].visibility > 0.3]

    if not head_xs or not head_ys:
        return None,None

    center_x = (min(head_xs) + max(head_xs)) / 2
    center_y = (min(head_ys) + max(head_ys)) / 2

    raw_width = max(head_xs) - min(head_xs)
    raw_height = max(head_ys) - min(head_ys)
    
    
    side_length = max(raw_width, raw_height) * 1.4

    crop_min_x = max(0, int(center_x - side_length / 2))
    crop_max_x = min(w, int(center_x + side_length / 2))
    crop_min_y = max(0, int(center_y - side_length / 2))
    crop_max_y = min(h, int(center_y + side_length / 2))

    
    cropped_np = image_np[crop_min_y:crop_max_y, crop_min_x:crop_max_x]
    #print("we have cropped and before the first if")
    if cropped_np.size > 0:
    
        target_size = 512  
        crop_h, crop_w = cropped_np.shape[:2]

        if crop_h < target_size or crop_w < target_size:
            cropped_np = cv2.resize(
                cropped_np, 
                (target_size, target_size), 
                interpolation=cv2.INTER_CUBIC
            )

        
        #Image.fromarray(cropped_np).show()

        
        cropped_mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(cropped_np))
        
        face_result = face_landmarker.detect(cropped_mp_image)
        #print(face_result.face_landmarks)
        if face_result and face_result.face_landmarks:
            # print("Face successfully detected in cropped head region!")
            return (face_result,cropped_np)

    return (None,None)

def get_region_mask(indices:list,w:int,h:int,landmarks):
    """
    return a array of the given indices
    :param indices : the section we want to extract as a list of num
    :param w : wide of the matplot
    :param h : height of the matplot
    :param landmarks : all the point made by mediapipe
    :return : array of the given segment
    """
    mask = Image.new('L', (w, h), 0)
    draw = ImageDraw.Draw(mask)
    points = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices]
    draw.polygon(points, fill=255)
    return np.array(mask)

def apply_mask(image, mask):
    result = image.copy()
    result[mask == 255] = 0
    return result

def extract_face_regions(image_input):
    
    face_landmarker,pose_landmarker = media_pipe()
    if isinstance(image_input, Image.Image):
        pil_image = image_input.convert('RGB')
    else:
        pil_image = Image.open(image_input).convert('RGB')
    image_np = np.array(pil_image)
    h, w = image_np.shape[:2]

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_np)
    face_result = face_landmarker.detect(mp_image)

    if not face_result or not face_result.face_landmarks:
        print("Direct face detection failed. Triggering pose fallback...")
        pose_result = pose_landmarker.detect(mp_image)
    
        face_result,image_np = fall_back(pose_result, face_landmarker, image_np, w, h)

        if not face_result or not face_result.face_landmarks:
                print("No face detected after fallback attempts.")
                return None

        h, w = image_np.shape[:2]

   
        
        

    landmarks = face_result.face_landmarks[0]
    
    
    left_eye_mask   = get_region_mask(BodyPartMask.LEFT_EYE.value,w,h,landmarks)
    right_eye_mask  = get_region_mask(BodyPartMask.RIGHT_EYE.value,w,h,landmarks)
    eyes_e_mask   = get_region_mask(BodyPartMask.EYES_E.value,w,h,landmarks)
    nose_mask       = get_region_mask(BodyPartMask.NOSE.value,w,h,landmarks)
    lips_mask       = get_region_mask(BodyPartMask.LIPS.value,w,h,landmarks)
    chin_mask       = get_region_mask(BodyPartMask.CHIN.value,w,h,landmarks)
    bulls_eye_mask   = get_region_mask(BodyPartMask.BULLS_EYE.value,w,h,landmarks)
    upper_mask       = get_region_mask(BodyPartMask.UPPER_FACE.value,w,h,landmarks)
    lower_mask       = get_region_mask(BodyPartMask.LOWER_FACE.value,w,h,landmarks)
    


    
    #dont forget need to use the eyses_img in combo with left and right eys img
    left_eye_img    = apply_mask(image_np, left_eye_mask)
    right_eye_img   = apply_mask(image_np, right_eye_mask)
    eyes_e_img    = apply_mask(image_np, eyes_e_mask)
    eyses_img   = apply_mask(left_eye_img, right_eye_mask)
    nose_img        = apply_mask(image_np, nose_mask)
    lips_img        = apply_mask(image_np, lips_mask)
    chin_img        = apply_mask(image_np,chin_mask)
    bulls_eye_img    = apply_mask(image_np, bulls_eye_mask)
    upper_img    = apply_mask(image_np, upper_mask)
    lower_mask_img    = apply_mask(image_np, lower_mask)
    
    

    
    fig, axes = plt.subplots(1, 7, figsize=(18, 4))
    titles = ['Original', 'Left Eye', 'Right Eye', 'Nose', 'Lips','Chin','eyes']
    images = [image_np, left_eye_img, right_eye_img, nose_img, lips_img, chin_img, eyses_img]
    #images = [image_np, eyses_img, nose_img, lips_img, chin_img]

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')

    plt.tight_layout()
    plt.show()

# extract_face_regions(r'..\utkface_images\105_1_0_20170112213507183.jpg')
extract_face_regions(r"C:\Users\imanj\Desktop\Age-Estimation\data\utkface_images\25_1_0_20170104021710995.jpg")
#extract_face_regions(r"C:\Users\imanj\Pictures\Screenshots\Screenshot 2026-09-07 195310.png")
#extract_face_regions(r"C:\Users\imanj\Pictures\Camera Roll\WIN_20260903_14_02_38_Pro.jpg")
#extract_face_regions(r"C:\Users\imanj\Pictures\Camera Roll\WIN_20260903_14_02_43_Pro.jpg")
# extract_face_regions(r'C:\Users\imanj\Desktop\Age-Estimation\utkface_images\21_1_1_20170116214444631.jpg')
# extract_face_regions(r'C:\Users\imanj\Desktop\Age-Estimation\utkface_images\21_0_1_20170116030053264.jpg')

"""
this script is used for masking data with given folder as input and put them the output path
.along the way there is also log function that showes the faild attempt.
"""
from pathlib import Path
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw
import cv2
from mediapipe_init import media_pipe
from body_part import BodyPartMask
from failurelog import FailureLog


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_ROOT = PROJECT_ROOT / "data" / "splits"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "masked" / "bulls_eye_masked"
VARIANT_NAME = "bulls_eye_masked"
LOG_PATH = PROJECT_ROOT / "data" / "masking_bulls_eye_masked_log.csv"

SPLIT_NAMES = ("train", "val", "test")

VARIANT_DEFINITIONS = {
    "left_eye_masked":    [BodyPartMask.LEFT_EYE.value],
    "right_eye_masked":   [BodyPartMask.RIGHT_EYE.value],
    "eyes_masked":        [BodyPartMask.LEFT_EYE.value, BodyPartMask.RIGHT_EYE.value],
    "eyes_e_masked":      [BodyPartMask.EYES_E.value],
    "nose_masked":        [BodyPartMask.NOSE.value],
    "lips_masked":        [BodyPartMask.LIPS.value],
    "chin_masked":        [BodyPartMask.CHIN.value],
    "bulls_eye_masked":   [BodyPartMask.BULLS_EYE.value],
    "upper_face_masked":  [BodyPartMask.UPPER_FACE.value],
    "lower_face_masked":  [BodyPartMask.LOWER_FACE.value],
}



def fall_back(pose_result, face_landmarker, image_np, w: int, h: int):
    """
    this function trigger when face could not be detected so it will try to zoom in the picture and call mediapipe detection again
        args:
            pose_result: list of the pose landmark. used to found the face boundry
            face_landmark : used to call face detection after crop in
            image_np: image as a numpy array
            w: width of image
            h:height of image
        return: 
            None,None if function fail
            face_result,cropped_np if function successed: return list of face landmarks and the cropped image numpy array
    """
    
    if not pose_result or not pose_result.pose_landmarks:
        return None, None

    pose_landmarks = pose_result.pose_landmarks[0]

    head_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    head_xs = [pose_landmarks[i].x * w for i in head_indices if pose_landmarks[i].visibility > 0.3]
    head_ys = [pose_landmarks[i].y * h for i in head_indices if pose_landmarks[i].visibility > 0.3]

    if not head_xs or not head_ys:
        return None, None

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
    if cropped_np.size == 0:
        return None, None

    target_size = 512
    crop_h, crop_w = cropped_np.shape[:2]
    if crop_h < target_size or crop_w < target_size:
        cropped_np = cv2.resize(cropped_np, (target_size, target_size), interpolation=cv2.INTER_CUBIC)

    cropped_mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(cropped_np))
    face_result = face_landmarker.detect(cropped_mp_image)

    if face_result and face_result.face_landmarks:
        return face_result, cropped_np

    return None, None


def get_region_mask(indices: list, w: int, h: int, landmarks):
    """
    this function crop out the given point in indices
        args:
            indices : the section we want to extract as a list of num
            w : wide of the matplot
            h : height of the matplot
            landmarks : all the point made by mediapipe
        return : 
            array of the given segment
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


def detect_face_landmarks(image_path, face_landmarker, pose_landmarker):
    """
    this function is used to distinguished between the method that mediapipe found the face or not at
    all.
    args:
        image_path: image path that currently is being investigated.
        face_landmarkder: mediapipe landmarker for detecting the face.
        pose_landmarker: mediapipe landmarker that will be used inside the fall_back function.
    returns:
        three possible value : direct-> showing face founded without need to call fallback function
        fallback-> show the state that the face could not be found the first go around and fall back function called and
        it was sucsessfull.
        failed -> showes that mediapipe could not found any face even after fallback
    """
    pil_image = Image.open(image_path).convert('RGB')
    image_np = np.array(pil_image)
    h, w = image_np.shape[:2]

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_np)
    face_result = face_landmarker.detect(mp_image)

    if face_result and face_result.face_landmarks:
        return "direct", image_np, face_result.face_landmarks[0], w, h

    pose_result = pose_landmarker.detect(mp_image)
    face_result, cropped_np = fall_back(pose_result, face_landmarker, image_np, w, h)

    if face_result and face_result.face_landmarks:
        h, w = cropped_np.shape[:2]
        return "fallback", cropped_np, face_result.face_landmarks[0], w, h

    return "failed", image_np, None, w, h


def apply_masking_variant(image_np, landmarks, w: int, h: int, indices_groups: list):
    """
    function to add mulitply mask together.
    """
    combined_mask = np.zeros((h, w), dtype=np.uint8)
    for indices in indices_groups:
        region_mask = get_region_mask(indices, w, h, landmarks)
        combined_mask = np.maximum(combined_mask, region_mask)
    return apply_mask(image_np, combined_mask)


def mask_dataset(input_root, output_root, variant_name, indices_groups, log_path):
    """
    function to iterate through the whole folder and put the copy in the output folder in case of sucess and 
    log them if mediapipe could not detect a face
    """
    input_root = Path(input_root)
    output_root = Path(output_root)
    face_landmarker, pose_landmarker = media_pipe()

    with FailureLog(log_path) as log:
        for split_name in SPLIT_NAMES:
            split_in = input_root / split_name
            split_out = output_root / split_name

            if not split_in.exists():
                print(f"Skipping '{split_name}': {split_in} does not exist")
                continue

            split_out.mkdir(parents=True, exist_ok=True)

            for image_path in split_in.iterdir():
                if not image_path.is_file():
                    continue

                status, image_np, landmarks, w, h = detect_face_landmarks(
                    image_path, face_landmarker, pose_landmarker
                )

                if status == "failed":
                    log.log_failure(image_path.name, split_name, variant_name)
                    continue

                masked_np = apply_masking_variant(image_np, landmarks, w, h, indices_groups)
                Image.fromarray(masked_np).save(split_out / image_path.name)




def main():
    mask_dataset(
        input_root=INPUT_ROOT,
        output_root=OUTPUT_ROOT,
        variant_name=VARIANT_NAME,
        indices_groups=VARIANT_DEFINITIONS[VARIANT_NAME],
        log_path=LOG_PATH,
    )


if __name__ == "__main__":
    main()
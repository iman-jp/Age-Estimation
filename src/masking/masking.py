import csv
from pathlib import Path
from datetime import datetime
import argparse
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw
import cv2

from mediapipe_init import media_pipe
from body_part import BodyPartMask


def fall_back(pose_result, face_landmarker, image_np, w: int, h: int):
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
    combined_mask = np.zeros((h, w), dtype=np.uint8)
    for indices in indices_groups:
        region_mask = get_region_mask(indices, w, h, landmarks)
        combined_mask = np.maximum(combined_mask, region_mask)
    return apply_mask(image_np, combined_mask)


class FailureLog:
    def __init__(self, log_path, flush_every: int = 100):
        self.log_path = Path(log_path)
        self.flush_every = flush_every
        self._buffer = []
        is_new_file = not self.log_path.exists() or self.log_path.stat().st_size == 0
        self._file = open(self.log_path, mode='a', newline='', encoding='utf-8')
        self._writer = csv.writer(self._file)
        if is_new_file:
            self._writer.writerow(["image_name", "split", "variant", "reason", "timestamp"])

    def log_failure(self, image_name, split, variant, reason="no_face_detected"):
        self._buffer.append([
            image_name, split, variant, reason,
            datetime.now().isoformat(timespec='seconds'),
        ])
        if len(self._buffer) >= self.flush_every:
            self._flush()

    def _flush(self):
        if not self._buffer:
            return
        self._writer.writerows(self._buffer)
        self._file.flush()
        self._buffer.clear()

    def close(self):
        self._flush()
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


SPLIT_NAMES = ("train", "val", "test")


def mask_dataset(input_root, output_root, variant_name, indices_groups, log_path):
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


VARIANT_DEFINITIONS = {
    "eyes_masked":        [BodyPartMask.LEFT_EYE.value, BodyPartMask.RIGHT_EYE.value],
    "eyes_e_masked":      [BodyPartMask.EYES_E.value],
    "nose_masked":        [BodyPartMask.NOSE.value],
    "lips_masked":        [BodyPartMask.LIPS.value],
    "chin_masked":        [BodyPartMask.CHIN.value],
    "bulls_eye_masked":   [BodyPartMask.BULLS_EYE.value],
    "upper_face_masked":  [BodyPartMask.UPPER_FACE.value],
    "lower_face_masked":  [BodyPartMask.LOWER_FACE.value],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--variant", required=True, choices=sorted(VARIANT_DEFINITIONS.keys()))
    parser.add_argument("--log", default="masking_log.csv")
    args = parser.parse_args()

    mask_dataset(
        input_root=args.input,
        output_root=args.output,
        variant_name=args.variant,
        indices_groups=VARIANT_DEFINITIONS[args.variant],
        log_path=args.log,
    )


if __name__ == "__main__":
    main()
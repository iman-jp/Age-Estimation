import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def media_pipe():
    face_base_options = python.BaseOptions(model_asset_path=r'C:\Users\imanj\Desktop\Age-Estimation\mediapipe/face_landmarker.task')
    face_options = vision.FaceLandmarkerOptions(
        base_options=face_base_options,
        num_faces=1,
        min_face_detection_confidence=0.5
    )
    face_landmarker = vision.FaceLandmarker.create_from_options(face_options)

    pose_base_options = python.BaseOptions(model_asset_path=r'C:\Users\imanj\Desktop\Age-Estimation\mediapipe/pose_landmarker_heavy.task')
    pose_options = vision.PoseLandmarkerOptions(
        base_options=pose_base_options,
        min_pose_detection_confidence=0.5
    )
    pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)
    return (face_landmarker,pose_landmarker)
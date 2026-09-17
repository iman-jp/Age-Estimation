"""this script is to use the mediapipe and calling it for the face detection"""
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FACE_MODEL_PATH = PROJECT_ROOT / "mediapipe" / "face_landmarker.task"
POSE_MODEL_PATH = PROJECT_ROOT / "mediapipe" / "pose_landmarker_heavy.task"

def media_pipe():
    """function to initalize mediapipe with the given configuraion
    configuration that can be change are :  num_faces-> max number of faces that model would detect;
                                            min_face_presence_confidence-> The minimum confidence score of face presence score in the face landmark detection.
                                            min_face_detection_confidence->The minimum confidence score for the face detection to be considered successful.
                                            min_pose_detection_confidence->The minimum confidence score for the pose detection to be considered successful
    for more inforamtion about on mediapipe: http://developers.google.com/edge/mediapipe
    returns:
        face landmark and pose landmark in a tuple
    
    """
    face_base_options = python.BaseOptions(model_asset_path=str(FACE_MODEL_PATH))
    face_options = vision.FaceLandmarkerOptions(
        base_options=face_base_options,
        num_faces=1,
        min_face_presence_confidence=0.5,
        min_face_detection_confidence=0.5
    )
    face_landmarker = vision.FaceLandmarker.create_from_options(face_options)

    pose_base_options = python.BaseOptions(model_asset_path=str(POSE_MODEL_PATH))
    pose_options = vision.PoseLandmarkerOptions(
        base_options=pose_base_options,
        min_pose_detection_confidence=0.5
    )
    pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)
    return (face_landmarker, pose_landmarker)
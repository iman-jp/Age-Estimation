from enum import Enum

class BodyPartMask(Enum):
    """
    in this class we have all the body part for the mediapipe segmentaion that we can use to mask the picture
    """
    LEFT_EYE = [463, 341, 256, 252, 253, 254, 339, 255, 359, 467, 260, 259, 257, 258, 288, 411]
    RIGHT_EYE = [130, 25, 110, 24, 23, 22, 26, 112, 243, 190, 56, 28, 27, 29, 30, 247]
    NOSE = [1, 2, 98, 327, 168, 197, 195, 5]
    LIPS = [57, 43, 106, 182, 83, 18, 313, 406, 335, 273, 287, 410, 322, 391, 393, 164, 167, 165, 92, 186]
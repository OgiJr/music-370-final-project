import math
import os

os.environ["GLOG_minloglevel"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
import mediapipe as mp
from mediapipe.tasks import python as _mp_python
from mediapipe.tasks.python import vision as _mp_vision
import numpy as np

_DEFAULT_MODEL = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../assets/", "hand_landmarker.task"
)

# pairs of landmark indices that should be connected when drawing the hand skeleton
HAND_CONNECTIONS = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
)


class HandDetector:
    def __init__(
        self,
        model_path: str = _DEFAULT_MODEL,
        num_hands: int = 2,
        detection_confidence: float = 0.70,
        presence_confidence: float = 0.60,
        tracking_confidence: float = 0.60,
    ):
        opts = _mp_vision.HandLandmarkerOptions(
            base_options=_mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=_mp_vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=detection_confidence,
            min_hand_presence_confidence=presence_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self._landmarker = _mp_vision.HandLandmarker.create_from_options(opts)

    def detect(self, bgr_frame: np.ndarray, timestamp_ms: int) -> tuple:
        fh, fw = bgr_frame.shape[:2]
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = self._landmarker.detect_for_video(mp_img, timestamp_ms)

        right: tuple = (None, False, None)
        left: tuple = (None, 0, None)

        # we flip the frame horizontally before this so mediapipe's "Left" is actually
        # the user's right hand. yes this confused me for a whole afternoon
        for i, lms in enumerate(res.hand_landmarks):
            is_right = bool(
                res.handedness
                and len(res.handedness) > i
                and res.handedness[i]
                and res.handedness[i][0].category_name == "Left"
            )
            palm = self._palm_center(lms, fw, fh)
            if is_right:
                right = (palm, self._is_fist(lms), lms)
            else:
                left = (palm, self.finger_count(lms), lms)

        return right, left

    def draw_skeleton(
        self,
        frame: np.ndarray,
        lms,
        fw: int,
        fh: int,
        bone_color: tuple,
        joint_color: tuple,
    ) -> None:
        pts = [(int(lm.x * fw), int(lm.y * fh)) for lm in lms]
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], bone_color, 2, cv2.LINE_AA)
        for px, py in pts:
            cv2.circle(frame, (px, py), 4, joint_color, -1, cv2.LINE_AA)

    def close(self) -> None:
        self._landmarker.close()

    @staticmethod
    def finger_count(lms) -> int:
        # finger is "extended" if its tip is higher (smaller y) than its knuckle
        # ignores thumb because thumb geometry is different
        tips = [8, 12, 16, 20]
        mcps = [5, 9, 13, 17]
        return sum(1 for tip, mcp in zip(tips, mcps) if lms[tip].y < lms[mcp].y)

    @staticmethod
    def full_finger_count(lms) -> int:
        # same as above but counts thumb if it's far enough from the palm
        tips = [8, 12, 16, 20]
        mcps = [5, 9, 13, 17]
        count = sum(1 for tip, mcp in zip(tips, mcps) if lms[tip].y < lms[mcp].y)
        dx = lms[4].x - lms[9].x
        dy = lms[4].y - lms[9].y
        if math.sqrt(dx * dx + dy * dy) > 0.12:
            count += 1
        return count

    @staticmethod
    def all_fingertips_visible(lms) -> bool:
        for idx in (4, 8, 12, 16, 20):
            lm = lms[idx]
            if not (0.0 <= lm.x <= 1.0 and 0.0 <= lm.y <= 1.0):
                return False
        return True

    @staticmethod
    def is_pinch(lms, threshold: float = 0.055) -> bool:
        # thumb tip + index tip close together
        dx = lms[4].x - lms[8].x
        dy = lms[4].y - lms[8].y
        return math.sqrt(dx * dx + dy * dy) < threshold

    @staticmethod
    def pinch_point(lms, fw: int, fh: int) -> tuple:
        return (
            int((lms[4].x + lms[8].x) / 2 * fw),
            int((lms[4].y + lms[8].y) / 2 * fh),
        )

    @staticmethod
    def is_twist(lms, threshold: float = 0.07) -> bool:
        # like pinch but with middle finger
        dx = lms[4].x - lms[12].x
        dy = lms[4].y - lms[12].y
        return math.sqrt(dx * dx + dy * dy) < threshold

    @staticmethod
    def twist_angle(lms) -> float:
        # angle from wrist (0) to middle MCP (9). used to track wrist rotation
        return math.atan2(lms[9].y - lms[0].y, lms[9].x - lms[0].x)

    @staticmethod
    def is_shaka(lms) -> bool:
        # 🤙 thumb out, pinky out, middle 3 curled
        three_down = (
            lms[8].y > lms[6].y and lms[12].y > lms[10].y and lms[16].y > lms[14].y
        )
        pinky_up = lms[20].y < lms[17].y
        dx = lms[4].x - lms[9].x
        dy = lms[4].y - lms[9].y
        thumb_out = math.sqrt(dx * dx + dy * dy) > 0.10
        return three_down and pinky_up and thumb_out

    def draw_pinch(
        self, frame: np.ndarray, lms, fw: int, fh: int, pinching: bool
    ) -> None:
        t = (int(lms[4].x * fw), int(lms[4].y * fh))
        ix = (int(lms[8].x * fw), int(lms[8].y * fh))
        col = (83, 172, 249) if pinching else (127, 22, 148)
        cv2.line(frame, t, ix, col, 2, cv2.LINE_AA)
        mid = self.pinch_point(lms, fw, fh)
        cv2.circle(frame, mid, 8 if pinching else 5, col, -1, cv2.LINE_AA)
        if pinching:
            cv2.circle(frame, mid, 14, col, 2, cv2.LINE_AA)

    def draw_twist(
        self, frame: np.ndarray, lms, fw: int, fh: int, grabbed: bool
    ) -> None:
        t = (int(lms[4].x * fw), int(lms[4].y * fh))
        m = (int(lms[12].x * fw), int(lms[12].y * fh))
        col = (83, 172, 249) if grabbed else (127, 22, 148)
        cv2.line(frame, t, m, col, 2, cv2.LINE_AA)
        mid = ((t[0] + m[0]) // 2, (t[1] + m[1]) // 2)
        cv2.circle(frame, mid, 8 if grabbed else 5, col, -1, cv2.LINE_AA)
        if grabbed:
            cv2.circle(frame, mid, 14, col, 2, cv2.LINE_AA)

    @staticmethod
    def _is_fist(lms) -> bool:
        # at least 3 of 4 fingers curled = fist (one finger of slack so it doesn't flicker)
        tips = [8, 12, 16, 20]
        mcps = [5, 9, 13, 17]
        return sum(1 for tip, mcp in zip(tips, mcps) if lms[tip].y > lms[mcp].y) >= 3

    @staticmethod
    def _palm_center(lms, fw: int, fh: int) -> tuple:
        # average of wrist + 4 finger MCPs. more stable than just the wrist landmark
        keys = [0, 5, 9, 13, 17]
        x = sum(lms[i].x for i in keys) / len(keys)
        y = sum(lms[i].y for i in keys) / len(keys)
        return int(x * fw), int(y * fh)

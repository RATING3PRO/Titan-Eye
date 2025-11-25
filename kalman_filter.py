import numpy as np


class KalmanFilter:
    """
    简单的边界框常速 Kalman，状态: [cx, cy, w, h, vx, vy, vw, vh]
    """

    def __init__(self, dt: float = 1.0):
        self.dt = dt
        self._build_matrices()
        self.x = np.zeros((8, 1))
        self.P = np.eye(8) * 10.0
        self.is_initialized = False

    def _build_matrices(self):
        dt = self.dt
        self.A = np.eye(8)
        for i in range(4):
            self.A[i, i + 4] = dt

        self.H = np.zeros((4, 8))
        self.H[0, 0] = 1  # cx
        self.H[1, 1] = 1  # cy
        self.H[2, 2] = 1  # w
        self.H[3, 3] = 1  # h

        # 噪声可按需要调优
        self.Q = np.eye(8) * 1e-2
        self.R = np.eye(4) * 1e0

    def init_state(self, bbox_xyxy: np.ndarray):
        cx, cy, w, h = self._xyxy_to_cxcywh(bbox_xyxy)
        self.x[:4, 0] = np.array([cx, cy, w, h])
        self.is_initialized = True

    def predict(self) -> np.ndarray | None:
        if not self.is_initialized:
            return None
        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q
        return self.get_state()

    def update(self, bbox_xyxy: np.ndarray) -> np.ndarray:
        if not self.is_initialized:
            self.init_state(bbox_xyxy)
            return self.get_state()

        z = np.array(self._xyxy_to_cxcywh(bbox_xyxy)).reshape(4, 1)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        y = z - (self.H @ self.x)
        self.x = self.x + K @ y
        I = np.eye(8)
        self.P = (I - K @ self.H) @ self.P
        return self.get_state()

    def get_state(self) -> np.ndarray:
        cx, cy, w, h = self.x[:4, 0]
        return self._cxcywh_to_xyxy((cx, cy, w, h))

    @staticmethod
    def _xyxy_to_cxcywh(bbox):
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w / 2
        cy = y1 + h / 2
        return cx, cy, w, h

    @staticmethod
    def _cxcywh_to_xyxy(c):
        cx, cy, w, h = c
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return np.array([x1, y1, x2, y2])

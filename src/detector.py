import logging
import time
from typing import Any

import cv2
import numpy as np

try:
    from deepface import DeepFace
except ModuleNotFoundError as exc:  # pragma: no cover
    if exc.name and exc.name.split(".")[0] != "deepface":
        raise
    DeepFace = None

try:
    from retinaface import RetinaFace
except ModuleNotFoundError as exc:  # pragma: no cover
    if exc.name and exc.name.split(".")[0] != "retinaface":
        raise
    RetinaFace = None

try:
    import mlflow

    MLFLOW_AVAILABLE = True
except ImportError:
    mlflow = None
    MLFLOW_AVAILABLE = False

logger = logging.getLogger(__name__)


class FaceDetector:
    """Wrapper for RetinaFace detection, alignment, and DeepFace verification."""

    def __init__(
        self,
        threshold: float = 0.9,
        verification_model: str = "ArcFace",
        use_mlflow: bool = False,
    ):
        self.threshold = threshold
        self.verification_model = verification_model
        self.use_mlflow = use_mlflow and MLFLOW_AVAILABLE

        if self.use_mlflow:
            mlflow.set_experiment("RetinaFace_Inference_Monitoring")

    def _require_retinaface(self):
        if RetinaFace is None:
            raise ImportError("retina-face package is required for face detection.")
        return RetinaFace

    def _require_deepface(self):
        if DeepFace is None:
            raise ImportError("deepface package is required for identity verification.")
        return DeepFace

    def verify(self, img1_path: str, img2_path: str) -> dict[str, Any]:
        """Verify whether two images contain the same person."""
        deepface = self._require_deepface()
        start_time = time.time()
        try:
            result = deepface.verify(
                img1_path=img1_path,
                img2_path=img2_path,
                model_name=self.verification_model,
                detector_backend="retinaface",
                enforce_detection=False,
                align=True,
            )
        except Exception as exc:
            raise RuntimeError(f"DeepFace verification failed: {exc}") from exc

        latency_ms = (time.time() - start_time) * 1000
        if self.use_mlflow:
            self._log_verification_metrics(result, latency_ms)

        return result

    def detect(self, image_input: str | np.ndarray) -> dict[str, Any]:
        """Detect faces and 5-point landmarks in an image path or BGR array."""
        retinaface = self._require_retinaface()
        start_time = time.time()
        results = retinaface.detect_faces(image_input, threshold=self.threshold)
        latency_ms = (time.time() - start_time) * 1000

        if not results or not isinstance(results, dict):
            if self.use_mlflow:
                self._log_detection_metrics(0, latency_ms, 0.0)
            return {}

        num_faces = len(results)
        avg_confidence = float(
            np.mean([data.get("score", 0.0) for data in results.values()])
        )
        if self.use_mlflow:
            self._log_detection_metrics(num_faces, latency_ms, avg_confidence)

        return results

    def extract_faces(
        self, image_input: str | np.ndarray, align: bool = True
    ) -> list[np.ndarray]:
        """Extract and optionally align detected face crops."""
        retinaface = self._require_retinaface()
        return retinaface.extract_faces(
            img_path=image_input, threshold=self.threshold, align=align
        )

    def _log_detection_metrics(
        self, count: int, latency_ms: float, confidence: float
    ) -> None:
        try:
            with mlflow.start_run(run_name="Detection_Inference", nested=True):
                mlflow.log_param("threshold", self.threshold)
                mlflow.log_metric("face_count", count)
                mlflow.log_metric("latency_ms", latency_ms)
                mlflow.log_metric("avg_confidence", confidence)
        except Exception as exc:
            logger.warning("MLflow detection logging failed: %s", exc)

    def _log_verification_metrics(
        self, result: dict[str, Any], latency_ms: float
    ) -> None:
        try:
            with mlflow.start_run(run_name="Verification_Task", nested=True):
                mlflow.log_param("model", self.verification_model)
                mlflow.log_metric("latency_ms", latency_ms)
                mlflow.log_metric("verified", 1 if result.get("verified") else 0)
                mlflow.log_metric("distance", result.get("distance", 0.0))
        except Exception as exc:
            logger.warning("MLflow verification logging failed: %s", exc)

    @staticmethod
    def draw_results(image: np.ndarray, results: dict[str, Any]) -> np.ndarray:
        """Draw bounding boxes, scores, and 5-point landmarks on a copy of image."""
        output_img = image.copy()

        for face_id, data in results.items():
            area = data.get("facial_area", [])
            score = data.get("score", 0.0)

            if len(area) == 4:
                x1, y1, x2, y2 = (int(v) for v in area)
                cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{face_id} ({score:.2f})"
                cv2.putText(
                    output_img,
                    label,
                    (x1, max(y1 - 10, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

            landmarks = data.get("landmarks", {})
            for coords in landmarks.values():
                cv2.circle(
                    output_img, (int(coords[0]), int(coords[1])), 2, (0, 0, 255), -1
                )

        return output_img

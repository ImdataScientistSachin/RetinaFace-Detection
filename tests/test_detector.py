from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.detector import FaceDetector


@pytest.fixture
def sample_detection():
    return {
        "face_1": {
            "score": 0.992,
            "facial_area": [20, 25, 80, 95],
            "landmarks": {
                "right_eye": [38.0, 48.0],
                "left_eye": [62.0, 48.0],
                "nose": [50.0, 62.0],
                "mouth_right": [40.0, 78.0],
                "mouth_left": [60.0, 78.0],
            },
        }
    }


def test_detector_initialization_defaults():
    detector = FaceDetector()
    assert detector.threshold == 0.9
    assert detector.verification_model == "ArcFace"
    assert detector.use_mlflow is False


def test_detect_returns_results_and_forwards_threshold(sample_detection):
    detector = FaceDetector(threshold=0.85)
    mock_rf = MagicMock()
    mock_rf.detect_faces.return_value = sample_detection

    with patch("src.detector.RetinaFace", mock_rf):
        results = detector.detect("dummy.jpg")

    mock_rf.detect_faces.assert_called_once_with("dummy.jpg", threshold=0.85)
    assert "face_1" in results
    assert results["face_1"]["facial_area"] == [20, 25, 80, 95]


def test_detect_accepts_numpy_array(sample_detection):
    detector = FaceDetector(threshold=0.9)
    frame = np.zeros((120, 120, 3), dtype=np.uint8)
    mock_rf = MagicMock()
    mock_rf.detect_faces.return_value = sample_detection

    with patch("src.detector.RetinaFace", mock_rf):
        results = detector.detect(frame)

    mock_rf.detect_faces.assert_called_once_with(frame, threshold=0.9)
    assert len(results) == 1


@pytest.mark.parametrize("empty_return", [{}, (), None])
def test_detect_handles_no_faces(empty_return):
    detector = FaceDetector()
    mock_rf = MagicMock()
    mock_rf.detect_faces.return_value = empty_return

    with patch("src.detector.RetinaFace", mock_rf):
        results = detector.detect(np.zeros((64, 64, 3), dtype=np.uint8))

    assert results == {}


def test_extract_faces_forwards_threshold_and_align():
    detector = FaceDetector(threshold=0.75)
    crop = np.zeros((112, 112, 3), dtype=np.uint8)
    mock_rf = MagicMock()
    mock_rf.extract_faces.return_value = [crop]

    with patch("src.detector.RetinaFace", mock_rf):
        faces = detector.extract_faces("face.jpg", align=False)

    mock_rf.extract_faces.assert_called_once_with(
        img_path="face.jpg", threshold=0.75, align=False
    )
    assert len(faces) == 1
    assert faces[0].shape == (112, 112, 3)


def test_verify_returns_deepface_payload():
    detector = FaceDetector(verification_model="ArcFace")
    payload = {
        "verified": True,
        "distance": 0.21,
        "threshold": 0.68,
        "model": "ArcFace",
    }
    mock_df = MagicMock()
    mock_df.verify.return_value = payload

    with patch("src.detector.DeepFace", mock_df):
        result = detector.verify("a.jpg", "b.jpg")

    mock_df.verify.assert_called_once_with(
        img1_path="a.jpg",
        img2_path="b.jpg",
        model_name="ArcFace",
        detector_backend="retinaface",
        enforce_detection=False,
        align=True,
    )
    assert result["verified"] is True
    assert result["model"] == "ArcFace"


def test_verify_wraps_backend_errors():
    detector = FaceDetector()
    mock_df = MagicMock()
    mock_df.verify.side_effect = ValueError("corrupt image")

    with patch("src.detector.DeepFace", mock_df):
        with pytest.raises(RuntimeError, match="DeepFace verification failed"):
            detector.verify("a.jpg", "b.jpg")


def test_draw_results_draws_on_copy_without_mutating_input(sample_detection):
    img = np.zeros((120, 120, 3), dtype=np.uint8)
    annotated = FaceDetector.draw_results(img, sample_detection)

    assert annotated.shape == img.shape
    assert np.array_equal(img, np.zeros((120, 120, 3), dtype=np.uint8))
    assert not np.array_equal(img, annotated)


def test_draw_results_handles_empty_results():
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    annotated = FaceDetector.draw_results(img, {})

    assert annotated is not img
    assert np.array_equal(img, annotated)

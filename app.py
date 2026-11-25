import os
import tempfile

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from src.detector import FaceDetector

st.set_page_config(
    page_title="RetinaFace Detection & Verification",
    layout="wide",
)


@st.cache_resource
def get_detector(verification_model: str = "ArcFace") -> FaceDetector:
    return FaceDetector(threshold=0.9, verification_model=verification_model)


def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Mode",
        ["Face Detection", "Identity Verification", "System Info"],
    )

    st.sidebar.divider()
    st.sidebar.subheader("Settings")
    threshold = st.sidebar.slider("Detection Threshold", 0.1, 1.0, 0.9, 0.05)

    detector = get_detector()
    detector.threshold = threshold

    if page == "Face Detection":
        render_detection_page(detector)
    elif page == "Identity Verification":
        render_verification_page(detector)
    else:
        render_info_page()


def render_detection_page(detector: FaceDetector):
    st.title("Multi-Face Detection & Extraction")
    st.info(
        "Upload one or more images to detect faces, localize landmarks, "
        "and extract aligned portraits."
    )

    uploaded_files = st.file_uploader(
        "Choose Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True
    )

    if not uploaded_files:
        return

    for uploaded_file in uploaded_files:
        st.divider()
        st.subheader(f"File: {uploaded_file.name}")

        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if image is None:
            st.error(f"Could not decode image: {uploaded_file.name}")
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        col1, col2 = st.columns(2)

        with col1:
            st.image(image_rgb, caption="Original Image", use_container_width=True)

        with st.spinner(f"Analyzing {uploaded_file.name}..."):
            results = detector.detect(image)
            annotated_image = detector.draw_results(image, results)
            annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

        with col2:
            st.image(
                annotated_image_rgb,
                caption="Processed Image (Detection)",
                use_container_width=True,
            )
            st.success(f"Found {len(results)} face(s).")

        if results:
            with st.expander(f"View Extracted Portraits for {uploaded_file.name}"):
                faces = detector.extract_faces(image)
                if not faces:
                    st.info("No face crops returned.")
                else:
                    cols = st.columns(min(len(faces), 6))
                    for idx, face in enumerate(faces):
                        with cols[idx % 6]:
                            st.image(
                                face,
                                caption=f"Face {idx + 1}",
                                use_container_width=True,
                            )


def render_verification_page(detector: FaceDetector):
    st.title("Identity Verification (1-to-Many)")
    st.info(
        "Compare a reference image against one or more test images "
        "using ArcFace embeddings."
    )

    col_ref, col_test = st.columns(2)

    with col_ref:
        st.subheader("1. Reference Image")
        ref_file = st.file_uploader(
            "Upload reference image", type=["jpg", "jpeg", "png"], key="ref"
        )
        if ref_file:
            st.image(ref_file, width=200, caption="Reference Identity")

    with col_test:
        st.subheader("2. Test Images")
        test_files = st.file_uploader(
            "Upload test images",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="tests",
        )

    if not (ref_file and test_files):
        return

    st.divider()
    if not st.button("Run Identity Comparison"):
        return

    results_data = []
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_ref:
        tmp_ref.write(ref_file.getvalue())
        ref_path = tmp_ref.name

    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        for idx, test_file in enumerate(test_files):
            status_text.text(f"Verifying {test_file.name}...")

            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_test:
                tmp_test.write(test_file.getvalue())
                test_path = tmp_test.name

            try:
                res = detector.verify(ref_path, test_path)
                match_status = "MATCH" if res["verified"] else "NO MATCH"
                threshold_val = res.get("threshold") or 1.0
                confidence = (1.0 - (res["distance"] / threshold_val)) * 100
                results_data.append(
                    {
                        "File": test_file.name,
                        "Status": match_status,
                        "Distance": f"{res['distance']:.4f}",
                        "Confidence": f"{confidence:.2f}%",
                        "Model": res["model"],
                    }
                )
            except Exception as exc:
                results_data.append(
                    {
                        "File": test_file.name,
                        "Status": f"Error: {exc}",
                        "Distance": "N/A",
                        "Confidence": "N/A",
                        "Model": "N/A",
                    }
                )
            finally:
                if os.path.exists(test_path):
                    os.unlink(test_path)

            progress_bar.progress((idx + 1) / len(test_files))

        st.subheader("Comparison Summary")
        st.table(pd.DataFrame(results_data))

        st.subheader("Result Preview")
        preview_cols = st.columns(min(len(test_files), 5))
        for idx, test_file in enumerate(test_files):
            with preview_cols[idx % 5]:
                st.image(
                    test_file,
                    caption=results_data[idx]["Status"],
                    use_container_width=True,
                )
    finally:
        if os.path.exists(ref_path):
            os.unlink(ref_path)
        status_text.empty()
        st.success("Verification complete.")


def render_info_page():
    st.title("System Information")
    st.markdown(
        """
    ### Tech Stack
    - **Face Detection:** RetinaFace (ResNet50 + Feature Pyramid Network)
    - **Identity Verification:** DeepFace (ArcFace embeddings)
    - **Backend:** TensorFlow / Keras, OpenCV
    - **UI:** Streamlit

    ### Architecture
    - **Core Wrapper:** `src/detector.py` (`FaceDetector`)
    - **CLI:** `main.py` (`detect` and `verify` subcommands)
    - **Deployment:** Dockerized for Hugging Face Spaces (port 7860)
    """
    )


if __name__ == "__main__":
    main()

import streamlit as st
import cv2
import numpy as np
import os
import tempfile
import pandas as pd
from src.detector import FaceDetector

# --- Page Configuration ---
st.set_page_config(
    page_title="RatinaFace Pro - Multi-Face AI Suite",
    page_icon="🤖",
    layout="wide"
)

# Initialize detector globally
detector = FaceDetector(threshold=0.9, verification_model="ArcFace")

def main():
    # --- Sidebar Navigation (Better Visibility) ---
    st.sidebar.title("🚀 Navigation")
    page = st.sidebar.radio(
        "Select Functionality",
        ["🔍 Face Detection", "🛡️ Identity Verification", "ℹ️ System Info"]
    )
    
    st.sidebar.divider()
    st.sidebar.subheader("Settings")
    threshold = st.sidebar.slider("Detection Threshold", 0.1, 1.0, 0.9, 0.05)
    detector.threshold = threshold

    if page == "🔍 Face Detection":
        render_detection_page()
    elif page == "🛡️ Identity Verification":
        render_verification_page()
    else:
        render_info_page()

def render_detection_page():
    st.title("🔍 Multi-Face Detection & Extraction")
    st.info("Upload one or **multiple** images to detect faces, identify landmarks, and extract aligned portraits.")

    # --- File Upload (Multi-Select) ---
    uploaded_files = st.file_uploader(
        "Choose Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            st.divider()
            st.subheader(f"🖼️ File: {uploaded_file.name}")
            
            # Processing
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            col1, col2 = st.columns([1, 1])

            with col1:
                st.image(image_rgb, caption="Original Image", use_container_width=True)

            with st.spinner(f"Analyzing {uploaded_file.name}..."):
                results = detector.detect(image)
                annotated_image = detector.draw_results(image, results)
                annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

            with col2:
                st.image(annotated_image_rgb, caption="Processed Image (Detection)", use_container_width=True)
                st.success(f"Found {len(results)} faces in this image.")

            # --- Extraction ---
            if results:
                with st.expander(f"View Extracted Portraits for {uploaded_file.name}"):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                        cv2.imwrite(tmp_file.name, image)
                        tmp_path = tmp_file.name
                    
                    try:
                        faces = detector.extract_faces(tmp_path)
                        cols = st.columns(min(len(faces), 6))
                        for idx, face in enumerate(faces):
                            with cols[idx % 6]:
                                st.image(face, caption=f"Face {idx+1}", use_container_width=True)
                    finally:
                        if os.path.exists(tmp_path):
                            os.unlink(tmp_path)

def render_verification_page():
    st.title("🛡️ Identity Verification (1-to-Many)")
    st.info("Compare a **Reference Image** against one or more **Test Images** to verify identity matches.")

    col_ref, col_test = st.columns(2)

    with col_ref:
        st.subheader("1. Reference Image")
        ref_file = st.file_uploader("Upload image of the known person", type=["jpg", "jpeg", "png"], key="ref")
        if ref_file:
            st.image(ref_file, width=200, caption="Reference Identity")

    with col_test:
        st.subheader("2. Test Images (Group/Multiple)")
        test_files = st.file_uploader("Upload images to verify against reference", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="tests")

    if ref_file and test_files:
        st.divider()
        if st.button("🚀 Run Identity Comparison Suite"):
            results_data = []
            
            # Save Reference to Temp
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
                        # DeepFace Verification
                        res = detector.verify(ref_path, test_path)
                        match_emoji = "✅ MATCH" if res["verified"] else "❌ NO MATCH"
                        results_data.append({
                            "File": test_file.name,
                            "Status": match_emoji,
                            "Distance": f"{res['distance']:.4f}",
                            "Confidence": f"{ (1 - (res['distance']/res['threshold'])) * 100:.2f}%",
                            "Model": res["model"]
                        })
                    except Exception:
                        results_data.append({"File": test_file.name, "Status": "⚠️ Error", "Distance": "N/A", "Confidence": "N/A", "Model": "N/A"})
                    finally:
                        if os.path.exists(test_path):
                            os.unlink(test_path)
                    
                    progress_bar.progress((idx + 1) / len(test_files))

                # Display Results Table
                st.subheader("📊 Comparison Summary")
                df = pd.DataFrame(results_data)
                st.table(df)

                # Visual Thumbnails
                st.subheader("🖼️ Result Preview")
                preview_cols = st.columns(min(len(test_files), 5))
                for idx, test_file in enumerate(test_files):
                    with preview_cols[idx % 5]:
                        st.image(test_file, caption=results_data[idx]["Status"], use_container_width=True)

            finally:
                if os.path.exists(ref_path):
                    os.unlink(ref_path)
                status_text.empty()
                st.success("Verification Pipeline Complete.")

def render_info_page():
    st.title("ℹ️ System Information")
    st.markdown("""
    ### 🧬 Tech Stack
    - **Face Detection:** RetinaFace (Multi-stage ResNet50)
    - **Identity Verification:** DeepFace (ArcFace Architecture)
    - **Backend:** TensorFlow 2.13 / Keras
    - **UI:** Streamlit 1.25+
    
    ### 📦 Architecture
    - **Modular Design:** Core logic in `src/detector.py`.
    - **Industrial Ready:** Optimized for Batch and Real-time high-throughput analysis.
    - **Deployment:** Pre-configured Docker for Hugging Face Spaces.
    """)

if __name__ == "__main__":
    main()

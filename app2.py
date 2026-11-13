
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from ultralytics import YOLO
from pathlib import Path
import cv2
import tempfile
import os
import io
from collections import Counter, defaultdict


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="SteelSight AI | Defect Inspection",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_PATH = "best (1).pt"


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown(
    """
    <style>

    /* Main background - unchanged */
    .stApp {
        background: linear-gradient(
            135deg, #07111f 0%, #101d30 100%
        );
        color: #F1F5F9;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #0b1626;
        border-right: 1px solid #23344d;
    }

    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] li,
    [data-testid="stSidebar"] label {
        color: #E2E8F0 !important;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #FFFFFF !important;
    }

    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        color: #B8CBE0 !important;
    }

    /* Main text */
    .stApp h1,
    .stApp h2,
    .stApp h3,
    .stApp h4,
    .stApp h5,
    .stApp h6 {
        color: #FFFFFF !important;
    }

    .stApp p,
    .stApp label,
    .stApp li {
        color: #E2E8F0;
    }

    /* Hero section */
    .hero {
        padding: 28px;
        border-radius: 18px;
        background: linear-gradient(
            120deg, #14263e, #203a58
        );
        border: 1px solid #2d4a69;
        margin-bottom: 24px;
    }

    .hero h1 {
        color: #FFFFFF !important;
        font-size: 38px;
        margin-bottom: 8px;
    }

    .hero p {
        color: #DCE7F5 !important;
        font-size: 16px;
    }

    /* Metric cards */
    .metric-card {
        background: #122238;
        border: 1px solid #29415e;
        padding: 18px;
        border-radius: 14px;
        text-align: center;
    }

    .metric-label {
        color: #CBD5E1 !important;
        font-size: 14px;
    }

    .metric-value {
        color: #60A5FA !important;
        font-size: 30px;
        font-weight: 700;
    }

    /* Section headings */
    .section-title {
        color: #FFFFFF !important;
        font-size: 22px;
        font-weight: 600;
        margin-top: 14px;
        margin-bottom: 12px;
    }

    /* Buttons */
    div.stButton > button {
        background: linear-gradient(
            90deg, #2563eb, #0891b2
        );
        color: #FFFFFF !important;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        padding: 10px 18px;
        width: 100%;
    }

    div.stButton > button span {
        color: #FFFFFF !important;
    }

    div.stButton > button:hover {
        border: 1px solid #93c5fd;
        color: #FFFFFF !important;
    }

    /* Upload labels and slider */
    [data-testid="stFileUploader"] label,
    [data-testid="stSlider"] label {
        color: #F1F5F9 !important;
    }

    /* Info and success messages */
    [data-testid="stAlert"] p {
        color: #E2E8F0 !important;
    }

    /* Tab labels */
    .stTabs [data-baseweb="tab"] {
        color: #E2E8F0 !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.markdown("## 🏭 SteelSight AI")
    st.caption("AI-powered surface inspection")

    st.divider()

    st.markdown("### Model Information")
    st.write("**Architecture:** YOLO26")
    st.write("**Task:** Object Detection")
    st.write(f"**Model file:** `{MODEL_PATH}`")

    st.divider()

    st.markdown("### Defect Classes")

    st.write("• Crazing")
    st.write("• Inclusion")
    st.write("• Patches")
    st.write("• Pitted Surface")
    st.write("• Rolled-in Scale")
    st.write("• Scratches")

    st.divider()

    st.markdown("### Inspection Modes")
    st.write("🖼️ Image Detection")
    st.write("🎬 Video Detection")

    st.divider()

    st.caption("SteelSight AI | Surface Defect Detection")


# --------------------------------------------------
# HERO SECTION
# --------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <h1>Steel Surface Inspection</h1>
        <p>
            Detect, inspect, and analyze steel surface defects
            using a trained YOLO26 computer vision model.
            Upload images or videos to perform automated inspection.
            Create
        </p>
        <div style="
            margin-top: 15px;
            font-size: 15px;
            color: #00D4FF;
            font-weight: 600;
        ">
            Created with ❤️ by K.V.$.V!n@y
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# MODEL CHECK
# --------------------------------------------------

if not Path(MODEL_PATH).exists():

    st.error(
        f"Model file not found: {MODEL_PATH}. "
        "Place your trained best.pt file beside app.py."
    )

    st.stop()

model = load_model(MODEL_PATH)


# --------------------------------------------------
# INSPECTION MODE SELECTION
# --------------------------------------------------

st.markdown(
    '<div class="section-title">🔍 Choose Inspection Mode</div>',
    unsafe_allow_html=True
)

mode = st.radio(
    "Select detection type",
    ["🖼️ Image Inspection", "🎬 Video Inspection"],
    horizontal=True,
    label_visibility="collapsed"
)

confidence = st.slider(
    "Confidence Threshold",
    min_value=0.05,
    max_value=1.00,
    value=0.25,
    step=0.05
)


# ==================================================
# IMAGE INSPECTION
# ==================================================

if mode == "🖼️ Image Inspection":

    st.markdown(
        '<div class="section-title">📤 Upload Inspection Image</div>',
        unsafe_allow_html=True
    )

    uploaded_image = st.file_uploader(
        "Choose a steel surface image",
        type=["jpg", "jpeg", "png"],
        key="image_uploader"
    )

    if uploaded_image is not None:

        original_image = Image.open(
            uploaded_image
        ).convert("RGB")

        st.image(
            original_image,
            caption="Uploaded Steel Surface Image",
            use_container_width=True
        )

        if st.button(
            "🔍 Run Image Defect Detection",
            use_container_width=True
        ):

            with st.spinner("Detecting steel surface defects..."):

                results = model.predict(
                    source=original_image,
                    conf=confidence,
                    verbose=False
                )

                result = results[0]

                annotated_image = Image.fromarray(
                    result.plot()[:, :, ::-1]
                )

                detections = []

                if result.boxes is not None:

                    for box in result.boxes:

                        class_id = int(box.cls[0].item())
                        class_name = model.names[class_id]
                        score = float(box.conf[0].item())

                        x1, y1, x2, y2 = (
                            box.xyxy[0].tolist()
                        )

                        detections.append({
                            "Defect": class_name,
                            "Confidence": round(score, 4),
                            "X1": round(x1, 1),
                            "Y1": round(y1, 1),
                            "X2": round(x2, 1),
                            "Y2": round(y2, 1)
                        })

                detection_df = pd.DataFrame(detections)

                st.session_state["image_result"] = {
                    "annotated": annotated_image,
                    "original": original_image,
                    "detections": detection_df
                }

    # Display image results

    if "image_result" in st.session_state:

        image_data = st.session_state["image_result"]

        annotated_image = image_data["annotated"]
        original_image = image_data["original"]
        detection_df = image_data["detections"]

        total_detections = len(detection_df)

        unique_classes = (
            detection_df["Defect"].nunique()
            if total_detections > 0 else 0
        )

        average_confidence = (
            detection_df["Confidence"].mean()
            if total_detections > 0 else 0
        )

        st.divider()

        st.markdown(
            '<div class="section-title">📊 Image Inspection Summary</div>',
            unsafe_allow_html=True
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Total Defects</div>
                    <div class="metric-value">{total_detections}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Defect Types</div>
                    <div class="metric-value">{unique_classes}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Average Confidence</div>
                    <div class="metric-value">
                        {average_confidence:.1%}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.divider()

        st.markdown(
            '<div class="section-title">🖼️ Visual Inspection</div>',
            unsafe_allow_html=True
        )

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Original Image**")
            st.image(
                original_image,
                use_container_width=True
            )

        with col2:
            st.markdown("**Detected Defects**")
            st.image(
                annotated_image,
                use_container_width=True
            )

        st.divider()

        st.markdown(
            '<div class="section-title">🔎 Detection Details</div>',
            unsafe_allow_html=True
        )

        if total_detections > 0:

            st.dataframe(
                detection_df,
                use_container_width=True,
                hide_index=True
            )

            defect_counts = (
                detection_df["Defect"]
                .value_counts()
                .rename_axis("Defect")
                .reset_index(name="Count")
            )

            st.markdown("**Defect Distribution**")

            st.bar_chart(
                defect_counts,
                x="Defect",
                y="Count"
            )

        else:
            st.success(
                "No defects detected above the selected "
                "confidence threshold."
            )

        # Download annotated image

        image_buffer = io.BytesIO()

        annotated_image.save(
            image_buffer,
            format="PNG"
        )

        st.download_button(
            label="⬇️ Download Annotated Image",
            data=image_buffer.getvalue(),
            file_name="steelsight_image_result.png",
            mime="image/png",
            use_container_width=True
        )


# ==================================================
# VIDEO INSPECTION
# ==================================================

elif mode == "🎬 Video Inspection":

    st.markdown(
        '<div class="section-title">🎬 Upload Inspection Video</div>',
        unsafe_allow_html=True
    )

    uploaded_video = st.file_uploader(
        "Choose a steel surface video",
        type=["mp4", "avi", "mov", "mkv"],
        key="video_uploader"
    )

    if uploaded_video is not None:

        st.video(uploaded_video)

        st.info(
            "The model will inspect the video frame by frame. "
            "Processing time depends on video length, resolution, "
            "and your computer's hardware."
        )

        if st.button(
            "🎯 Run Video Defect Detection",
            use_container_width=True
        ):

            input_path = None
            output_path = None
            cap = None
            writer = None

            try:

                # Save uploaded video temporarily

                input_temp = tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=Path(uploaded_video.name).suffix
                )

                input_path = input_temp.name
                input_temp.write(uploaded_video.getvalue())
                input_temp.close()

                # Open input video

                cap = cv2.VideoCapture(input_path)

                if not cap.isOpened():
                    st.error(
                        "Unable to open this video. "
                        "Please try another video file."
                    )
                    st.stop()

                fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                total_frames = int(
                    cap.get(cv2.CAP_PROP_FRAME_COUNT)
                )

                if fps <= 0:
                    fps = 25.0

                if width <= 0 or height <= 0:
                    st.error("Invalid video dimensions.")
                    st.stop()

                # Create output video

                output_temp = tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".mp4"
                )

                output_path = output_temp.name
                output_temp.close()

                fourcc = cv2.VideoWriter_fourcc(*"mp4v")

                writer = cv2.VideoWriter(
                    output_path,
                    fourcc,
                    fps,
                    (width, height)
                )

                if not writer.isOpened():
                    st.error(
                        "Unable to create the output video. "
                        "Check your OpenCV installation."
                    )
                    st.stop()

                # Progress indicators

                progress_bar = st.progress(0)
                status_text = st.empty()

                frame_number = 0
                total_detections = 0

                class_counts = Counter()
                class_confidences = defaultdict(list)

                # Process every frame

                with st.spinner("Processing video..."):

                    while True:

                        success, frame = cap.read()

                        if not success:
                            break

                        frame_number += 1

                        # Run YOLO detection on current frame

                        results = model.predict(
                            source=frame,
                            conf=confidence,
                            verbose=False
                        )

                        result = results[0]

                        # YOLO plot returns BGR, suitable for OpenCV

                        annotated_frame = result.plot()

                        # Collect detection statistics

                        if result.boxes is not None:

                            for box in result.boxes:

                                class_id = int(
                                    box.cls[0].item()
                                )

                                class_name = model.names[class_id]

                                score = float(
                                    box.conf[0].item()
                                )

                                total_detections += 1

                                class_counts[class_name] += 1

                                class_confidences[
                                    class_name
                                ].append(score)

                        # Write annotated frame

                        writer.write(annotated_frame)

                        # Update progress

                        if total_frames > 0:

                            progress = min(
                                frame_number / total_frames,
                                1.0
                            )

                            progress_bar.progress(progress)

                        if frame_number % 10 == 0:

                            status_text.write(
                                f"Processed {frame_number} "
                                f"of {total_frames if total_frames > 0 else 'unknown'} "
                                "frames..."
                            )

                # Release video resources

                cap.release()
                cap = None

                writer.release()
                writer = None

                progress_bar.progress(1.0)

                status_text.success(
                    f"Video processing completed! "
                    f"Total frames processed: {frame_number}"
                )

                # Read processed video

                with open(output_path, "rb") as video_file:
                    output_video_bytes = video_file.read()

                # Create video summary

                average_confidence = (
                    sum(
                        sum(scores)
                        for scores in class_confidences.values()
                    ) / total_detections
                    if total_detections > 0 else 0
                )

                video_summary = []

                for class_name, count in class_counts.items():

                    scores = class_confidences[class_name]

                    video_summary.append({
                        "Defect": class_name,
                        "Detection Count": count,
                        "Average Confidence": round(
                            sum(scores) / len(scores), 4
                        )
                    })

                video_df = pd.DataFrame(video_summary)

                # Save results in session state

                st.session_state["video_result"] = {
                    "video_bytes": output_video_bytes,
                    "total_detections": total_detections,
                    "total_frames": frame_number,
                    "unique_classes": len(class_counts),
                    "average_confidence": average_confidence,
                    "details": video_df
                }

            except Exception as e:

                st.error(f"Video processing error: {e}")

            finally:

                if cap is not None:
                    cap.release()

                if writer is not None:
                    writer.release()

                if input_path and os.path.exists(input_path):
                    os.remove(input_path)

                if output_path and os.path.exists(output_path):
                    os.remove(output_path)

    # Display video results

    if "video_result" in st.session_state:

        video_data = st.session_state["video_result"]

        st.divider()

        st.markdown(
            '<div class="section-title">📊 Video Inspection Summary</div>',
            unsafe_allow_html=True
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">
                        Total Detections
                    </div>
                    <div class="metric-value">
                        {video_data["total_detections"]}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">
                        Frames Processed
                    </div>
                    <div class="metric-value">
                        {video_data["total_frames"]}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">
                        Defect Types
                    </div>
                    <div class="metric-value">
                        {video_data["unique_classes"]}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col4:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">
                        Average Confidence
                    </div>
                    <div class="metric-value">
                        {video_data["average_confidence"]:.1%}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.divider()

        st.markdown(
            '<div class="section-title">🎥 Processed Video</div>',
            unsafe_allow_html=True
        )

        st.video(video_data["video_bytes"])

        st.download_button(
            label="⬇️ Download Annotated Video",
            data=video_data["video_bytes"],
            file_name="steelsight_video_result.mp4",
            mime="video/mp4",
            use_container_width=True
        )

        st.divider()

        st.markdown(
            '<div class="section-title">🔎 Video Detection Details</div>',
            unsafe_allow_html=True
        )

        if not video_data["details"].empty:

            st.dataframe(
                video_data["details"],
                use_container_width=True,
                hide_index=True
            )

            st.markdown("**Defect Distribution Across Frames**")

            st.bar_chart(
                video_data["details"],
                x="Defect",
                y="Detection Count"
            )

        else:

            st.success(
                "No defects detected in the uploaded video "
                "above the selected confidence threshold."
            )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.markdown(
    """
    <div style="text-align:center; color:#A8BED8; padding:15px;">
        <p>🏭 SteelSight AI | YOLO26 Surface Defect Detection</p>
        <p>AI-powered quality inspection for steel manufacturing</p>
    </div>
    """,
    unsafe_allow_html=True
)

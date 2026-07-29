import streamlit as st
import cv2
import numpy as np
import av
from PIL import Image

from streamlit_webrtc import (
    webrtc_streamer,
    VideoProcessorBase,
)

from utils.predictor import predict_emotion

from utils.mediapipe_helper import (
    extract_landmarks_image,
    extract_landmarks_video,
)

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Facial Emotion Recognition",
    page_icon="😊",
    layout="wide"
)

# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown("""
<style>

.main{
    padding-top:20px;
}

.block-container{
    padding-top:2rem;
}

.title{
    font-size:40px;
    font-weight:bold;
    color:#1f77b4;
}

.subtitle{
    font-size:18px;
    color:gray;
}

.metric-card{
    background:#f8f9fa;
    padding:15px;
    border-radius:12px;
    border:1px solid #dcdcdc;
}

.footer{
    text-align:center;
    color:gray;
    font-size:14px;
}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# TITLE
# ==========================================================

st.markdown(
    "<div class='title'>😊 Facial Emotion Recognition using ANN</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>MediaPipe Tasks API + Artificial Neural Network</div>",
    unsafe_allow_html=True
)

st.markdown("---")

# ==========================================================
# EMOTION INFO
# ==========================================================

emotion_icons = {

    "happy":"😊",

    "sad":"😢",

    "angry":"😠",

    "neutral":"😐",

    "surprise":"😲"

}

emotion_colors = {

    "happy":"green",

    "sad":"blue",

    "angry":"red",

    "neutral":"orange",

    "surprise":"violet"

}

# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title("Project")

st.sidebar.success("Artificial Neural Network")

st.sidebar.info("MediaPipe Tasks API")

st.sidebar.markdown("---")

st.sidebar.write("### Model Accuracy")

st.sidebar.metric(
    label="Validation Accuracy",
    value="67.92 %"
)

st.sidebar.markdown("---")

st.sidebar.write("### Emotion Classes")

for e in emotion_icons:

    st.sidebar.write(
        emotion_icons[e],
        e.capitalize()
    )

st.sidebar.markdown("---")

mode = st.sidebar.radio(

    "Select Mode",

    [

        "Upload Image",

        "Live Webcam"

    ]

)

# ==========================================================
# HEADER
# ==========================================================

st.subheader(mode)

st.markdown("---")

# ==========================================================
# UPLOAD IMAGE
# ==========================================================

if mode=="Upload Image":

    uploaded = st.file_uploader(

        "Upload an Image",

        type=[

            "jpg",

            "jpeg",

            "png"

        ]

    )

    if uploaded is not None:

        image = Image.open(uploaded)

        image = np.array(image)

        image = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2BGR
        )

        landmarks = extract_landmarks_image(
            image
        )

        if landmarks is None:

            st.error(
                "No face detected."
            )

        else:

            emotion, confidence = predict_emotion(
                landmarks
            )

            rgb = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            )

            col1, col2 = st.columns([2,1])

            with col1:

                st.image(
                    rgb,
                    use_container_width=True
                )

            with col2:

                st.success("Prediction")

                st.metric(

                    "Emotion",

                    f"{emotion_icons[emotion]} {emotion.capitalize()}"

                )

                st.metric(

                    "Confidence",

                    f"{confidence*100:.2f}%"

                )

                st.progress(
                    float(confidence)
                )

                st.write("### Prediction Status")

                if confidence > 0.80:

                    st.success("Very High Confidence")

                elif confidence > 0.60:

                    st.info("Good Confidence")

                elif confidence > 0.40:

                    st.warning("Moderate Confidence")

                else:

                    st.error("Low Confidence")

# ==========================================================
# LIVE WEBCAM PROCESSOR
# ==========================================================

class EmotionProcessor(VideoProcessorBase):

    def recv(self, frame):

        image = frame.to_ndarray(format="bgr24")

        output = image.copy()

        landmarks = extract_landmarks_video(image)

        if landmarks is not None:

            emotion, confidence = predict_emotion(
                landmarks
            )

            h, w = image.shape[:2]

            points = landmarks.reshape(478, 3)

            xs = (points[:, 0] * w).astype(np.int32)
            ys = (points[:, 1] * h).astype(np.int32)

            x1 = max(0, np.min(xs) - 20)
            y1 = max(0, np.min(ys) - 20)

            x2 = min(w, np.max(xs) + 20)
            y2 = min(h, np.max(ys) + 20)

            color = (0,255,0)

            if emotion == "angry":
                color = (0,0,255)

            elif emotion == "sad":
                color = (255,0,0)

            elif emotion == "neutral":
                color = (0,255,255)

            elif emotion == "surprise":
                color = (255,0,255)

            cv2.rectangle(
                output,
                (x1,y1),
                (x2,y2),
                color,
                2
            )

            cv2.rectangle(
                output,
                (x1,y1-65),
                (x2,y1),
                color,
                -1
            )

            cv2.putText(
                output,
                emotion.upper(),
                (x1+10,y1-38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255,255,255),
                2
            )

            cv2.putText(
                output,
                f"{confidence*100:.1f}%",
                (x1+10,y1-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (255,255,255),
                2
            )

        return av.VideoFrame.from_ndarray(
            output,
            format="bgr24"
        )


# ==========================================================
# LIVE WEBCAM MODE
# ==========================================================

if mode == "Live Webcam":

    st.info(
        "Click START and allow camera permission."
    )

    webrtc_streamer(

        key="emotion-recognition",

        video_processor_factory=EmotionProcessor,

        media_stream_constraints={
            "video":True,
            "audio":False
        },

        async_processing=True

    )

    st.markdown("---")

    st.markdown("### Supported Emotions")

    c1,c2,c3,c4,c5 = st.columns(5)

    with c1:
        st.metric("😊","Happy")

    with c2:
        st.metric("😠","Angry")

    with c3:
        st.metric("😐","Neutral")

    with c4:
        st.metric("😢","Sad")

    with c5:
        st.metric("😲","Surprise")


# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")

st.markdown(
"""
<div class='footer'>

Facial Emotion Recognition using
<b>MediaPipe Tasks API</b> +
<b>Artificial Neural Network</b>

</div>
""",
unsafe_allow_html=True
)
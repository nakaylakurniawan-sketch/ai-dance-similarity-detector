import math
import tempfile
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import streamlit as st

import os
from pathlib import Path

# Memastikan model MediaPipe dapat dibaca
import mediapipe as mp

model_path = (
    Path(mp.__file__).parent
    / "modules"
    / "pose_landmark"
    / "pose_landmark_lite.tflite"
)

if model_path.exists():
    os.chmod(model_path, 0o644)

# ------------------------------------------------------------
# DanceAI | AI Dance Similarity Detector
# ------------------------------------------------------------
st.set_page_config(
    page_title="DanceAI | Dance Similarity",
    page_icon="💃",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main {background-color: #fbf8fc;}
    .block-container {padding-top: 2rem; max-width: 1150px;}
    .hero {
        padding: 1.5rem 1.7rem;
        border-radius: 18px;
        background: linear-gradient(120deg, #f8e8f2, #eeeaff);
        margin-bottom: 1.2rem;
    }
    .hero h1 {margin: 0; color: #392b45;}
    .hero p {margin: .4rem 0 0 0; color: #65576d;}
    .metric-card {
        padding: 1rem; border-radius: 14px;
        background: white; border: 1px solid #eee5f1;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>💃 DanceAI</h1>
      <p>AI-powered dance movement similarity analyzer</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write(
    "Bandingkan video dance referensi dengan video latihanmu. "
    "Sistem mengekstrak pose tubuh, menyelaraskan urutan gerakan, "
    "lalu menghitung skor kemiripan."
)

st.info(
    "Tips: gunakan video dengan satu orang, seluruh tubuh terlihat, "
    "kamera relatif diam, dan gerakan yang sama. Untuk demo awal, "
    "gunakan video pendek sekitar 5–20 detik."
)

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# MediaPipe landmark indices used in the analysis.
BODY_IDS = list(range(11, 17)) + list(range(23, 33))
ARM_IDS = [11, 12, 13, 14, 15, 16]
LEG_IDS = [23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
TORSO_IDS = [11, 12, 23, 24]
BODY_POS = {landmark_id: i for i, landmark_id in enumerate(BODY_IDS)}
ARM_POS = [BODY_POS[i] for i in ARM_IDS]
LEG_POS = [BODY_POS[i] for i in LEG_IDS]
TORSO_POS = [BODY_POS[i] for i in TORSO_IDS]


def extract_pose_video(video_bytes: bytes, sample_fps: int = 5, max_frames: int = 75):
    """Read uploaded video bytes and extract visible MediaPipe pose landmarks."""
    temp_path = None
    cap = None
    poses = []
    visibility_sequences = []
    preview_rgb = []
    timestamps = []

    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp:
            temp.write(video_bytes)
            temp_path = temp.name

        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            raise ValueError(
                "Video tidak dapat dibaca. Coba gunakan MP4 (H.264) atau MOV yang umum."
            )

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or not np.isfinite(fps) or fps <= 0:
            fps = 30.0
        interval = max(1, int(round(fps / sample_fps)))

        frame_index = 0
        with mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as pose_model:
            while len(poses) < max_frames:
                ok, frame = cap.read()
                if not ok:
                    break

                if frame_index % interval == 0:
                    h, w = frame.shape[:2]
                    target_w = min(480, w)
                    target_h = max(1, int(h * target_w / w))
                    frame = cv2.resize(frame, (target_w, target_h))
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    result = pose_model.process(rgb)

                    if result.pose_landmarks:
                        landmarks = result.pose_landmarks.landmark
                        coords = np.array(
                            [[lm.x, lm.y, lm.z] for lm in landmarks],
                            dtype=np.float32,
                        )
                        visibility = np.array([lm.visibility for lm in landmarks])
                        if float(np.mean(visibility[BODY_IDS])) >= 0.30:
                            annotated = frame.copy()
                            mp_drawing.draw_landmarks(
                                annotated,
                                result.pose_landmarks,
                                mp_pose.POSE_CONNECTIONS,
                            )
                            poses.append(coords)
                            visibility_sequences.append(visibility[BODY_IDS])
                            preview_rgb.append(
                                cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                            )
                            timestamps.append(frame_index / fps)

                frame_index += 1

        if len(poses) < 3:
            raise ValueError(
                "Pose tubuh belum cukup terdeteksi. Pastikan tubuh terlihat jelas "
                "dan video memiliki pencahayaan yang cukup."
            )

        return np.asarray(poses), np.asarray(visibility_sequences), preview_rgb, timestamps

    finally:
        if cap is not None:
            cap.release()
        if temp_path and Path(temp_path).exists():
            Path(temp_path).unlink(missing_ok=True)


def normalize_sequence(sequence: np.ndarray) -> np.ndarray:
    """Center landmarks at hip midpoint and scale by torso length."""
    normalized = []
    for frame in sequence:
        hip_center = (frame[23] + frame[24]) / 2.0
        shoulder_center = (frame[11] + frame[12]) / 2.0
        scale = float(np.linalg.norm(shoulder_center[:2] - hip_center[:2]))
        scale = max(scale, 0.025)
        centered = (frame - hip_center) / scale
        normalized.append(centered[BODY_IDS])
    return np.asarray(normalized, dtype=np.float32)


def dtw_distance(seq_a: np.ndarray, seq_b: np.ndarray, positions=None) -> float:
    """Classic DTW with a frame-to-frame mean landmark distance."""
    if positions is not None:
        seq_a = seq_a[:, positions, :]
        seq_b = seq_b[:, positions, :]

    n, m = len(seq_a), len(seq_b)
    dp = np.full((n + 1, m + 1), np.inf, dtype=np.float32)
    dp[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            frame_cost = float(
                np.mean(np.linalg.norm(seq_a[i - 1] - seq_b[j - 1], axis=1))
            )
            dp[i, j] = frame_cost + min(
                dp[i - 1, j],
                dp[i, j - 1],
                dp[i - 1, j - 1],
            )

    # Normalized accumulated cost. This is a distance, not a probability.
    return float(dp[n, m] / max(n + m, 1))


def distance_to_score(distance: float) -> float:
    """Map DTW distance to an easy-to-read heuristic score (not a probability)."""
    return round(float(np.clip(100.0 * math.exp(-2.0 * distance), 0, 100)), 1)


def analyze(reference_bytes: bytes, user_bytes: bytes):
    ref_raw, ref_visibility, ref_previews, ref_times = extract_pose_video(reference_bytes)
    usr_raw, usr_visibility, usr_previews, usr_times = extract_pose_video(user_bytes)

    ref = normalize_sequence(ref_raw)
    usr = normalize_sequence(usr_raw)

    overall_distance = dtw_distance(ref, usr)
    arm_distance = dtw_distance(ref, usr, ARM_POS)
    leg_distance = dtw_distance(ref, usr, LEG_POS)
    torso_distance = dtw_distance(ref, usr, TORSO_POS)

    return {
        "reference_frames": ref_raw,
        "user_frames": usr_raw,
        "reference_previews": ref_previews,
        "user_previews": usr_previews,
        "reference_times": ref_times,
        "user_times": usr_times,
        "overall": distance_to_score(overall_distance),
        "arms": distance_to_score(arm_distance),
        "legs": distance_to_score(leg_distance),
        "torso": distance_to_score(torso_distance),
        "distance": overall_distance,
    }


# Upload interface
left, right = st.columns(2)
with left:
    st.subheader("1. Reference dance")
    reference_file = st.file_uploader(
        "Upload video referensi",
        type=["mp4", "mov", "avi", "m4v"],
        key="reference",
    )
    if reference_file:
        st.video(reference_file)

with right:
    st.subheader("2. Your dance")
    user_file = st.file_uploader(
        "Upload video latihanmu",
        type=["mp4", "mov", "avi", "m4v"],
        key="user",
    )
    if user_file:
        st.video(user_file)

st.caption(
    "Batas demo: maksimal 50 MB per video. Pemrosesan mengambil hingga "
    "75 frame sampel per video agar lebih ringan."
)

analyze_button = st.button(
    "✨ Analyze dance movement",
    type="primary",
    use_container_width=True,
    disabled=(reference_file is None or user_file is None),
)

if analyze_button:
    if reference_file is None or user_file is None:
        st.warning("Upload kedua video terlebih dahulu.")
    elif reference_file.size > 50 * 1024 * 1024 or user_file.size > 50 * 1024 * 1024:
        st.error("Ukuran setiap video maksimal 50 MB untuk prototype ini.")
    else:
        try:
            with st.spinner("Menganalisis pose dan membandingkan gerakan..."):
                result = analyze(reference_file.getvalue(), user_file.getvalue())
            st.session_state["dance_result"] = result
        except Exception as exc:
            st.session_state.pop("dance_result", None)
            st.error(f"Analisis belum berhasil: {exc}")
            st.markdown(
                "**Coba periksa:** seluruh tubuh terlihat, video tidak rusak, "
                "dan format video didukung."
            )

result = st.session_state.get("dance_result")

if result:
    st.divider()
    st.header("📊 Analysis result")

    metric1, metric2, metric3, metric4 = st.columns(4)
    metric1.metric("Overall similarity", f'{result["overall"]}%')
    metric2.metric("Arm movement", f'{result["arms"]}%')
    metric3.metric("Leg movement", f'{result["legs"]}%')
    metric4.metric("Torso movement", f'{result["torso"]}%')

    if result["overall"] >= 85:
        interpretation = "Gerakan yang terdeteksi sangat mirip pada video ini."
    elif result["overall"] >= 70:
        interpretation = "Gerakan cukup mirip, tetapi masih ada perbedaan."
    elif result["overall"] >= 50:
        interpretation = "Kemiripan sedang. Coba perhatikan kembali urutan gerak."
    else:
        interpretation = "Gerakan cukup berbeda dari referensi. Coba ulangi dengan posisi kamera yang serupa."

    st.info(
        f"**Interpretasi skor:** {interpretation}\n\n"
        "Skor merupakan indikator heuristik dari jarak landmark dan DTW, "
        "bukan probabilitas atau penilaian profesional atas kualitas dance."
    )

    chart_data = pd.DataFrame({
        "Bagian tubuh": ["Arms", "Legs", "Torso"],
        "Similarity (%)": [
            result["arms"],
            result["legs"],
            result["torso"],
        ],
    })
    st.subheader("Body-part comparison")
    st.bar_chart(chart_data, x="Bagian tubuh", y="Similarity (%)")

    st.subheader("Pose visualization")
    ref_col, user_col = st.columns(2)
    ref_mid = result["reference_previews"][len(result["reference_previews"]) // 2]
    user_mid = result["user_previews"][len(result["user_previews"]) // 2]

    with ref_col:
        st.markdown("**Reference pose**")
        st.image(ref_mid, use_container_width=True)
        st.caption(f'{len(result["reference_frames"])} pose frames detected')

    with user_col:
        st.markdown("**Your pose**")
        st.image(user_mid, use_container_width=True)
        st.caption(f'{len(result["user_frames"])} pose frames detected')

    st.caption(
        f'Normalized DTW distance: {result["distance"]:.4f}. '
        "Nilai jarak yang lebih kecil menunjukkan landmark yang lebih dekat."
    )

    st.download_button(
        "Download result summary",
        data=(
            "AI Dance Similarity Detector\n"
            f'Overall similarity: {result["overall"]}%\n'
            f'Arm similarity: {result["arms"]}%\n'
            f'Leg similarity: {result["legs"]}%\n'
            f'Torso similarity: {result["torso"]}%\n'
            f'Normalized DTW distance: {result["distance"]:.4f}\n'
            "Note: scores are heuristic indicators, not validated probabilities.\n"
        ),
        file_name="dance_similarity_result.txt",
        mime="text/plain",
    )

st.divider()
st.caption(
    "Prototype akademik — MediaPipe Pose + Dynamic Time Warping. "
    "Video diproses saat analisis berlangsung dan tidak disimpan sebagai database."
)

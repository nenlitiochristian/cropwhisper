import os
import tempfile
from pathlib import Path

import cv2
import numpy as np


DEFAULT_OUTPUT_DIR = os.path.join(tempfile.gettempdir(), "cropwhisper_keyframes")
MAX_JPEG_SIZE_BYTES = 500_000
JPEG_INITIAL_QUALITY = 85
JPEG_MIN_QUALITY = 30


def validate_video(video_path: str) -> tuple[bool, str]:
    """Check if video file is valid and can be processed.
    Returns (is_valid, error_message).
    """
    if not os.path.exists(video_path):
        return False, f"File not found: {video_path}"

    if not os.path.isfile(video_path):
        return False, f"Not a file: {video_path}"

    if os.path.getsize(video_path) == 0:
        return False, "File is empty"

    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            return False, "Cannot open video file (corrupted or unsupported format)"

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        if frame_count <= 0:
            return False, "Video contains no frames"

        if fps <= 0:
            return False, "Video has invalid frame rate"

        ret, frame = cap.read()
        if not ret or frame is None:
            return False, "Cannot read any frames from video"

        return True, ""
    finally:
        cap.release()


def _compute_histogram(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist


def _sample_frames(video_path: str) -> list[tuple[int, np.ndarray]]:
    cap = cv2.VideoCapture(video_path)
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        step = max(1, int(fps // 3)) if fps > 3 else 1
        step = max(step, total_frames // 100) if total_frames > 100 else step

        frames = []
        for idx in range(0, total_frames, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret and frame is not None:
                frames.append((idx, frame))

        return frames
    finally:
        cap.release()


def _select_keyframes(
    frames: list[tuple[int, np.ndarray]], num_keyframes: int
) -> list[tuple[int, np.ndarray]]:
    if len(frames) <= num_keyframes:
        return frames

    histograms = [_compute_histogram(f) for _, f in frames]

    n = len(frames)
    similarity = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            score = cv2.compareHist(histograms[i], histograms[j], cv2.HISTCMP_CORREL)
            similarity[i][j] = score
            similarity[j][i] = score
        similarity[i][i] = 1.0

    selected = [0]

    while len(selected) < num_keyframes:
        best_idx = -1
        best_score = float("inf")

        for i in range(n):
            if i in selected:
                continue
            max_sim_to_selected = max(similarity[i][j] for j in selected)
            if max_sim_to_selected < best_score:
                best_score = max_sim_to_selected
                best_idx = i

        if best_idx == -1:
            break
        selected.append(best_idx)

    selected.sort()
    return [frames[i] for i in selected]


def _save_frame(frame: np.ndarray, path: str) -> str:
    quality = JPEG_INITIAL_QUALITY
    while quality >= JPEG_MIN_QUALITY:
        cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if os.path.getsize(path) <= MAX_JPEG_SIZE_BYTES:
            return path
        quality -= 10

    h, w = frame.shape[:2]
    scale = 0.7
    resized = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(path, resized, [cv2.IMWRITE_JPEG_QUALITY, JPEG_MIN_QUALITY])
    return path


def extract_keyframes(
    video_path: str, num_keyframes: int = 3, output_dir: str | None = None
) -> list[str]:
    """Extract visually distinct keyframes from a video file.

    Args:
        video_path: Path to the video file
        num_keyframes: Number of keyframes to extract (2-3)
        output_dir: Directory for temp files (defaults to /tmp/cropwhisper_keyframes)

    Returns:
        List of file paths to extracted keyframe images
    """
    is_valid, error = validate_video(video_path)
    if not is_valid:
        raise ValueError(f"Invalid video: {error}")

    num_keyframes = max(1, min(num_keyframes, 5))

    output_dir = output_dir or DEFAULT_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    frames = _sample_frames(video_path)
    if not frames:
        raise ValueError("Could not extract any frames from video")

    keyframes = _select_keyframes(frames, num_keyframes)

    video_name = Path(video_path).stem
    paths = []
    for i, (frame_idx, frame) in enumerate(keyframes):
        filename = f"{video_name}_kf{i}_{frame_idx}.jpg"
        filepath = os.path.join(output_dir, filename)
        _save_frame(frame, filepath)
        paths.append(filepath)

    return paths

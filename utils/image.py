import base64
import os
from pathlib import Path

MAX_TOTAL_SIZE_BYTES = 500 * 1024  # 500KB


def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def encode_multiple_images(image_paths: list[str]) -> list[str]:
    return [encode_image_to_base64(p) for p in image_paths if p and os.path.exists(p)]


def get_file_size(path: str) -> int:
    return os.path.getsize(path) if os.path.exists(path) else 0


def validate_image_sizes(paths: list[str]) -> tuple[bool, str, int]:
    """Validate that total image size does not exceed MAX_TOTAL_SIZE_BYTES.

    Returns (is_valid, error_message, total_bytes).
    """
    total = sum(get_file_size(p) for p in paths if p)
    if total > MAX_TOTAL_SIZE_BYTES:
        return False, f"Total image size ({total // 1024}KB) exceeds the 500KB limit.", total
    return True, "", total


def get_image_size_display(paths: list[str]) -> str:
    total = sum(get_file_size(p) for p in paths if p)
    remaining = MAX_TOTAL_SIZE_BYTES - total
    return f"{total // 1024}KB / {MAX_TOTAL_SIZE_BYTES // 1024}KB used ({max(0, remaining) // 1024}KB remaining)"

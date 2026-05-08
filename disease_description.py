import argparse
import json
import os

from dotenv import load_dotenv
load_dotenv()

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import OpenAI

from utils.image import encode_image_to_base64


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def read_prompt(name: str) -> str:
    prompt_path = Path("prompts") / f"{name}.md"
    return prompt_path.read_text(encoding="utf-8")


def parse_label(folder_name: str) -> dict[str, str]:
    if "___" in folder_name:
        crop, condition = folder_name.split("___", maxsplit=1)
    else:
        crop, condition = folder_name, "unknown"
    return {
        "folder_label": folder_name,
        "crop": crop.replace("_", " "),
        "condition": condition.replace("_", " "),
    }


def build_visual_client() -> OpenAI:
    base = os.environ.get("VLLM_BASE_URL", "http://localhost")
    return OpenAI(base_url=f"{base}:8000/v1", api_key="none")


def resolve_model_name(client: OpenAI) -> str:
    models = client.models.list()
    if not models.data:
        raise RuntimeError("No models are available on the visual vLLM endpoint (port 8000).")
    return models.data[0].id


def image_files_in(folder: Path) -> list[Path]:
    return sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS],
        key=lambda p: p.name,
    )


def call_visual_model(
    client: OpenAI,
    model_name: str,
    image_path: Path,
    system_prompt: str,
    max_tokens: int,
) -> tuple[str, Any]:
    b64_image = encode_image_to_base64(str(image_path))
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"},
                    },
                    {
                        "type": "text",
                        "text": "Describe this crop leaf image following your instructions exactly.",
                    },
                ],
            },
        ],
        max_tokens=max_tokens,
    )

    raw = response.choices[0].message.content or ""
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = {"raw_output": raw, "parse_error": True}
    return raw, parsed


def write_rag_header(out_file: Path, metadata: dict[str, Any]) -> None:
    with out_file.open("w", encoding="utf-8") as f:
        f.write("# CropWhisper RAG Seed Data\n")
        f.write("# Format: JSON Lines (one document per line)\n")
        f.write(f"# metadata={json.dumps(metadata, ensure_ascii=False)}\n")


def append_rag_document(out_file: Path, doc: dict[str, Any]) -> None:
    with out_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(doc, ensure_ascii=False) + "\n")


def generate_rag(
    dataset_dir: Path,
    output_file: Path,
    prompt_name: str,
    max_tokens: int,
    sleep_seconds: float,
    max_images_per_folder: int | None,
    max_total_images: int | None,
    retry_count: int,
) -> None:
    if not dataset_dir.exists() or not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_dir}")

    client = build_visual_client()
    model_name = resolve_model_name(client)
    system_prompt = read_prompt(prompt_name)

    folders = sorted([p for p in dataset_dir.iterdir() if p.is_dir()], key=lambda p: p.name)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    write_rag_header(
        output_file,
        {
            "run_id": run_id,
            "prompt": prompt_name,
            "model": model_name,
            "dataset_dir": str(dataset_dir),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )

    total_written = 0
    total_scanned = 0

    for folder in folders:
        label_info = parse_label(folder.name)
        images = image_files_in(folder)
        if max_images_per_folder is not None:
            images = images[:max_images_per_folder]

        for image_path in images:
            if max_total_images is not None and total_written >= max_total_images:
                print(f"Reached max_total_images={max_total_images}. Stopping.")
                print(f"RAG output written to: {output_file}")
                return

            total_scanned += 1
            print(f"[{total_scanned}] Processing {image_path}")

            last_error = None
            for attempt in range(1, retry_count + 2):
                try:
                    raw, parsed = call_visual_model(
                        client=client,
                        model_name=model_name,
                        image_path=image_path,
                        system_prompt=system_prompt,
                        max_tokens=max_tokens,
                    )
                    now_iso = datetime.now(timezone.utc).isoformat()
                    doc = {
                        "doc_id": f"{run_id}:{folder.name}:{image_path.stem}",
                        "source": "plant_disease_images",
                        "label": label_info,
                        "image": {
                            "file_name": image_path.name,
                            "relative_path": str(image_path.relative_to(dataset_dir)),
                        },
                        "prompt": {"name": prompt_name},
                        "model": {"name": model_name, "endpoint": "vllm:8000"},
                        "timestamps": {"generated_at_utc": now_iso},
                        "visual_description": parsed,
                        "raw_output": raw,
                    }
                    append_rag_document(output_file, doc)
                    total_written += 1
                    break
                except Exception as exc:
                    last_error = str(exc)
                    print(f"  Attempt {attempt} failed: {exc}")
                    if attempt <= retry_count:
                        time.sleep(1.0)
            else:
                print(f"  Skipping {image_path.name} after retries. Last error: {last_error}")

            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

    print(f"Done. Scanned={total_scanned}, Written={total_written}")
    print(f"RAG output written to: {output_file}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate RAG seed data from plant disease images using vLLM visual model."
    )
    parser.add_argument("--dataset-dir", default="plant_disease_images", help="Dataset root folder.")
    parser.add_argument("--output", default="RAG.txt", help="Output text file path.")
    parser.add_argument(
        "--prompt-name",
        default="disease_visual",
        help="Prompt file name in prompts/ without extension.",
    )
    parser.add_argument("--max-tokens", type=int, default=2048, help="LLM max token budget.")
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="Delay between requests to reduce endpoint pressure.",
    )
    parser.add_argument(
        "--max-images-per-folder",
        type=int,
        default=None,
        help="Optional cap per class folder.",
    )
    parser.add_argument(
        "--max-total-images",
        type=int,
        default=None,
        help="Optional cap across the full dataset.",
    )
    parser.add_argument("--retry-count", type=int, default=2, help="Retries after first failed attempt.")
    args = parser.parse_args()

    generate_rag(
        dataset_dir=Path(args.dataset_dir),
        output_file=Path(args.output),
        prompt_name=args.prompt_name,
        max_tokens=args.max_tokens,
        sleep_seconds=args.sleep_seconds,
        max_images_per_folder=args.max_images_per_folder,
        max_total_images=args.max_total_images,
        retry_count=args.retry_count,
    )


if __name__ == "__main__":
    main()
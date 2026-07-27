import argparse
import hashlib
import io
import json
import os
import re
import sys
import tempfile
from statistics import median
from pathlib import Path

# 라이브러리는 개인 가상환경에 설치하고, 대용량 Qwen 모델만 팀 공용 캐시를 사용한다.
os.environ.setdefault("HF_HOME", "/data/model_cache/huggingface")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import fitz
import pandas as pd
import torch
import yaml
from paddleocr import PaddleOCR
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

from src.ocr_extractor import extract_hwp_images, extract_pdf_images

def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def image_size(image_bytes: bytes) -> tuple[int, int]:
    with Image.open(io.BytesIO(image_bytes)) as image:
        return image.size


def document_images(path: Path, min_pdf_text_length: int) -> list[dict]:
    if path.suffix.lower() == ".hwp":
        return extract_hwp_images(path)

    images = extract_pdf_images(path)
    document = fitz.open(path)
    try:
        for page_number, page in enumerate(document, start=1):
            if len(page.get_text("text").strip()) >= min_pdf_text_length:
                continue
            rendered = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            images.append(
                {
                    "page_number": page_number,
                    "image_number": None,
                    "image_bytes": rendered.tobytes("png"),
                    "image_extension": "png",
                    "source_type": "rendered_pdf_page",
                }
            )
    finally:
        document.close()
    return images


def image_type(vlm_text: str) -> str:
    types = "diagram|form|table|chart|logo|photo|other"
    # 1) '유형:'/'type:' 접두사가 있으면 우선 사용
    match = re.search(rf"^(?:유형|type)\s*[:：]\s*({types})\b", vlm_text, re.MULTILINE | re.IGNORECASE)
    if match:
        return match.group(1).lower()
    # 2) 접두사가 없어도 본문에 나온 유형 단어를 인식 (관대한 파싱)
    match = re.search(rf"\b({types})\b", vlm_text, re.IGNORECASE)
    return match.group(1).lower() if match else "other"


def ocr_items(result: dict) -> list[dict]:
    items=[]
    for text, box in zip(result["rec_texts"], result["rec_boxes"]):
        left, top, right, bottom=(int(v) for v in box)
        items.append({"text": text, "bbox": {"left": left, "top": top, "right": right, "bottom": bottom}, "_x": (left+right)/2, "_y": (top+bottom)/2, "_h": bottom-top})
    return items


def spatial_layout(items: list[dict], gap_multiplier: float) -> dict:
    if not items:
        return {"row_gap_multiplier": gap_multiplier, "rows": []}
    ordered=sorted(items, key=lambda item: (item["_y"], item["_x"]))
    threshold=median(item["_h"] for item in ordered)*gap_multiplier
    rows=[[ordered[0]]]
    for item in ordered[1:]:
        if item["_y"]-rows[-1][-1]["_y"] > threshold:
            rows.append([item])
        else:
            rows[-1].append(item)
    return {"row_gap_multiplier": gap_multiplier, "rows": [
        {"row_index": index, "items": [
            {"text": item["text"], "bbox": item["bbox"]}
            for item in sorted(row, key=lambda item: item["_x"])
        ]}
        for index, row in enumerate(rows, start=1)
    ]}


def combined_context(ocr_text: str, layout: dict, vlm_text: str | None) -> str:
    rows = "\n".join(
        f"\ud589 {row['row_index']}: " + " | ".join(item["text"] for item in row["items"])
        for row in layout["rows"]
    )
    parts = [
        "\uc774\ubbf8\uc9c0 OCR \ud14d\uc2a4\ud2b8:\n" + ocr_text,
        "\uacf5\uac04 \ubc30\uce58(\ud654\uc0b4\ud45c \ud750\ub984 \uc544\ub2d8):\n" + rows,
    ]
    if vlm_text:
        parts.append("\uac80\ud1a0 \ud544\uc694 VLM \uac1c\uc694:\n" + vlm_text)
    return "\n\n".join(parts)


class MultimodalExtractor:
    def __init__(self, model_name: str, gap_multiplier: float):
        self.ocr=PaddleOCR(lang="korean", ocr_version="PP-OCRv5", use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False, device="cpu")
        self.model_name=model_name
        self.gap_multiplier=gap_multiplier
        self.model=None
        self.processor=None

    def extract_ocr(self, image_path: str) -> tuple[str, dict]:
        result=self.ocr.predict(image_path)[0]
        return "\n".join(result["rec_texts"]).strip(), spatial_layout(ocr_items(result), self.gap_multiplier)

    def load_vlm(self) -> None:
        if self.model is not None:
            return
        quantization=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        self.model=Qwen2_5_VLForConditionalGeneration.from_pretrained(self.model_name, quantization_config=quantization, device_map="auto")
        self.processor=AutoProcessor.from_pretrained(self.model_name)

    def extract_vlm(self, image_path: str, ocr_text: str | None = None) -> str:
        self.load_vlm()
        ocr_section = f"이미지에서 PaddleOCR로 추출한 정확한 텍스트:\n---\n{ocr_text}\n---\n\n" if ocr_text else ""
        instruction = ocr_section + (
            "위 텍스트와 이미지를 함께 참고하세요.\n"
            "유형 기준:\n"
            "- diagram: 상자/요소가 화살표나 선으로 연결된 순서도/구성도/조직도\n"
            "- table: 행과 열 격자에 값이 채워진 표\n"
            "- form: 항목과 입력란이 있는 서식/양식 문서\n"
            "- chart: 막대/선/원 등으로 수치를 나타낸 그래프나 워드클라우드\n"
            "- logo: 기관/서비스 로고나 심볼\n"
            "- photo: 실사 사진이나 지도\n"
            "- other: 위에 해당하지 않는 것\n"
            "규칙: 위 텍스트와 이미지에 실제로 보이는 것만 근거로 삼으세요. "
            "이미지에 없는 정보(수치, 기술 용어, 관계, 의도)를 지어내지 마세요. 확실하지 않으면 그 부분은 쓰지 마세요.\n\n"
            "반드시 아래 두 줄 형식으로만 답하세요. 두 줄 모두 채우세요.\n"
            "유형: <diagram|table|form|chart|logo|photo|other 중 하나>\n"
            "요약: <이 이미지에 무엇이 담겨 있는지 한국어 한 문장>"
        )
        messages = [{"role": "user", "content": [
            {"type": "image", "image": f"file://{image_path}"},
            {"type": "text", "text": instruction},
        ]}]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(text=[prompt], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(**inputs, max_new_tokens=180, do_sample=False)
        output_ids = [row[len(input_ids):] for input_ids, row in zip(inputs.input_ids, generated_ids)]
        return self.processor.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()


def report_row(
    doc_id: str,
    file_name: str,
    image: dict,
    image_hash: str,
    status: str,
    reason: str,
    width: int | None = None,
    height: int | None = None,
) -> dict:
    return {
        "doc_id": doc_id,
        "file_name": file_name,
        "source_type": image.get("source_type", "embedded_image"),
        "page_number": image.get("page_number"),
        "image_number": image.get("image_number"),
        "image_sha256": image_hash,
        "width": width,
        "height": height,
        "status": status,
        "reason": reason,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    paths = config["paths"]
    options = config["multimodal"]
    metadata = pd.read_csv(paths["metadata"], encoding="utf-8")
    raw_documents = Path(paths["raw_documents"])
    if args.limit is not None:
        metadata = metadata.head(args.limit)

    extractor = MultimodalExtractor(options["model"], options["layout_row_gap_multiplier"])
    records: list[dict] = []
    report: list[dict] = []

    for index, row in metadata.iterrows():
        doc_id = f"doc_{index + 1:03d}"
        file_name = str(row["파일명"]).strip()
        file_path = raw_documents / file_name
        if not file_path.is_file():
            report.append(
                {
                    "doc_id": doc_id,
                    "file_name": file_name,
                    "status": "failed",
                    "reason": "source_file_not_found",
                }
            )
            continue

        seen_hashes: set[str] = set()
        try:
            images = document_images(file_path, options["min_pdf_text_length"])
        except Exception as error:
            report.append(
                {
                    "doc_id": doc_id,
                    "file_name": file_name,
                    "status": "failed",
                    "reason": f"{type(error).__name__}: {error}",
                }
            )
            continue

        for image in images:
            payload = image["image_bytes"]
            image_hash = sha256(payload)
            if image_hash in seen_hashes:
                report.append(
                    report_row(
                        doc_id, file_name, image, image_hash,
                        "duplicate", "document_image_duplicate"
                    )
                )
                continue
            seen_hashes.add(image_hash)

            try:
                width, height = image_size(payload)
                if width < options["min_width"] or height < options["min_height"]:
                    report.append(
                        report_row(
                            doc_id, file_name, image, image_hash,
                            "excluded", "image_too_small", width, height
                        )
                    )
                    continue

                extension = image.get("image_extension", "png")
                with tempfile.NamedTemporaryFile(suffix=f".{extension}") as temporary:
                    temporary.write(payload)
                    temporary.flush()
                    ocr_text, layout = extractor.extract_ocr(temporary.name)
                    if width < options["min_vlm_width"] or height < options["min_vlm_height"]:
                        detected_type, vlm_text, vlm_status = "unclassified", None, "skipped_image_too_small"
                    else:
                        candidate = extractor.extract_vlm(temporary.name, ocr_text)
                        detected_type = image_type(candidate)
                        if detected_type in {"logo", "photo"}:
                            vlm_text, vlm_status = None, "discarded_logo_or_photo"
                        else:
                            vlm_text, vlm_status = candidate, "applied"


                records.append(
                    {
                        "doc_id": doc_id,
                        "file_name": file_name,
                        "file_type": file_path.suffix.lower().lstrip("."),
                        "source_type": image.get("source_type", "embedded_image"),
                        "page_number": image.get("page_number"),
                        "image_number": image.get("image_number"),
                        "image_sha256": image_hash,
                        "width": width,
                        "height": height,
                        "ocr_engine": "PaddleOCR PP-OCRv5",
                        "ocr_text": ocr_text,
                        "spatial_layout": layout,
                        "combined_context": combined_context(ocr_text, layout, vlm_text),
                        "vlm_model": options["model"],
                        "vlm_status": vlm_status,
                        "image_type": detected_type,
                        "vlm_text": vlm_text,
                        "review_required": True,
                    }
                )
                report.append(
                    report_row(
                        doc_id, file_name, image, image_hash,
                        "applied", "", width, height
                    )
                )
            except Exception as error:
                report.append(
                    report_row(
                        doc_id, file_name, image, image_hash,
                        "failed", f"{type(error).__name__}: {error}"
                    )
                )

    output = Path(paths["multimodal_documents"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    pd.DataFrame(report).to_csv(paths["multimodal_report"], index=False, encoding="utf-8")

    print(f"멀티모달 처리 완료: {len(records)}건")
    print(f"결과 파일: {output}")
    print(f"처리 리포트: {paths['multimodal_report']}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import hashlib
import io
import json
import zlib
from pathlib import Path


def extract_pdf_images(path: str | Path) -> list[dict]:
    import fitz

    records = []
    document = fitz.open(path)
    try:
        for page_index, page in enumerate(document, start=1):
            for image_index, image in enumerate(page.get_images(full=True), start=1):
                image_data = document.extract_image(image[0])
                records.append(
                    {
                        "page_number": page_index,
                        "image_number": image_index,
                        "image_bytes": image_data["image"],
                        "image_extension": image_data.get("ext", "bin"),
                    }
                )
    finally:
        document.close()
    return records


def extract_hwp_images(path: str | Path) -> list[dict]:
    import olefile

    path = Path(path)
    if not olefile.isOleFile(path):
        raise ValueError("Unsupported HWP 5.x OLE document.")

    records = []
    with olefile.OleFileIO(path) as hwp:
        for stream_path in hwp.listdir():
            if len(stream_path) != 2 or stream_path[0] != "BinData":
                continue

            payload = hwp.openstream(stream_path).read()
            for candidate in _binary_candidates(payload):
                image_bytes, extension = _find_image_payload(candidate)
                if image_bytes:
                    records.append(
                        {
                            "page_number": None,
                            "image_number": len(records) + 1,
                            "image_bytes": image_bytes,
                            "image_extension": extension,
                            "stream_name": "/".join(stream_path),
                        }
                    )
                    break
    return records


def _binary_candidates(payload: bytes) -> list[bytes]:
    candidates = [payload]
    try:
        candidates.append(zlib.decompress(payload, -15))
    except zlib.error:
        pass
    return candidates


def _find_image_payload(payload: bytes) -> tuple[bytes | None, str | None]:
    signatures = ((b"\x89PNG\r\n\x1a\n", "png"), (b"\xff\xd8\xff", "jpg"))
    matches = [(payload.find(signature), extension) for signature, extension in signatures]
    matches = [(index, extension) for index, extension in matches if index >= 0]
    if not matches:
        return None, None

    start, extension = min(matches)
    return payload[start:], extension


def image_dimensions(image_bytes: bytes) -> tuple[int, int]:
    from PIL import Image

    with Image.open(io.BytesIO(image_bytes)) as image:
        return image.size


def extract_ocr_text(
    image_bytes: bytes,
    language: str,
    *,
    page_seg_mode: int,
    image_scale: int,
) -> str:
    import pytesseract
    from PIL import Image, ImageOps

    with Image.open(io.BytesIO(image_bytes)) as image:
        prepared = ImageOps.autocontrast(image.convert("L"))
        if image_scale > 1:
            prepared = prepared.resize(
                (prepared.width * image_scale, prepared.height * image_scale),
                Image.Resampling.LANCZOS,
            )
        return pytesseract.image_to_string(
            prepared,
            lang=language,
            config=f"--psm {page_seg_mode}",
        ).strip()

def merge_ocr_text(document_text: str, ocr_texts: list[str]) -> str:
    usable_texts = [text.strip() for text in ocr_texts if text and text.strip()]
    if not usable_texts:
        return document_text.strip()

    return "\n\n".join([document_text.strip(), "[OCR image/table text]", *usable_texts]).strip()


def load_ocr_texts_by_doc_id(path: str | Path) -> dict[str, list[str]]:
    texts_by_doc_id: dict[str, list[str]] = {}
    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("status") != "applied":
                continue
            text = str(record.get("ocr_text", "")).strip()
            if text:
                texts_by_doc_id.setdefault(record["doc_id"], []).append(text)
    return texts_by_doc_id


def sha256_text(value: str | bytes) -> str:
    payload = value.encode("utf-8") if isinstance(value, str) else value
    return hashlib.sha256(payload).hexdigest()

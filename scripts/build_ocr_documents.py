import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ocr_extractor import (
    extract_hwp_images,
    extract_ocr_text,
    extract_pdf_images,
    image_dimensions,
    sha256_text,
)


REPORT_COLUMNS = [
    "doc_id",
    "file_name",
    "file_type",
    "page_number",
    "image_number",
    "image_extension",
    "image_sha256",
    "status",
    "reason",
    "width",
    "height",
    "text_length",
    "ocr_sha256",
]


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def extract_images(file_path: Path) -> list[dict]:
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return extract_pdf_images(file_path)
    if suffix == ".hwp":
        return extract_hwp_images(file_path)
    raise ValueError(f"Unsupported OCR file type: {suffix}")


def build_report_row(
    *,
    doc_id: str,
    file_name: str,
    file_type: str,
    image: dict,
    status: str,
    reason: str,
    width: int | None = None,
    height: int | None = None,
    ocr_text: str = "",
) -> dict:
    return {
        "doc_id": doc_id,
        "file_name": file_name,
        "file_type": file_type,
        "page_number": image.get("page_number"),
        "image_number": image.get("image_number"),
        "image_extension": image.get("image_extension"),
        "image_sha256": sha256_text(image["image_bytes"]),
        "status": status,
        "reason": reason,
        "width": width,
        "height": height,
        "text_length": len(ocr_text),
        "ocr_sha256": sha256_text(ocr_text) if ocr_text else "",
    }


def process_document(
    *,
    file_path: Path,
    doc_id: str,
    language: str,
    min_width: int,
    min_height: int,
    min_text_length: int,
    general_psm: int,
    image_scale: int,
) -> tuple[list[dict], list[dict]]:
    records = []
    report_rows = []
    file_name = file_path.name
    file_type = file_path.suffix.lower().lstrip(".")

    for image in extract_images(file_path):
        try:
            width, height = image_dimensions(image["image_bytes"])
            if width < min_width or height < min_height:
                report_rows.append(
                    build_report_row(
                        doc_id=doc_id,
                        file_name=file_name,
                        file_type=file_type,
                        image=image,
                        status="excluded",
                        reason="image_too_small",
                        width=width,
                        height=height,
                    )
                )
                continue

            ocr_text = extract_ocr_text(
                image["image_bytes"],
                language,
                page_seg_mode=general_psm,
                image_scale=image_scale,
            )
            if len(ocr_text) < min_text_length:
                report_rows.append(
                    build_report_row(
                        doc_id=doc_id,
                        file_name=file_name,
                        file_type=file_type,
                        image=image,
                        status="excluded",
                        reason="ocr_text_too_short",
                        width=width,
                        height=height,
                        ocr_text=ocr_text,
                    )
                )
                continue

            record = {
                "doc_id": doc_id,
                "file_name": file_name,
                "file_type": file_type,
                "page_number": image.get("page_number"),
                "image_number": image.get("image_number"),
                "image_extension": image.get("image_extension"),
                "image_sha256": sha256_text(image["image_bytes"]),
                "status": "applied",
                "ocr_text": ocr_text,
                "ocr_sha256": sha256_text(ocr_text),
            }
            records.append(record)
            report_rows.append(
                build_report_row(
                    doc_id=doc_id,
                    file_name=file_name,
                    file_type=file_type,
                    image=image,
                    status="applied",
                    reason="",
                    width=width,
                    height=height,
                    ocr_text=ocr_text,
                )
            )
        except Exception as error:
            report_rows.append(
                build_report_row(
                    doc_id=doc_id,
                    file_name=file_name,
                    file_type=file_type,
                    image=image,
                    status="failed",
                    reason=f"{type(error).__name__}: {error}",
                )
            )

    return records, report_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract OCR text from RFP images and tables")
    parser.add_argument("--config", default=PROJECT_ROOT / "config/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    paths = config["paths"]
    ocr_config = config["ocr"]
    metadata_path = resolve_path(paths["metadata"])
    raw_documents_path = resolve_path(paths["raw_documents"])
    output_path = resolve_path(paths["ocr_documents"])
    report_path = resolve_path(paths["ocr_report"])

    metadata_frame = pd.read_csv(metadata_path, encoding="utf-8")
    required_columns = {"\ud30c\uc77c\uba85", "\ud30c\uc77c\ud615\uc2dd"}
    missing_columns = required_columns - set(metadata_frame.columns)
    if missing_columns:
        raise ValueError(f"Missing metadata columns: {sorted(missing_columns)}")

    all_records = []
    all_report_rows = []
    for index, row in metadata_frame.iterrows():
        doc_id = f"doc_{index + 1:03d}"
        file_name = str(row["\ud30c\uc77c\uba85"]).strip()
        file_path = raw_documents_path / file_name
        if not file_path.is_file():
            all_report_rows.append(
                {
                    **dict.fromkeys(REPORT_COLUMNS, ""),
                    "doc_id": doc_id,
                    "file_name": file_name,
                    "file_type": str(row["\ud30c\uc77c\ud615\uc2dd"]),
                    "status": "failed",
                    "reason": "FileNotFoundError: source_document_not_found",
                }
            )
            continue

        try:
            records, report_rows = process_document(
                file_path=file_path,
                doc_id=doc_id,
                language=ocr_config["language"],
                min_width=ocr_config["min_width"],
                min_height=ocr_config["min_height"],
                min_text_length=ocr_config["min_text_length"],
                general_psm=ocr_config["general_psm"],
                image_scale=ocr_config["image_scale"],
            )
            all_records.extend(records)
            all_report_rows.extend(report_rows)
        except Exception as error:
            all_report_rows.append(
                {
                    **dict.fromkeys(REPORT_COLUMNS, ""),
                    "doc_id": doc_id,
                    "file_name": file_name,
                    "file_type": str(row["\ud30c\uc77c\ud615\uc2dd"]),
                    "status": "failed",
                    "reason": f"{type(error).__name__}: {error}",
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for record in all_records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_report_rows, columns=REPORT_COLUMNS).to_csv(
        report_path, index=False, encoding="utf-8"
    )
    status_counts = pd.Series([row["status"] for row in all_report_rows]).value_counts()
    print(f"OCR applied: {status_counts.get('applied', 0)}")
    print(f"OCR excluded: {status_counts.get('excluded', 0)}")
    print(f"OCR failed: {status_counts.get('failed', 0)}")
    print(f"OCR text output: {output_path}")
    print(f"OCR report output: {report_path}")


if __name__ == "__main__":
    main()

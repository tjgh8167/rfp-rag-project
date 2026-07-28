import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ocr_extractor import extract_hwp_images, extract_pdf_images, image_dimensions, sha256_text
from src.table_extractor import extract_image_table_markdown, extract_pdf_tables, extract_table_ocr, is_table_image


REPORT_COLUMNS = [
    "doc_id",
    "file_name",
    "file_type",
    "source_type",
    "page_number",
    "image_number",
    "table_number",
    "status",
    "reason",
    "width",
    "height",
    "text_length",
    "extraction_method",
]


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def extract_images(file_path: Path) -> list[dict]:
    if file_path.suffix.lower() == ".pdf":
        return extract_pdf_images(file_path)
    if file_path.suffix.lower() == ".hwp":
        return extract_hwp_images(file_path)
    raise ValueError(f"Unsupported file type: {file_path.suffix}")


# 표로 처리하지 않고 건너뛴 이미지도 사유를 남겨야 전체 이미지 수가 리포트에서 맞는다.
def skipped_image_row(
    doc_id: str,
    file_name: str,
    file_type: str,
    image: dict,
    reason: str,
    width: int,
    height: int,
    status: str = "excluded",
) -> dict:
    return {
        "doc_id": doc_id,
        "file_name": file_name,
        "file_type": file_type,
        "source_type": "table_image",
        "page_number": image.get("page_number"),
        "image_number": image.get("image_number"),
        "table_number": None,
        "status": status,
        "reason": reason,
        "width": width,
        "height": height,
        "text_length": 0,
        "extraction_method": "",
    }


def process_document(file_path: Path, doc_id: str, ocr_config: dict) -> tuple[list[dict], list[dict]]:
    file_name = file_path.name
    file_type = file_path.suffix.lower().lstrip(".")
    records = []
    report_rows = []

    if file_type == "pdf":
        try:
            for table in extract_pdf_tables(file_path):
                markdown = table["table_markdown"]
                records.append(
                    {
                        "doc_id": doc_id,
                        "file_name": file_name,
                        "file_type": file_type,
                        "source_type": "native_pdf_table",
                        "page_number": table["page_number"],
                        "table_number": table["table_number"],
                        "status": "extracted",
                        "table_markdown": markdown,
                        "table_sha256": sha256_text(markdown),
                        "extraction_method": "pymupdf_find_tables",
                    }
                )
                report_rows.append(
                    {
                        "doc_id": doc_id,
                        "file_name": file_name,
                        "file_type": file_type,
                        "source_type": "native_pdf_table",
                        "page_number": table["page_number"],
                        "image_number": None,
                        "table_number": table["table_number"],
                        "status": "extracted",
                        "reason": "",
                        "width": None,
                        "height": None,
                        "text_length": len(markdown),
                        "extraction_method": "pymupdf_find_tables",
                    }
                )
        except Exception as error:
            report_rows.append(
                {
                    "doc_id": doc_id,
                    "file_name": file_name,
                    "file_type": file_type,
                    "source_type": "native_pdf_table",
                    "page_number": None,
                    "image_number": None,
                    "table_number": None,
                    "status": "failed",
                    "reason": f"{type(error).__name__}: {error}",
                    "width": None,
                    "height": None,
                    "text_length": 0,
                    "extraction_method": "pymupdf_find_tables",
                }
            )

    for image in extract_images(file_path):
        try:
            width, height = image_dimensions(image["image_bytes"])
        except Exception as error:
            report_rows.append(
                {
                    "doc_id": doc_id,
                    "file_name": file_name,
                    "file_type": file_type,
                    "source_type": "table_image",
                    "page_number": image.get("page_number"),
                    "image_number": image.get("image_number"),
                    "table_number": None,
                    "status": "failed",
                    "reason": f"{type(error).__name__}: {error}",
                    "width": None,
                    "height": None,
                    "text_length": 0,
                    "extraction_method": "image_decode",
                }
            )
            continue
        if width < ocr_config["min_width"] or height < ocr_config["min_height"]:
            report_rows.append(
                skipped_image_row(
                    doc_id, file_name, file_type, image, "image_too_small", width, height
                )
            )
            continue
        # 픽셀을 읽는 단계라 크기 확인을 통과한 이미지도 깨져 있을 수 있다.
        # 한 장 때문에 100건 실행이 중단되지 않도록 사유를 남기고 넘어간다.
        try:
            table_like = is_table_image(
                image["image_bytes"],
                dark_pixel_threshold=ocr_config["table_dark_pixel_threshold"],
                line_density=ocr_config["table_line_density"],
                min_line_run_length=ocr_config["table_min_line_run_length"],
                min_grid_lines=ocr_config["table_grid_min_lines"],
            )
            if not table_like:
                report_rows.append(
                    skipped_image_row(
                        doc_id, file_name, file_type, image, "not_table_image", width, height
                    )
                )
                continue

            table_markdown = extract_image_table_markdown(
                image["image_bytes"],
                ocr_config["language"],
                page_seg_mode=ocr_config["cell_psm"],
                image_scale=ocr_config["image_scale"],
                line_density=ocr_config["table_line_density"],
                dark_pixel_threshold=ocr_config["table_dark_pixel_threshold"],
                min_line_run_length=ocr_config["table_min_line_run_length"],
                min_grid_lines=ocr_config["table_cell_grid_min_lines"],
                cell_crop_margin=ocr_config["table_cell_crop_margin"],
                min_cell_width=ocr_config["min_cell_width"],
                min_cell_height=ocr_config["min_cell_height"],
            )
            text = table_markdown or extract_table_ocr(
                image["image_bytes"],
                ocr_config["language"],
                ocr_config["table_psm"],
                ocr_config["image_scale"],
            )
        except Exception as error:
            report_rows.append(
                skipped_image_row(
                    doc_id,
                    file_name,
                    file_type,
                    image,
                    f"{type(error).__name__}: {error}",
                    width,
                    height,
                    status="failed",
                )
            )
            continue
        # 레코드와 리포트가 같은 값을 쓰도록 한 곳에서만 정한다.
        extraction_method = (
            f"cell_ocr_psm_{ocr_config['cell_psm']}"
            if table_markdown
            else f"tesseract_psm_{ocr_config['table_psm']}"
        )
        status = "review_required" if len(text) >= ocr_config["min_text_length"] else "excluded"
        reason = "table_ocr_requires_review" if status == "review_required" else "ocr_text_too_short"
        record = {
            "doc_id": doc_id,
            "file_name": file_name,
            "file_type": file_type,
            "source_type": "table_image",
            "page_number": image.get("page_number"),
            "image_number": image.get("image_number"),
            "status": status,
            "table_markdown": table_markdown,
            "table_ocr_text": text if not table_markdown else "",
            "text_sha256": sha256_text(text) if text else "",
            "extraction_method": extraction_method,
        }
        if status == "review_required":
            records.append(record)
        report_rows.append(
            {
                "doc_id": doc_id,
                "file_name": file_name,
                "file_type": file_type,
                "source_type": "table_image",
                "page_number": image.get("page_number"),
                "image_number": image.get("image_number"),
                "table_number": None,
                "status": status,
                "reason": reason,
                "width": width,
                "height": height,
                "text_length": len(text),
                "extraction_method": extraction_method,
            }
        )

    return records, report_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract structured PDF tables and table-image OCR results")
    parser.add_argument("--config", default=PROJECT_ROOT / "config/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    paths = config["paths"]
    metadata = pd.read_csv(paths["metadata"], encoding="utf-8")

    records = []
    report_rows = []
    raw_documents = Path(paths["raw_documents"])
    for index, row in metadata.iterrows():
        file_name = str(row["파일명"]).strip()
        file_path = raw_documents / file_name
        if not file_path.is_file():
            continue
        doc_records, doc_report_rows = process_document(
            file_path,
            f"doc_{index + 1:03d}",
            config["ocr"],
        )
        records.extend(doc_records)
        report_rows.extend(doc_report_rows)

    output_path = Path(paths["table_documents"])
    report_path = Path(paths["table_report"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    pd.DataFrame(report_rows, columns=REPORT_COLUMNS).to_csv(
        report_path,
        index=False,
        encoding="utf-8",
    )
    print(f"Structured table/table-image extraction: {len(records)}")
    print(f"Output: {output_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()

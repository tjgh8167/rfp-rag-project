import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.parser_chunker import (
    extract_hwp_tables,
    hwp_table_density,
    hwp_table_to_markdown,
    is_hwp_content_table,
)

REPORT_COLUMNS = [
    "doc_id",
    "file_name",
    "table_number",
    "rows",
    "columns",
    "depth",
    "cells_found",
    "cells_filled",
    "density",
    "text_length",
    "status",
    "reason",
]


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def drop_reason(table: dict, options: dict) -> str:
    if table["rows"] < options["min_rows"] or table["columns"] < options["min_columns"]:
        return "layout_table"
    return "low_density"


def process_document(file_path: Path, doc_id: str, options: dict) -> tuple[list[dict], list[dict]]:
    records: list[dict] = []
    report_rows: list[dict] = []

    for table_number, table in enumerate(extract_hwp_tables(file_path), start=1):
        markdown = hwp_table_to_markdown(table)
        density = hwp_table_density(table)
        filled = sum(1 for cell in table["cells"] if cell["text"])
        keep = is_hwp_content_table(
            table, options["min_rows"], options["min_columns"], options["min_density"]
        )
        # 제외한 표도 사유와 함께 남겨야 기준을 나중에 조정할 수 있습니다.
        report_rows.append(
            {
                "doc_id": doc_id,
                "file_name": file_path.name,
                "table_number": table_number,
                "rows": table["rows"],
                "columns": table["columns"],
                "depth": table["depth"],
                "cells_found": len(table["cells"]),
                "cells_filled": filled,
                "density": round(density, 3),
                "text_length": len(markdown),
                "status": "extracted" if keep else "excluded",
                "reason": "" if keep else drop_reason(table, options),
            }
        )
        if not keep:
            continue

        records.append(
            {
                "doc_id": doc_id,
                "file_name": file_path.name,
                "file_type": "hwp",
                "source_type": "native_hwp_table",
                "table_number": table_number,
                "rows": table["rows"],
                "columns": table["columns"],
                "depth": table["depth"],
                "density": round(density, 3),
                "table_markdown": markdown,
                "extraction_method": "hwp_record_parsing",
                "review_required": True,
            }
        )

    return records, report_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=PROJECT_ROOT / "config" / "default.yaml")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    config = load_config(args.config)
    paths = config["paths"]
    options = config["hwp_table"]
    metadata = pd.read_csv(paths["metadata"], encoding="utf-8")
    raw_documents = Path(paths["raw_documents"])
    if args.limit is not None:
        metadata = metadata.head(args.limit)

    records: list[dict] = []
    report: list[dict] = []

    for index, row in metadata.iterrows():
        doc_id = f"doc_{index + 1:03d}"
        file_path = raw_documents / str(row["파일명"]).strip()
        if file_path.suffix.lower() != ".hwp":
            continue
        if not file_path.is_file():
            report.append(
                {
                    "doc_id": doc_id,
                    "file_name": file_path.name,
                    "status": "failed",
                    "reason": "source_file_not_found",
                }
            )
            continue

        try:
            document_records, document_report = process_document(file_path, doc_id, options)
        except Exception as error:
            report.append(
                {
                    "doc_id": doc_id,
                    "file_name": file_path.name,
                    "status": "failed",
                    "reason": f"{type(error).__name__}: {error}",
                }
            )
            continue

        records.extend(document_records)
        report.extend(document_report)
        print(f"{doc_id} 표 {len(document_report)}개 중 {len(document_records)}개 채택", flush=True)

    documents_path = Path(paths["hwp_table_documents"])
    report_path = Path(paths["hwp_table_report"])
    documents_path.parent.mkdir(parents=True, exist_ok=True)

    with documents_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    pd.DataFrame(report, columns=REPORT_COLUMNS).to_csv(report_path, index=False, encoding="utf-8")

    excluded = sum(1 for row in report if row["status"] == "excluded")
    failed = sum(1 for row in report if row["status"] == "failed")
    print()
    print(f"표 {len(report):,}개 중 채택 {len(records):,}개 / 제외 {excluded:,}개 / 실패 {failed:,}개")
    print(f"{documents_path}")
    print(f"{report_path}")


if __name__ == "__main__":
    main()

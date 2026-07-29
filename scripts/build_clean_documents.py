import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml

from src.parser_chunker import read_document
from src.text_cleaner import clean_document_text, verify_preservation


PROJECT_ROOT = Path(__file__).resolve().parents[1]
# 클리닝 후에도 반드시 남아 있어야 하는 핵심 업무 정보 열입니다.
# 입찰 마감일은 CSV가 "2024-10-15 17:00:00" 형식이라 문서 표기와 일치하지 않아 제외하고,
# 대신 숫자 자릿수 대조(unexplained_digit_loss)로 일정·금액 손실을 잡습니다.
PRESERVE_COLUMNS = ("사업명", "발주 기관", "사업 금액")
REPORT_COLUMNS = (
    "doc_id",
    "file_name",
    "status",
    "original_length",
    "cleaned_length",
    "pua_replaced",
    "invalid_removed",
    "space_normalized",
    "duplicate_paragraphs_removed",
    "duplicate_digits_removed",
    "blank_lines_removed",
    "requirement_id_count",
    "lost_requirement_ids",
    "digit_count_before",
    "digit_count_after",
    "unexplained_digit_loss",
    "checked_values",
    "lost_metadata_values",
    "preserved",
    "error",
)


# YAML 설정 파일을 읽어 경로와 클리닝 설정을 반환합니다.
def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8-sig") as file:
        return yaml.safe_load(file)


# 상대 경로를 프로젝트 루트 기준의 절대 경로로 변환합니다.
def resolve_project_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


# 메타데이터 행에서 보존 여부를 확인할 값들을 문서에 쓰인 표기 형태로 뽑습니다.
def preservation_values(row: pd.Series) -> list[str]:
    values = [row["사업명"], row["발주 기관"]]

    # 문서에는 금액이 천 단위 구분 기호와 함께 적히므로 두 표기를 모두 후보로 둔다.
    budget = row["사업 금액"]
    if not pd.isna(budget):
        values.extend([f"{int(budget):,}", str(int(budget))])

    return [
        str(value).strip()
        for value in values
        if not pd.isna(value) and str(value).strip()
    ]


# 원본 문서 100건을 클리닝해 cleaned_documents.jsonl과 cleaning_report.csv로 저장합니다.
def main() -> None:
    parser = argparse.ArgumentParser(description="원본 RFP 본문을 클리닝해 JSONL로 저장")
    parser.add_argument("--config", default=PROJECT_ROOT / "config/default.yaml")
    # 표를 표 청크로 따로 관리할 때 씁니다. 본문에 표 글자가 남으면 같은 내용이 두 벌이 됩니다.
    parser.add_argument(
        "--exclude-tables",
        action="store_true",
        help="HWP 본문에서 표 안 글자를 빼고 cleaned_documents_no_table 경로에 저장",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    metadata_path = resolve_project_path(config["paths"]["metadata"])
    raw_documents_path = resolve_project_path(config["paths"]["raw_documents"])
    output_path = resolve_project_path(
        config["paths"]["cleaned_documents_no_table"]
        if args.exclude_tables
        else config["paths"]["cleaned_documents"]
    )
    report_path = resolve_project_path(config["paths"]["cleaning_report"])
    max_blank_lines = config["cleaning"]["max_blank_lines"]
    min_duplicate_paragraph_length = config["cleaning"]["min_duplicate_paragraph_length"]
    requirement_id_pattern = config["cleaning"]["requirement_id_pattern"]

    metadata_frame = pd.read_csv(metadata_path, encoding="utf-8-sig")
    missing_columns = sorted(set(PRESERVE_COLUMNS) - set(metadata_frame.columns))
    if missing_columns:
        raise ValueError(f"메타데이터 CSV에 필요한 열이 없습니다: {missing_columns}")

    documents = []
    report_rows = []

    for index, row in metadata_frame.iterrows():
        doc_id = f"doc_{index + 1:03d}"
        file_name = "" if pd.isna(row["파일명"]) else str(row["파일명"]).strip()
        record = {"doc_id": doc_id, "file_name": file_name, "status": "ok", "error": ""}

        try:
            original_text = read_document(
                raw_documents_path / file_name,
                exclude_tables=args.exclude_tables,
                table_options=config["hwp_table"],
            )
            cleaned_text, stats = clean_document_text(
                original_text,
                max_blank_lines=max_blank_lines,
                min_duplicate_paragraph_length=min_duplicate_paragraph_length,
            )
            if not cleaned_text.strip():
                raise ValueError("클리닝 후 본문이 비어 있습니다.")

            verification = verify_preservation(
                original_text,
                cleaned_text,
                preservation_values(row),
                requirement_id_pattern,
                stats["duplicate_digits_removed"],
            )
            record.update(stats)
            record.update(verification)

            documents.append(
                {
                    "doc_id": doc_id,
                    "file_name": file_name,
                    "text": cleaned_text,
                    "text_sha256": hashlib.sha256(cleaned_text.encode("utf-8")).hexdigest(),
                }
            )
        except Exception as error:
            record["status"] = "error"
            record["error"] = f"{type(error).__name__}: {error}"

        report_rows.append(record)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for document in documents:
            file.write(json.dumps(document, ensure_ascii=False) + "\n")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = pd.DataFrame(report_rows, columns=REPORT_COLUMNS)
    report.to_csv(report_path, index=False, encoding="utf-8-sig")

    failures = report[report["status"] == "error"]
    not_preserved = report[(report["status"] == "ok") & (~report["preserved"].astype(bool))]

    print(f"클리닝 문서: {len(documents)}/{len(metadata_frame)}건")
    print(f"제거한 PUA 글머리 기호: {report['pua_replaced'].sum():,.0f}자")
    print(f"제거한 제어·미할당 문자: {report['invalid_removed'].sum():,.0f}자")
    print(f"정규화한 특수 공백: {report['space_normalized'].sum():,.0f}자")
    print(f"제거한 연속 중복 문단: {report['duplicate_paragraphs_removed'].sum():,.0f}줄")
    print(f"제거한 빈 줄: {report['blank_lines_removed'].sum():,.0f}줄")
    print(f"본문 길이: {report['original_length'].sum():,.0f} -> {report['cleaned_length'].sum():,.0f}자")
    print(f"저장 경로: {output_path}")
    print(f"클리닝 리포트: {report_path}")
    print(f"핵심 정보 검증: {report['checked_values'].sum():,.0f}개 값 대조, 보존 실패 {len(not_preserved)}건")
    print(f"클리닝 실패: {len(failures)}건")

    if len(not_preserved):
        print("\n핵심 정보가 사라진 문서")
        for _, row in not_preserved.iterrows():
            print(f"- {row['doc_id']} | {row['file_name']} | {row['lost_metadata_values']} | {row['lost_requirement_ids']}")

    if len(failures):
        raise RuntimeError(f"클리닝에 실패한 문서가 {len(failures)}건 있습니다.")


if __name__ == "__main__":
    main()

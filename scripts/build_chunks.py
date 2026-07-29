import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from src.parser_chunker import (
    build_chunks_from_text,
    load_chunks_jsonl,
    save_chunks_jsonl,
    validate_chunk_contract,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METADATA_COLUMNS = {
    "공고 번호": "notice_number",
    "공고 차수": "notice_round",
    "사업명": "title",
    "사업 금액": "budget",
    "발주 기관": "agency",
    "공개 일자": "published_at",
    "입찰 참여 시작일": "bid_start_at",
    "입찰 참여 마감일": "bid_end_at",
    "사업 요약": "summary",
    "파일형식": "file_type",
    "파일명": "file_name",
}


# YAML 설정 파일을 읽어 경로와 청킹 설정을 반환합니다.
def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8-sig") as file:
        return yaml.safe_load(file)


# 상대 경로를 프로젝트 루트 기준의 절대 경로로 변환합니다.
def resolve_project_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


# CSV 한 행에서 비어 있지 않은 메타데이터만 공통 영문 키로 변환합니다.
def build_metadata(row: pd.Series) -> dict:
    metadata = {}
    for source_column, target_key in METADATA_COLUMNS.items():
        value = row[source_column]
        if pd.isna(value):
            continue
        if hasattr(value, "item"):
            value = value.item()
        metadata[target_key] = value

    # 청크 계약(REQUIRED_METADATA_FIELDS)은 사업명을 project_name으로도 요구한다.
    if "title" in metadata:
        metadata["project_name"] = metadata["title"]
    return metadata


# 클리닝된 본문 JSONL을 읽어 doc_id별 본문으로 반환합니다.
def load_cleaned_documents(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        documents = [json.loads(line) for line in file if line.strip()]
    if not documents:
        raise ValueError(f"클리닝된 문서가 없습니다: {path}")
    return {document["doc_id"]: document for document in documents}


# 저장된 JSONL이 필수 형식과 처리 문서 수를 만족하는지 검증합니다.
def validate_chunks(output_path: Path, expected_document_count: int) -> list[dict]:
    loaded_chunks = load_chunks_jsonl(output_path)
    required_keys = {"chunk_id", "doc_id", "text", "metadata"}

    if not loaded_chunks:
        raise ValueError("생성된 청크가 없습니다.")

    invalid_chunks = [
        chunk
        for chunk in loaded_chunks
        if not required_keys.issubset(chunk) or not chunk["text"].strip()
    ]
    if invalid_chunks:
        raise ValueError(f"필수 키가 없거나 텍스트가 빈 청크가 {len(invalid_chunks)}개 있습니다.")

    document_count = len({chunk["doc_id"] for chunk in loaded_chunks})
    if document_count != expected_document_count:
        raise ValueError(
            f"청크의 문서 수가 예상과 다릅니다: {document_count}/{expected_document_count}"
        )

    return loaded_chunks


# 실제 메타데이터와 원본 문서 전체를 읽어 하나의 chunks.jsonl로 저장합니다.
def main() -> None:
    parser = argparse.ArgumentParser(description="실제 RFP 문서를 JSONL 청크로 변환")
    parser.add_argument("--config", default=PROJECT_ROOT / "config/default.yaml")
    parser.add_argument("--output", help="설정 파일의 출력 경로 대신 사용할 JSONL 경로")
    args = parser.parse_args()

    config = load_config(args.config)
    metadata_path = resolve_project_path(config["paths"]["metadata"])
    raw_documents_path = resolve_project_path(config["paths"]["raw_documents"])
    output_path = resolve_project_path(args.output or config["paths"]["chunks"])
    cleaned_documents_path = resolve_project_path(config["paths"]["cleaned_documents"])
    failure_log_path = resolve_project_path(config["paths"]["extraction_failures"])
    chunk_size = config["chunking"]["chunk_size"]
    chunk_overlap = config["chunking"]["chunk_overlap"]

    cleaned_documents = load_cleaned_documents(cleaned_documents_path)
    metadata_frame = pd.read_csv(metadata_path, encoding="utf-8-sig")
    missing_columns = sorted(set(METADATA_COLUMNS) - set(metadata_frame.columns))
    if missing_columns:
        raise ValueError(f"메타데이터 CSV에 필요한 열이 없습니다: {missing_columns}")

    all_chunks = []
    failures = []
    successful_document_count = 0

    for index, row in metadata_frame.iterrows():
        doc_id = f"doc_{index + 1:03d}"

        try:
            if pd.isna(row["파일명"]):
                raise ValueError("메타데이터의 파일명이 비어 있습니다.")

            file_name = str(row["파일명"]).strip()
            if doc_id not in cleaned_documents:
                raise ValueError(f"클리닝된 본문이 없습니다: {cleaned_documents_path}")

            document_chunks = build_chunks_from_text(
                cleaned_documents[doc_id]["text"],
                doc_id,
                {
                    "file_name": file_name,
                    "source_path": str(raw_documents_path / file_name),
                    **build_metadata(row),
                },
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            if not document_chunks:
                raise ValueError("추출된 텍스트가 비어 있어 청크를 만들 수 없습니다.")

            all_chunks.extend(document_chunks)
            successful_document_count += 1
        except Exception as error:
            failures.append(
                {
                    "doc_id": doc_id,
                    "file_name": "" if pd.isna(row["파일명"]) else str(row["파일명"]),
                    "status": "error",
                    "error": f"{type(error).__name__}: {error}",
                }
            )

    save_chunks_jsonl(all_chunks, output_path)
    failure_log_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        failures, columns=["doc_id", "file_name", "status", "error"]
    ).to_csv(failure_log_path, index=False, encoding="utf-8-sig")

    loaded_chunks = validate_chunks(output_path, successful_document_count)
    validate_chunk_contract(loaded_chunks)
    print(f"클리닝 본문: {cleaned_documents_path}")
    print(f"처리 문서: {successful_document_count}/{len(metadata_frame)}건")
    print(f"생성 청크: {len(loaded_chunks)}개")
    print(f"청크 설정: size={chunk_size}, overlap={chunk_overlap}")
    print(f"저장 경로: {output_path}")
    print(f"실패 문서: {len(failures)}건 ({failure_log_path})")

    sample_doc_ids = []
    for chunk in loaded_chunks:
        if chunk["doc_id"] not in sample_doc_ids:
            sample_doc_ids.append(chunk["doc_id"])
        if len(sample_doc_ids) == 3:
            break

    print("\n실제 문서 청크 예시")
    for doc_id in sample_doc_ids:
        sample = next(chunk for chunk in loaded_chunks if chunk["doc_id"] == doc_id)
        print(f"- {doc_id} | {sample['metadata']['file_name']} | {sample['text'][:120]}")

    if failures:
        raise RuntimeError(f"청킹에 실패한 문서가 {len(failures)}건 있습니다.")


if __name__ == "__main__":
    main()

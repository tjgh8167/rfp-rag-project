"""본문·표·이미지를 하나의 청크 JSONL로 만듭니다.

기존 chunks는 그대로 두고 별도 파일로 저장해, 추출 미적용본과 검색 품질을 비교할 수 있게 합니다.
표는 행 단위, 이미지는 한 장에 한 청크로 나눕니다. 본문 청킹 방식은 기존과 같습니다.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_chunks import METADATA_COLUMNS, build_metadata, load_config, resolve_project_path
from src.parser_chunker import (
    Chunk,
    build_chunks_from_text,
    chunk_text,
    chunk_image,
    chunk_table,
    load_chunks_jsonl,
    save_chunks_jsonl,
    validate_chunk_contract,
)


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def group_by_doc(records: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = {}
    for record in records:
        grouped.setdefault(record["doc_id"], []).append(record)
    return grouped


# 절 제목을 청크 앞에 세웁니다.
# 청크만 보면 어느 항목인지 알 수 없어, 문장으로 들어오는 질문과 임베딩 거리가 멉니다.
# 잘린 청크마다 제목을 반복하는 것은 표 청크에서 헤더를 반복하는 것과 같은 이유입니다.
def body_chunks(document: dict, doc_id: str, metadata: dict, chunking: dict) -> list[Chunk]:
    sections = document.get("sections") if chunking["use_section_titles"] else None
    if not sections:
        return build_chunks_from_text(
            document["text"],
            doc_id,
            metadata,
            chunk_size=chunking["chunk_size"],
            chunk_overlap=chunking["chunk_overlap"],
        )

    chunks = []
    number = 0
    for section in sections:
        title = section["title"]
        for part in chunk_text(
            section["text"], chunking["chunk_size"], chunking["chunk_overlap"]
        ):
            number += 1
            chunk_metadata = {**metadata}
            if title:
                chunk_metadata["section_title"] = title
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}_chunk_{number:04d}",
                    doc_id=doc_id,
                    text=f"[{title}]\n{part}" if title else part,
                    metadata=chunk_metadata,
                )
            )
    return chunks


# 표 청크는 마크다운 격자뿐이라 어느 기관의 무슨 사업인지 텍스트에 없습니다.
# 임베딩은 text만 반영하므로 기관명을 언급하는 질문에서 표가 검색되지 않습니다.
def table_headline(metadata: dict) -> str:
    parts = [metadata.get("agency", ""), metadata.get("title", "")]
    parts = [part for part in parts if part]
    return f"[{' · '.join(parts)}]\n" if parts else ""


def table_chunks(
    records: list[dict],
    doc_id: str,
    metadata: dict,
    max_chars: int,
    headline: str = "",
) -> list[Chunk]:
    """표 하나가 상한을 넘으면 행 단위로 나뉘어 여러 청크가 됩니다."""
    chunks = []
    number = 0
    for record in records:
        markdown = record.get("table_markdown", "")
        for part in chunk_table(markdown, max_chars):
            number += 1
            chunk_metadata = {
                **metadata,
                "chunk_source": "table",
                "extraction_method": record["extraction_method"],
            }
            # 값이 없는 키는 Chroma metadata에 넣지 않습니다.
            for key in ("page_number", "table_number"):
                value = record.get(key)
                if value is not None:
                    chunk_metadata[key] = value
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}_table_{number:04d}",
                    doc_id=doc_id,
                    text=headline + part,
                    metadata=chunk_metadata,
                )
            )
    return chunks


# 이미지 청크 첫 줄에 세울 한 문장을 만듭니다.
# 뒤따르는 OCR 결과는 맥락 없는 낱말 나열이라, 이것만으로는 문장으로 들어오는 질문과
# 임베딩 거리가 멉니다. 이미지가 무엇인지 먼저 밝혀 검색에 걸리게 합니다.
def image_headline(record: dict) -> str:
    summary = (record.get("vlm_text") or "").replace("\n", " ").strip()
    # vlm_text는 "유형: diagram 요약: ..." 형태라 요약 부분만 씁니다.
    if "요약:" in summary:
        summary = summary.split("요약:", 1)[1].strip()
    image_type = record.get("image_type", "unclassified")
    if not summary:
        return f"[{image_type}]\n"
    return f"[{image_type}] {summary}\n"


def image_chunks(records: list[dict], doc_id: str, metadata: dict, max_chars: int) -> list[Chunk]:
    chunks = []
    number = 0
    for record in records:
        # combined_context에 OCR 텍스트와 공간 배치, 유형·요약이 함께 들어 있습니다.
        text = record.get("combined_context") or record.get("ocr_text") or ""
        for part in chunk_image(image_headline(record) + text, max_chars):
            number += 1
            chunk_metadata = {
                **metadata,
                "chunk_source": "image",
                "image_type": record.get("image_type", "unclassified"),
                "ocr_engine": record.get("ocr_engine", ""),
            }
            for key in ("page_number", "image_number"):
                value = record.get(key)
                if value is not None:
                    chunk_metadata[key] = value
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}_img_{number:04d}",
                    doc_id=doc_id,
                    text=part,
                    metadata=chunk_metadata,
                )
            )
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="본문·표·이미지를 합친 청크 JSONL 생성")
    parser.add_argument("--config", default=PROJECT_ROOT / "config/default.yaml")
    parser.add_argument("--output", help="설정의 chunks_with_extraction 대신 사용할 경로")
    args = parser.parse_args()

    config = load_config(args.config)
    paths = config["paths"]
    chunking = config["chunking"]

    metadata_frame = pd.read_csv(resolve_project_path(paths["metadata"]), encoding="utf-8-sig")
    raw_documents_path = resolve_project_path(paths["raw_documents"])
    output_path = resolve_project_path(args.output or paths["chunks_with_extraction"])

    cleaned = {
        record["doc_id"]: record
        for record in load_jsonl(resolve_project_path(paths["cleaned_documents_no_table"]))
    }
    if not cleaned:
        raise ValueError(
            "표를 뺀 본문이 없습니다. "
            "scripts.build_clean_documents --exclude-tables 를 먼저 실행하세요."
        )

    pdf_tables = group_by_doc(load_jsonl(resolve_project_path(paths["table_documents"])))
    hwp_tables = group_by_doc(load_jsonl(resolve_project_path(paths["hwp_table_documents"])))
    images = group_by_doc(load_jsonl(resolve_project_path(paths["multimodal_documents"])))

    all_chunks: list[Chunk] = []
    counts = {"body": 0, "table": 0, "image": 0}

    for index, row in metadata_frame.iterrows():
        doc_id = f"doc_{index + 1:03d}"
        if doc_id not in cleaned:
            continue

        file_name = str(row["파일명"]).strip()
        metadata = {
            "file_name": file_name,
            "source_path": str(raw_documents_path / file_name),
            **build_metadata(row),
        }

        body = body_chunks(cleaned[doc_id], doc_id, metadata, chunking)
        # PDF 표는 텍스트 레이어에서, HWP 표는 본문 레코드에서 나옵니다. 한 문서에 둘 다 있지는 않습니다.
        tables = table_chunks(
            pdf_tables.get(doc_id, []) + hwp_tables.get(doc_id, []),
            doc_id,
            metadata,
            chunking["table_chunk_size"],
            table_headline(metadata) if chunking["use_table_headline"] else "",
        )
        pictures = image_chunks(
            images.get(doc_id, []), doc_id, metadata, chunking["image_chunk_size"]
        )

        counts["body"] += len(body)
        counts["table"] += len(tables)
        counts["image"] += len(pictures)
        all_chunks.extend(body + tables + pictures)

    save_chunks_jsonl(all_chunks, output_path)
    loaded = load_chunks_jsonl(output_path)
    validate_chunk_contract(loaded)

    print(f"본문 청크  {counts['body']:,}개")
    print(f"표 청크    {counts['table']:,}개")
    print(f"이미지 청크 {counts['image']:,}개")
    print(f"합계       {len(loaded):,}개")
    print(f"문서       {len({chunk['doc_id'] for chunk in loaded})}건")
    print(f"저장 경로  {output_path}")


if __name__ == "__main__":
    main()

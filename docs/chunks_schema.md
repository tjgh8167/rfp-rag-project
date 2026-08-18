# 청크와 검색 결과 계약

## 목적

Data, Retriever, Generation이 같은 JSON 구조를 사용한다. 실제 RFP 원문은 Git에 저장하지 않고, 청크 JSONL의 구조와 검색 결과만 이 계약의 대상으로 한다.

## 청크 JSONL 입력

파일은 한 줄에 한 청크를 저장하는 JSONL 형식이다. 모든 청크에는 아래 최상위 필드가 반드시 있어야 한다.

| 필드 | 자료형 | 설명 |
| --- | --- | --- |
| `chunk_id` | string | 청크의 고유 식별자. 예: `doc_001_chunk_0001` |
| `doc_id` | string | 원본 RFP 문서 식별자. 예: `doc_001` |
| `text` | string | 임베딩 및 검색 대상 본문. 빈 문자열은 허용하지 않음 |
| `metadata` | object | 문서 공통 정보. 검색 필터와 출처 표시에 사용 |

```json
{
  "chunk_id": "doc_001_chunk_0001",
  "doc_id": "doc_001",
  "text": "제안서는 나라장터를 통해 온라인으로 제출해야 한다.",
  "metadata": {
    "title": "차세대 포털 구축 사업",
    "project_name": "차세대 포털 구축 사업",
    "agency": "고려대학교",
    "file_name": "고려대학교_차세대 포털 구축 사업.pdf",
    "document_type": "pdf",
    "source_path": "/data/original_data/files/고려대학교_차세대 포털 구축 사업.pdf"
  }
}
```

`metadata`의 표준 전체 목록과 생성 기준은 [metadata_schema.md](metadata_schema.md)를 따른다. Retriever의 기관·사업명 필터는 `chunk["metadata"]` 내부의 `agency`, `title`, `project_name`을 사용한다.

### metadata 필수 필드

모든 청크의 `metadata`에는 아래 필드가 비어 있지 않은 문자열로 있어야 한다.

| 필드 | 용도 |
| --- | --- |
| `title` | Generation 출처에 우선 표시할 문서 제목 |
| `project_name` | 사업명 기반 질의와 기존 코드 호환 |
| `agency` | 발주 기관 필터 |
| `file_name` | 원본 파일 출처 표시 |

## SearchResult 출력

Retriever는 `src/retriever.py`의 `SearchResult`와 같은 필드를 반환한다. 청크의 네 필드를 유지하고, 검색 유사도인 `score`를 추가한다.

| 필드 | 자료형 | 설명 |
| --- | --- | --- |
| `chunk_id` | string | 검색된 청크 식별자 |
| `doc_id` | string | 원본 문서 식별자 |
| `text` | string | Generation에 전달할 검색 본문 |
| `metadata` | object | 출처 표시와 후속 필터에 사용할 문서 정보 |
| `score` | float | Retriever가 계산한 관련도 점수 |

`score`는 청크 JSONL에 저장하지 않는다. 임베딩 모델과 검색 방식마다 값의 범위와 의미가 달라지므로, 같은 Retriever 실행 안에서만 순위 비교에 사용한다.

## 검증

```bash
python scripts/validate_chunk_contract.py
```

이 명령은 `samples/processed/sample_chunks.jsonl`을 읽어 청크 필수 필드와 `metadata` 객체를 확인한 뒤, `SimpleRetriever`의 결과가 `SearchResult` 다섯 필드와 일치하는지 검증한다.

실제 청크 파일도 같은 방식으로 확인할 수 있다.

```python
from src.parser_chunker import load_chunks_jsonl

chunks = load_chunks_jsonl("/data/processed/chunks_800_120.jsonl", validate=True)
```

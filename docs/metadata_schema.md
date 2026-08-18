# 표준 메타데이터 스키마

## 목적

원본 `/data/original_data/data_list.csv`의 한 행을 RFP 한 문서로 보고, Retriever와 Generation이 같은 이름의 메타데이터 필드를 사용하도록 `/data/processed/metadata.csv`를 생성한다.

- 행 수: 원본 CSV와 동일하게 문서당 1행
- `doc_id`: 원본 CSV의 현재 행 순서 기준 `doc_001` 형식
- 청크: 같은 문서의 모든 청크는 아래 공통 필드를 `metadata`에 복사한다.
- `page`: 현재 본문 단위 추출에서는 페이지 정보를 보존하지 않으므로 빈 값이다. PDF 페이지 단위 청킹을 도입할 때 채운다.

## 표준 필드

| 필드 | 자료형 | 필수 | 예시 | 원본/생성 기준 | 용도 |
| --- | --- | --- | --- | --- | --- |
| `doc_id` | string | 예 | `doc_001` | CSV 행 순서 | 청크와 문서 연결 키 |
| `file_name` | string | 예 | `기관명_사업명.hwp` | `파일명` | 원본 문서 식별 |
| `project_name` | string | 예 | `차세대 포털 구축사업` | `사업명` | 표준 사업명 |
| `title` | string | 예 | `차세대 포털 구축사업` | `사업명` | 기존 Retriever 호환 검색 필드 |
| `agency` | string | 예 | `고려대학교` | `발주 기관` | 기관 필터링 |
| `document_type` | string | 예 | `pdf` | `파일형식` | 표준 문서 형식 |
| `file_type` | string | 예 | `pdf` | `파일형식` | 기존 청크/코드 호환 필드 |
| `page` | string | 아니오 | 빈 값 | 현재 미추출 | 향후 페이지 단위 출처 |
| `source_path` | string | 예 | `/data/original_data/files/기관명_사업명.hwp` | 원본 폴더 + 파일명 | VM 원본 위치 |
| `source_exists` | boolean | 예 | `True` | 파일 존재 여부 | 원본 매칭 검증 |
| `notice_number` | string | 아니오 | `20241001798` | `공고 번호` | 공고 식별 |
| `notice_round` | string | 아니오 | `0` | `공고 차수` | 공고 차수 |
| `budget` | string | 아니오 | `130000000` | `사업 금액` | 예산 질의/필터 |
| `published_at` | string | 아니오 | `2024-10-04 13:51:23` | `공개 일자` | 공고 시점 |
| `bid_start_at` | string | 아니오 | `2024-10-07 09:00:00` | `입찰 참여 시작일` | 입찰 일정 |
| `bid_end_at` | string | 아니오 | `2024-10-15 17:00:00` | `입찰 참여 마감일` | 입찰 일정 |
| `summary` | string | 아니오 | `사업 목적 요약` | `사업 요약` | 검색 결과 보조 정보 |

## 생성과 검증

```bash
python scripts/build_metadata.py
```

명령은 다음을 한 번에 수행한다.

1. `/data/processed/metadata.csv`에 문서당 1행의 표준 메타데이터를 저장한다.
2. 기존 `/data/processed/chunks_800_120.jsonl`의 각 청크 metadata를 표준 필드로 보강한다.

청크 본문과 경계는 바꾸지 않는다. 따라서 Vector DB를 만들기 전에 한 번만 실행하면 되고, 이 작업 자체로 재청킹이나 Vector DB 재생성이 발생하지 않는다.

결과 데이터는 100행이어야 하며, `doc_id` 중복, 필수 필드 누락, 원본 파일 미매칭이 모두 0건인지 확인한다.

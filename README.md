# 입찰메이트 (BidMate) - 사내 RAG 입찰지원 시스템

## 1. 프로젝트 개요

공공·기업 RFP 문서를 기반으로 질문과 관련된 내용을 검색하고 요약하는 RAG 시스템입니다.
가상 RFP로 최소 End-to-End 흐름을 먼저 실행한 뒤 실제 PDF/HWP 데이터와 모델을 연결합니다.

- **진행 기간**: 2026년 7월 10일 ~ 2026년 8월 3일
- **목표**: 주요 요구사항, 발주기관, 예산, 제출 방식 등을 빠르게 검색하고 근거와 함께 답변합니다.

```text
RFP 문서 → 파싱·청킹 → chunks_800_120.jsonl → 임베딩·Chroma 검색 → LLM 답변 → 평가
```

## 2. 회의에서 확정한 구현 방향

- Retrieval 1과 Retrieval 2는 같은 `chunks_800_120.jsonl`을 입력으로 사용하고 처음부터 병렬 개발합니다.
- Retrieval 1은 OpenAI API 임베딩, Retrieval 2는 HuggingFace 로컬 임베딩을 사용합니다.
- 두 Retrieval 모두 Vector DB는 Chroma로 통일합니다.
- 각 파이프라인은 기본 유사도 검색으로 베이스라인을 먼저 완성합니다.
- 이후 MMR, Hybrid Search, Re-ranking을 추가 실험하고 성능을 비교합니다.
- Generation은 두 Retrieval이 반환하는 공통 결과 형식 하나만 사용합니다.
- Generation 모델과 프롬프트는 Generation 담당자가 비교 후 선정합니다.
- FAISS 등 다른 Vector DB 비교는 현재 기본 범위에서 제외합니다.

## 3. 팀원 역할

| 담당자 | 역할 | 주요 업무 | 주요 수정 파일 |
| :--- | :--- | :--- | :--- |
| 유재열 | PM + Data Engineer | 일정·이슈·PR 관리, PDF/HWP 파싱, 메타데이터, 청킹, 공통 입출력 계약 및 통합 | `src/parser_chunker.py`, `scripts/build_chunks.py`, `config/default.yaml` |
| 정서호 | Retrieval 1 | OpenAI 임베딩 모델 선정, OpenAI용 Chroma DB 구축, 기본 검색 및 고도화 실험 | `src/openai_chroma_retriever.py` |
| 이태훈 | Retrieval 2 | HuggingFace 로컬 임베딩 모델 선정, 로컬용 Chroma DB 구축, 기본 검색 및 고도화 실험 | `src/local_chroma_retriever.py` |
| 김효섭 | Generation | LLM 선정, 프롬프트, 근거 기반 답변, 출처와 대화 흐름 구현 | `src/rag_engine.py`, `api_main.py` |

`src/retriever.py`와 `src/retriever_factory.py`는 두 Retrieval이 함께 사용하는 계약 파일입니다. 변경이 필요하면 먼저 팀에 공유합니다.

## 4. 파트별 입력·출력 계약

| 파트 | 입력 | 출력 |
| :--- | :--- | :--- |
| Data | PDF/HWP/TXT, `data_list.csv` | 표준 `metadata.csv`, `chunks_800_120.jsonl` |
| Retrieval 1 | 질문, `chunks_800_120.jsonl` | OpenAI 임베딩·Chroma 검색 결과 |
| Retrieval 2 | 질문, `chunks_800_120.jsonl` | 로컬 임베딩·Chroma 검색 결과 |
| Generation | 질문, 공통 검색 결과 | 최종 답변과 출처 |
| Evaluation | 평가 질문·정답, 실행 결과 | 검색·답변 품질, 속도, 비용 비교 |

### 청크·검색 결과 공통 계약

상세 필드 정의와 검증 방법은 [청크·검색 결과 계약](docs/chunks_schema.md), [표준 메타데이터 스키마](docs/metadata_schema.md)를 기준으로 합니다. Retriever와 Generation은 이 계약의 필드명과 구조를 임의로 바꾸지 않습니다.

`chunks_800_120.jsonl`은 한 줄에 한 청크를 저장하는 JSONL 파일입니다. 모든 청크에는 `chunk_id`, `doc_id`, `text`, `metadata`가 있어야 하며, `metadata`에는 최소한 `title`, `project_name`, `agency`, `file_name`이 비어 있지 않은 문자열로 있어야 합니다.

```json
{
  "chunk_id": "doc_001_chunk_0001",
  "doc_id": "doc_001",
  "text": "문서 내용...",
  "metadata": {
    "title": "사업명",
    "project_name": "사업명",
    "agency": "발주기관",
    "file_name": "원본파일.pdf",
    "document_type": "pdf",
    "source_path": "/data/original_data/files/원본파일.pdf"
  }
}
```

두 Retrieval은 청크의 필드와 전체 `metadata`를 보존하고, 검색 유사도 `score`만 추가한 `SearchResult`를 반환합니다.

```json
{
  "chunk_id": "doc_001_chunk_0001",
  "doc_id": "doc_001",
  "text": "관련 청크 내용...",
  "metadata": {
    "title": "사업명",
    "project_name": "사업명",
    "agency": "발주기관",
    "file_name": "원본파일.pdf"
  },
  "score": 0.87
}
```

### 실제 데이터 및 Vector DB 규칙

- 실제 Retriever 입력 청크는 `/data/processed/chunks_800_120.jsonl`을 사용합니다.
- 원본 문서는 `/data/original_data/files`, 원본 메타데이터는 `/data/original_data/data_list.csv`, 정규화 메타데이터는 `/data/processed/metadata.csv`를 사용합니다.
- 기관명 메타데이터 키는 데이터 전처리 단계에서 정한 agency로 통일합니다. org_name, organization 등 다른 키는 사용하지 않습니다.
- Vector DB 생성 시 청크의 `metadata`를 통째로 보존하고, `chunk_id`, `doc_id`를 추가합니다. 최소 공통 컬럼은 `chunk_id`, `doc_id`, `title`, `project_name`, `agency`, `file_name`, `file_type`, `page`, `source_path`입니다.
- `title`은 사용자에게 보여 주는 문서 출처 제목, `project_name`은 사업명 검색·필터용으로 구분하여 둘 다 보존합니다.
- Vector DB 생성은 별도 빌드 스크립트에서 한 번만 수행합니다.
  - chunks_800_120.jsonl -> 임베딩 -> Chroma Vector DB 저장
- 질문 실행 시 Retriever는 이미 저장된 Vector DB를 열어 검색만 수행합니다.
- OpenAI와 Local Retriever는 별도 Chroma DB를 사용하되, 동일한 청크 및 metadata·검색 결과 규격을 따릅니다. Local DB 빌더는 같은 계약으로 별도 PR에서 추가합니다.
- 공유 저장 경로는 OpenAI `/data/processed/vector_db/openai`, Local `/data/processed/vector_db/local`로 통일합니다. 개인 홈 디렉터리나 프로젝트 상대 경로에는 저장하지 않습니다.
- 청크 내용, 청킹 설정(`chunk_size`, `chunk_overlap`), 임베딩 모델, collection 이름이 바뀌면 기존 컬렉션을 삭제하고 다시 생성합니다. 청크 수만으로 최신 여부를 판단하지 않습니다.

## 5. 병렬 개발 방식

```text
유재열: PDF/HWP → chunks_800_120.jsonl ─┬─→ 정서호: OpenAI 임베딩 + Chroma ─┐
                                └─→ 이태훈: 로컬 임베딩 + Chroma ───┤
가상 RFP + baseline 결과 ───────────→ 김효섭: Generation ────────────┤
                                                                  ↓
                                                    공통 평가 및 최종 조합 선정
```

- Data 담당은 공통 `chunks_800_120.jsonl` 형식을 보장합니다.
- Retrieval 담당자는 별도 Chroma 저장 경로를 사용하므로 DB가 충돌하지 않습니다.
- Generation 담당자는 실제 Retrieval 완성 전에도 baseline 검색 결과로 개발할 수 있습니다.
- 통합 시 `retrieval.active_profile`만 바꿔 같은 Generation과 연결합니다.

## 6. 프로젝트 구조

```text
sprint-ai-mid-project_team3/
├── README.md
├── .gitignore
├── config/
│   └── default.yaml
├── docs/
│   ├── chunks_schema.md               # 청크·SearchResult 공통 계약
│   └── metadata_schema.md             # 표준 metadata.csv 및 청크 metadata 규격
├── data/
│   ├── raw/                           # 실제 원본, Git 업로드 금지
│   └── processed/                     # 실제 metadata·청크·실패 로그, Git 업로드 금지
├── samples/
│   ├── raw/sample_rfp.txt             # 병렬 개발용 가상 RFP
│   └── processed/sample_chunks.jsonl
├── scripts/
│   ├── build_chunks.py                # 실제 PDF/HWP -> 공통 chunks JSONL
│   ├── build_metadata.py              # 원본 CSV -> 표준 metadata.csv·청크 metadata 보강
│   ├── validate_chunk_contract.py     # 청크·SearchResult 공통 계약 검증
│   └── build_api_vectordb.py          # OpenAI 임베딩 -> OpenAI Chroma DB
├── src/
│   ├── parser_chunker.py              # Data 담당
│   ├── retriever.py                   # 공통 결과 형식과 baseline
│   ├── retriever_factory.py           # Retrieval 프로필 선택
│   ├── openai_chroma_retriever.py     # Retrieval 1 담당
│   ├── local_chroma_retriever.py      # Retrieval 2 담당
│   └── rag_engine.py                  # Generation 담당
├── notebook/
│   ├── 00_data_inspection.ipynb        # 원본 메타데이터·파일 매칭 확인
│   ├── 01_parser_chunker_test.ipynb    # PDF/HWP 추출·청킹 확인
│   └── 05_chunk_contract_test.ipynb    # 실제·샘플 청크 계약 검증
├── api_main.py                        # 공통 실행 진입점
├── gcp_main.py                        # 로컬 Retrieval 실행 진입점
└── evaluate.py                        # 프로필별 평가
```

Vector DB는 Git에 올리지 않고 VM 공유 경로에 분리해 저장합니다.

```text
/data/processed/vector_db/
├── openai/
└── local/
```

## 7. 실행 방법

가상환경을 활성화한 뒤 가상 RFP를 청킹합니다.

```bash
python -m scripts.build_chunks
```

실제 데이터를 사용할 때는 먼저 메타데이터를 정규화하고, 청킹 후 공통 계약을 검증합니다.

```bash
python scripts/build_metadata.py
python -m scripts.build_chunks
python scripts/validate_chunk_contract.py
```

`validate_chunk_contract.py`는 필수 최상위 필드, 필수 metadata 필드, baseline `SearchResult` 형식을 확인합니다. 실제 청크 검증은 `notebook/05_chunk_contract_test.ipynb`에서도 확인할 수 있습니다.

현재 동작하는 최소 End-to-End baseline:

```bash
python api_main.py "사업 예산과 수행 기간은 어떻게 돼?" --profile baseline
python evaluate.py --profile baseline
```

각 Retrieval 구현 완료 후:

```bash
python api_main.py "사업 예산과 수행 기간은 어떻게 돼?" --profile openai
python api_main.py "사업 예산과 수행 기간은 어떻게 돼?" --profile local
python evaluate.py --profile openai
python evaluate.py --profile local
```

`sample_rfp.txt`는 실제 기관이나 사업과 관련 없는 가상 문서입니다.

### CLI 데모 출력 예시

`baseline` 프로필로 질문하면 질문 → Profile → 답변 → 출처 순서로 정리되어 출력됩니다.

```
python api_main.py "한영대학교 특성화 맞춤형 교육환경 구축 사업의 예산과 수행 기간은 어떻게 돼?" --profile baseline
```

예상 출력:

```
[질문] 한영대학교 특성화 맞춤형 교육환경 구축 사업의 예산과 수행 기간은 어떻게 돼?
[Profile] baseline

[답변]
- 예산: 130,000,000원(VAT 포함) 범위 내입니다 [2].
- 수행기간: 계약일로부터 3개월이며, 안정화기간 1개월 포함입니다. 다만 기간은 학교 사정과 용역대상자와의 협의에 따라 조정될 수 있습니다 [2].

참고문서: 한영대학_한영대학교 특성화 맞춤형 교육환경 구축 - 트랙운영 학사정보.hwp

[출처 3건]
1. 한국산업단지공단_산단 안전정보시스템 1차 구축 용역.hwp | 기관: 한국산업단지공단 | chunk_id: doc_095_chunk_0003 | score: 0.1592
2. 한영대학_한영대학교 특성화 맞춤형 교육환경 구축 - 트랙운영 학사정보.hwp | 기관: 한영대학 | chunk_id: doc_001_chunk_0001 | score: 0.1585
3. 한국산업단지공단_산단 안전정보시스템 1차 구축 용역.hwp | 기관: 한국산업단지공단 | chunk_id: doc_095_chunk_0007 | score: 0.1558
```

검색 결과가 없을 때는 출처 건수가 0건으로 표시됩니다.

```
python api_main.py "존재하지 않는 사업 문의" --filters '{"agency": "존재하지-않는-기관"}'
```

예상 출력:

```
[질문] 존재하지 않는 사업 문의
[Profile] baseline

[답변]
관련 문서 내용을 찾지 못했습니다. 원본 문서나 검색 조건을 다시 확인해 주세요.

[출처 0건]
```
### 대화형 모드 (후속 질문 처리)

`--interactive` 플래그를 쓰면 하나의 세션에서 여러 질문을 이어서 물어볼 수 있습니다. 후속 질문에서 기관명·사업명을 다시 언급하지 않아도 이전 대화 맥락을 반영해 검색·답변합니다. 빈 줄을 입력하면 종료됩니다.

```bash
python api_main.py --interactive
```

예상 출력:

```
대화형 모드입니다. 종료하려면 빈 줄을 입력하세요.
질문> 한영대학교 특성화 맞춤형 교육환경 구축 사업의 예산은 얼마야?
[답변]
- 예산은 130,000,000원(VAT 포함)입니다. [2]
- 참고 문서: 한영대학_한영대학교 특성화 맞춤형 교육환경 구축 - 트랙운영 학사정보.hwp [2]

질문> 그럼 수행 기간은?
[답변]
- 수행 기간은 계약일로부터 3개월이며, 안정화기간 1개월을 포함합니다. [2]
- 참고 문서: 한영대학_한영대학교 특성화 맞춤형 교육환경 구축 - 트랙운영 학사정보.hwp [2]
```

기관명을 다시 언급하지 않은 후속 질문("그럼 수행 기간은?")도 이전 대화(한영대학교)와 동일한 출처로 답변이 이어지는 것을 확인할 수 있습니다. 대화 보존 턴 수는 `config/default.yaml`의 `generation.history.max_turns`로 조정합니다.

## 8. 실험 순서

1. 가상 RFP로 baseline End-to-End 실행을 확인합니다.
2. Data 담당이 실제 PDF/HWP를 공통 `chunks_800_120.jsonl`로 변환합니다.
3. Retrieval 1과 Retrieval 2가 각자 기본 유사도 검색을 병렬 구현합니다.
4. Generation 담당이 공통 검색 결과로 근거 기반 답변을 생성합니다.
5. 같은 평가 질문으로 OpenAI와 로컬 Retrieval의 검색 품질·속도·비용을 비교합니다.
6. MMR, Hybrid Search, Re-ranking을 추가 실험합니다.
7. 검색 품질, 답변 품질, 응답 속도, 비용을 종합해 최종 조합을 선정합니다.

### 공통 평가 질문

공통 질문은 data/evaluation/questions.jsonl에서 관리합니다. 실제 RFP 근거를 바탕으로 단일 문서, 표 기반, 숫자 계산, 다문서 비교, 후속 질문, 문서 근거 없음 질문을 포함합니다.

OpenAI와 Local Retriever가 완성된 뒤 notebook/04_evaluation.ipynb에서 같은 질문 세트를 각각 실행합니다. 실행 결과는 Google Sheets에 질문 ID, 설정값, 검색 문서, 최종 답변, 응답 시간을 동일하게 기록해 비교합니다.

## 9. 주요 설정

공통 설정은 `config/default.yaml`에서 관리합니다.

- `paths.metadata`, `paths.normalized_metadata`, `paths.raw_documents`, `paths.chunks`: 실제 데이터와 산출물 경로
- `chunking.chunk_size`, `chunking.chunk_overlap`: 청크 크기와 중첩
- `retrieval.active_profile`: `baseline`, `openai`, `local`
- `retrieval.top_k`: 반환할 청크 수
- `retrieval.search_method`: `similarity`, 이후 `mmr`, `hybrid`, `rerank`
- `retrieval.profiles.*.embedding_model`: 각 담당자가 선정한 임베딩 모델
- `retrieval.profiles.*.persist_directory`, `collection_name`: Retriever별 공유 Chroma DB 저장 위치와 컬렉션 이름
- `generation.provider`, `generation.model`: Generation 담당자가 선정한 LLM

모델명과 실험값은 코드에 직접 적지 않고 설정 파일과 실험 기록에 남깁니다.
동일한 값을 코드에 직접 작성하지 않으며, 설정이 없을 때의 fallback 값도 필요한 경우에만 코드와 YAML에서 같은 기준으로 관리합니다.

### 데이터 변경 후 순서

원본 문서나 메타데이터가 추가·수정되면 아래 순서로 처리합니다.

1. `build_metadata.py`로 표준 `metadata.csv`와 청크 metadata를 갱신합니다.
2. `build_chunks.py`로 `paths.chunks`에 설정된 실제 청크 JSONL을 생성합니다.
3. 계약 검증 스크립트 또는 `05_chunk_contract_test.ipynb`로 필수 필드를 확인합니다.
4. Retriever 담당자가 변경된 청크와 설정을 기준으로 각자의 Chroma DB를 다시 생성합니다.

원본 RFP, 실제 청크, Vector DB는 Git이 아닌 VM 공유 경로에서만 관리합니다.

## 10. 협업 규칙

### 브랜치

브랜치 이름은 `이슈번호-역할-작업내용` 형식으로 작성합니다.

```text
3-data-pdf-hwp-parser
16-retrieval1-openai-chroma
17-retrieval2-local-chroma
8-generation-rag-prompt
```

### PR

- PR 제목에는 역할, 이슈 번호, 작업 내용을 적습니다.
- 완료되는 이슈는 `Closes #이슈번호`, 중간 작업은 `Related to #이슈번호`로 연결합니다.
- 화면 작업은 스크린샷, 코드·검색·모델 작업은 실행 결과나 로그 또는 표를 첨부합니다.
- 설정 변경은 변경된 설정값과 적용 결과를 설명합니다.

### 리뷰와 머지

- PR은 최소 2명 이상의 리뷰와 Approve를 받은 뒤 머지합니다.
- 본인이 올린 PR을 본인이 바로 머지하지 않습니다.
- 리뷰어는 코드, 실행 결과, 이슈 완료 기준 충족 여부를 확인합니다.
- 수정 요청을 반영한 뒤 다시 리뷰를 요청합니다.

### 저장 금지 항목

원본 RFP, 대용량 추출 데이터, Vector DB, `.env`, API 키, 모델 파일, 가상환경, GCP·SSH 키는 Git에 올리지 않습니다.

## 11. 작업 관리 및 팀 문서

| 목록 | 링크 |
| :--- | :--- |
| GitHub Issues | [Issues](https://github.com/tjgh8167/sprint-ai-mid-project_team3/issues) |
| GitHub Project | 팀장 Project URL 확정 후 기입 |
| 협업일지 | [협업일지 링크](https://docs.google.com/spreadsheets/d/1LoEBOuxMkzjaf2hdq9GiLNaeNl8UUU9WKPQUzh2PhEM/edit?usp=drive_link) |
| 보고서 작성 | [보고서 링크](https://canva.link/q0494iur016u7uq) |

### 협업일지 Discord 알림

- `.github/workflows/discord-collaboration-log-reminder.yml`이 평일 오후 6시 50분(KST, UTC 09:50)에 Discord로 `협업일지를 작성해주세요`를 전송합니다.
- 예약 실행은 기본 브랜치 `main`에 머지된 워크플로만 동작합니다.
- Repository Settings > Secrets and variables > Actions에 `DISCORD_WEBHOOK_URL` Secret을 등록해야 합니다. Webhook URL은 코드나 README에 넣지 않습니다.
- Actions 탭에서 `Discord Collaboration Log Reminder`를 선택한 뒤 `Run workflow`로 수동 발송 테스트를 할 수 있습니다.

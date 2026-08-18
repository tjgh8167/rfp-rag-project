# BidMate - RFP 기반 RAG 입찰지원 시스템

공공·기업 RFP 문서에서 사용자의 질문과 관련된 근거를 검색하고, 해당 근거를 바탕으로 답변을 생성하는 RAG 시스템입니다. 문서 파싱, 청킹, 임베딩, Chroma Vector DB 검색, LLM 답변 생성, 평가까지 이어지는 End-to-End 흐름을 구현했습니다.

## 프로젝트 요약

| 항목 | 내용 |
| --- | --- |
| 프로젝트 기간 | 2026.07.10 - 2026.08.03 |
| 문제 유형 | Document AI, RAG, Semantic Search |
| 대상 문서 | 공공·기업 RFP PDF/HWP 문서 |
| 주요 기능 | RFP 근거 검색, 메타데이터 필터링, 근거 기반 답변, 출처 반환 |
| Retriever | OpenAI Embedding + Chroma DB |
| Retriever 고도화 | Similarity Search, MMR, Hybrid Search, Re-ranking |
| 최종 검색 검증 | 정답 문서 12/12, 핵심 키워드 30/37, 빈 답변 0건 |

```text
RFP 문서
  -> 문서 파싱 / 텍스트 정제 / 청킹
  -> OpenAI Embedding
  -> Chroma Vector DB
  -> Retriever 검색
  -> LLM 답변 생성
  -> 출처와 함께 답변 반환
```

## 내가 맡은 역할

**정서호 / Retriever**

- `text-embedding-3-small` 기반 OpenAI Embedding과 Chroma DB 검색 흐름을 구현했습니다.
- `src/openai_chroma_retriever.py`에서 검색 방식 선택, metadata filtering, MMR, Hybrid Search, Re-ranking 연결을 담당했습니다.
- `src/hybrid_search.py`에서 BM25 기반 sparse search와 embedding 기반 dense search를 결합했습니다.
- Generation 파트와 연결하기 위해 `SearchResult` 공통 반환 형식을 유지했습니다.
- 질문별 검색 결과를 검증하면서 정답 문서 포함 여부, 핵심 키워드 포함 여부, 빈 답변 발생 여부를 기준으로 성능을 확인했습니다.

## 문제 정의

RFP 문서는 예산, 수행 기간, 제안 조건, 제출 방식, 기능 요구사항처럼 지원 판단에 필요한 정보가 여러 페이지에 흩어져 있습니다. 단순 키워드 검색만으로는 질문의 의도를 반영하기 어렵고, 의미 기반 검색만으로는 예산·기관명·사업명 같은 정확한 키워드를 놓칠 수 있습니다.

이 프로젝트에서는 RFP 문서를 청크 단위로 구조화하고, 질문과 관련된 근거 청크를 찾아 LLM 답변에 연결하는 흐름을 만들었습니다. 검색 결과에는 출처 metadata를 함께 반환해 답변의 근거를 추적할 수 있도록 했습니다.

## 시스템 구성

| 파트 | 역할 |
| --- | --- |
| Data | PDF/HWP 문서 파싱, metadata 정규화, 청킹, 표·이미지 추출 결과 반영 |
| Retriever | OpenAI embedding, Chroma DB 검색, metadata filtering, MMR, Hybrid Search, Re-ranking 실험 |
| Generation | 검색 결과 기반 답변 생성, 출처 표시, 후속 질문 처리 |

## 검색 고도화

### 1. OpenAI Embedding + Chroma DB

RFP 청크를 `text-embedding-3-small`로 임베딩하고 Chroma DB에 저장했습니다. 검색 시에는 이미 생성된 Vector DB를 열어 질문과 유사한 청크를 반환합니다.

구현 파일:

- `src/openai_chroma_retriever.py`
- `scripts/build_api_vectordb.py`

### 2. Metadata Filtering

기관명, 사업명, 문서명처럼 RFP 검색에서 중요한 metadata를 보존하고, 검색 시 필터 조건으로 활용할 수 있도록 구성했습니다. Chroma의 기본 필터가 정확 일치 중심이기 때문에, 검색 후보를 넓게 가져온 뒤 부분 일치 필터링을 적용했습니다.

### 3. MMR Search

유사한 청크가 반복적으로 반환되는 문제를 줄이기 위해 MMR을 실험했습니다. 관련성과 다양성의 균형을 조정하면서 같은 문서의 비슷한 청크만 몰리는 문제를 완화하고자 했습니다.

### 4. Hybrid Search

의미 기반 검색과 키워드 기반 검색을 함께 사용하기 위해 BM25 sparse search와 embedding dense search를 결합했습니다.

구현 파일:

- `src/hybrid_search.py`

핵심 로직:

```text
hybrid_score = dense_score * alpha + sparse_score * (1 - alpha)
```

이 방식은 문장 의미를 반영하면서도 예산, 기관명, 수행 기간처럼 정확한 키워드가 중요한 질문에서 보완 효과를 기대할 수 있었습니다.

### 5. Re-ranking

1차 검색 후보를 더 정밀하게 재정렬하기 위해 Re-ranking 구조를 연결했습니다. 최종 설정에서는 비용과 안정성을 함께 고려해 사용 여부를 설정값으로 제어할 수 있게 했습니다.

구현 파일:

- `src/reranker.py`

## 검색 검증 기준

검색 성능은 단순히 유사도 점수가 높은지를 보지 않고, 실제 답변에 필요한 근거가 검색되는지를 기준으로 확인했습니다.

| 기준 | 설명 |
| --- | --- |
| 정답 문서 포함 여부 | 질문에 답하기 위해 필요한 원본 문서가 검색 결과에 포함되는지 확인 |
| 핵심 키워드 포함 여부 | 예산, 수행 기간, 사업 범위, 제출 방식 등 답변에 필요한 키워드가 포함되는지 확인 |
| 빈 답변 발생 여부 | 검색과 생성 과정에서 답변이 비어 있거나 근거를 찾지 못하는 케이스 확인 |

최종 검증 결과:

| 항목 | 결과 |
| --- | --- |
| 정답 문서 검색 | 12/12 |
| 핵심 키워드 충족 | 30/37 |
| 빈 답변 | 0건 |

## 데이터 처리 개선

RFP 문서에는 표와 이미지에 중요한 정보가 들어 있는 경우가 많았습니다. 본문 텍스트만 청킹하면 표 구조나 이미지 안 텍스트가 검색에서 빠질 수 있어, Data 파트에서 표·이미지 추출 결과를 청크에 병합했습니다.

관련 실험에서는 절 제목을 청크에 함께 반영했을 때 키워드 적중률이 개선되었습니다.

| 청크 세트 | 키워드 적중률 | 정답 문서 평균 |
| --- | ---: | ---: |
| 기존 본문 청크 | 60.3% | 2.69/5 |
| 표·이미지 추출 반영 | 62.0% | 3.23/5 |
| 절 제목 반영 | 74.4% | 3.31/5 |

자세한 내용은 [청킹 실험 기록](docs/experiment_log_chunking.md)에 정리했습니다.

## 프로젝트 구조

```text
rfp-rag-project/
├── api_main.py
├── evaluate.py
├── config/
│   └── default.yaml
├── scripts/
│   ├── build_api_vectordb.py
│   ├── build_chunks.py
│   ├── build_extraction_chunks.py
│   └── validate_chunk_contract.py
├── src/
│   ├── openai_chroma_retriever.py
│   ├── hybrid_search.py
│   ├── reranker.py
│   ├── retriever.py
│   ├── retriever_factory.py
│   └── rag_engine.py
├── docs/
│   ├── chunks_schema.md
│   ├── metadata_schema.md
│   ├── experiment_log.md
│   ├── experiment_log_chunking.md
│   ├── experiment_log_generation.md
│   └── experiment_log_image_extraction.md
└── notebook/
    ├── 02_retrieval_openai_test.ipynb
    ├── 03_generation_test.ipynb
    └── 04_evaluate_test.ipynb
```

## 실행 방법

실제 RFP 원문, 청크 파일, Vector DB, API 키는 저장소에 포함하지 않습니다. 실행하려면 `.env`에 필요한 API 키를 설정하고, `config/default.yaml`의 경로에 맞게 데이터를 준비해야 합니다.

```bash
pip install -r requirements.txt
```

청크 생성:

```bash
python -m scripts.build_chunks
```

OpenAI Chroma DB 생성:

```bash
python -m scripts.build_api_vectordb
```

질문 실행:

```bash
python api_main.py "사업 예산과 수행 기간은 어떻게 돼?" --profile openai
```

평가 실행:

```bash
python evaluate.py --profile openai
```

## 기술 스택

- Python
- OpenAI Embedding
- Chroma DB
- LangChain
- BM25
- Cohere Re-ranker
- PyMuPDF / HWP parsing
- YAML config

## 회고

이 프로젝트에서 가장 중요했던 부분은 RAG 시스템의 품질이 생성 모델만으로 결정되지 않는다는 점이었습니다. 답변이 좋아지려면 먼저 질문에 맞는 근거 문서와 청크가 안정적으로 검색되어야 했고, 이를 위해 임베딩 모델, Vector DB, metadata, 검색 방식, 청크 구조를 함께 조정해야 했습니다.

특히 RFP 문서는 사업명, 기관명, 예산, 수행 기간처럼 정확한 표현이 중요한 정보와 문장 의미가 중요한 요구사항이 함께 존재합니다. 그래서 dense search만 사용하기보다 metadata filtering, MMR, Hybrid Search를 함께 실험하며 검색 품질을 검증했습니다.

이 경험을 통해 문서를 이해하고, 검색 기준을 세우고, 검색 결과를 실제 답변 품질과 연결하는 RAG 엔지니어링 흐름을 경험했습니다.

## 관련 문서

- [기여 정리](docs/contribution.md)
- [청크·검색 결과 계약](docs/chunks_schema.md)
- [Metadata schema](docs/metadata_schema.md)
- [Retrieval 실험 기록](docs/experiment_log.md)
- [청킹 실험 기록](docs/experiment_log_chunking.md)
- [Generation 실험 기록](docs/experiment_log_generation.md)
- [이미지 추출 실험 기록](docs/experiment_log_image_extraction.md)

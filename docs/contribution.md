# Contribution

## 역할

정서호 / Retriever 2

RFP 기반 RAG 시스템에서 사용자의 질문과 관련된 근거 청크를 찾는 Retriever 파트를 담당했습니다. OpenAI Embedding과 Chroma DB를 기반으로 기본 검색 흐름을 만들고, 검색 품질을 높이기 위해 metadata filtering, MMR, Hybrid Search, Re-ranking 구조를 연결했습니다.

## 담당 파일

| 파일 | 담당 내용 |
| --- | --- |
| `src/openai_chroma_retriever.py` | OpenAI Embedding, Chroma DB 연결, similarity/MMR/hybrid/rerank 검색 분기, metadata filtering |
| `src/hybrid_search.py` | BM25 sparse search와 embedding dense search 결합 |
| `src/reranker.py` | 1차 검색 후보 Re-ranking 구조 연결 |
| `scripts/build_api_vectordb.py` | 청크 임베딩 후 Chroma Vector DB 생성 |
| `notebook/02_retrieval_openai_test.ipynb` | 검색 방식별 실험과 검증 |

## 구현한 기능

### OpenAI Embedding + Chroma DB 검색

`text-embedding-3-small`을 사용해 RFP 청크를 임베딩하고, Chroma DB에 저장된 Vector DB를 검색하도록 구성했습니다. 질문 실행 시 매번 청크를 새로 임베딩하지 않고, 미리 생성된 Vector DB를 열어 검색하도록 분리했습니다.

### Metadata Filtering

RFP 검색에서는 기관명, 사업명, 문서명이 중요한 검색 조건이 됩니다. Chroma의 기본 metadata filter는 정확 일치에 가까워 실제 질문에서 쓰기 어렵기 때문에, 검색 후보를 넓게 가져온 뒤 문자열 정규화와 부분 일치 방식으로 필터링했습니다.

### MMR Search

유사한 청크가 반복적으로 검색되는 문제를 줄이기 위해 MMR 검색을 연결했습니다. 관련성만 보는 검색과 달리 다양성을 함께 고려해 top-k 결과가 한 문서 또는 비슷한 문장에 몰리지 않도록 실험했습니다.

### Hybrid Search

RFP 문서에는 의미 기반 검색이 필요한 요구사항 문장과, 정확한 키워드가 중요한 예산·기관명·수행 기간 정보가 함께 존재합니다. 이를 보완하기 위해 BM25 기반 sparse search와 embedding 기반 dense search를 결합했습니다.

```text
hybrid_score = dense_score * alpha + sparse_score * (1 - alpha)
```

### Re-ranking

1차 검색 후보를 더 정밀하게 정렬할 수 있도록 Re-ranking 구조를 추가했습니다. 최종 적용 여부는 설정값으로 제어할 수 있게 구성했습니다.

## 검증 기준

검색 결과는 단순히 점수나 순위만 보지 않고, 실제 답변에 필요한 근거를 포함하는지 기준으로 확인했습니다.

| 기준 | 확인 내용 |
| --- | --- |
| 정답 문서 포함 | 질문에 답하기 위한 원본 문서가 검색 결과에 포함되는지 |
| 핵심 키워드 포함 | 예산, 수행 기간, 사업 범위, 제출 방식 등 답변에 필요한 키워드가 있는지 |
| 빈 답변 여부 | 검색 또는 생성 과정에서 답변이 비는지 |

## 결과

| 항목 | 결과 |
| --- | --- |
| 정답 문서 검색 | 12/12 |
| 핵심 키워드 충족 | 30/37 |
| 빈 답변 | 0건 |

## 배운 점

RAG 시스템에서 답변 품질은 생성 모델 이전에 검색 품질에 크게 좌우됩니다. 질문에 맞는 근거가 검색되지 않으면 LLM이 아무리 좋아도 정확한 답변을 만들기 어렵습니다. 이 프로젝트를 통해 임베딩 모델, Vector DB, metadata, 검색 방식, 청크 구조를 함께 보며 검색 결과를 검증하는 과정이 중요하다는 점을 확인했습니다.

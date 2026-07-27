import os
import re
import unicodedata
from src.retriever import SearchResult
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

class OpenAIChromaRetriever:

    def __init__(self, config: dict):
        load_dotenv()  # 환경변수(.env)에서 OPENAI_API_KEY 로드
        self.config = config

        # retrieval 설정
        retrieval_config = self.config["retrieval"]
        openai_config = retrieval_config["profiles"]["openai"]
        
        embedding_model = openai_config["embedding_model"]
        persist_directory = openai_config["persist_directory"]
        collection_name = openai_config["collection_name"]
        
        # OpenAI 임베딩 생성기 초기화
        self.embeddings = OpenAIEmbeddings(model=embedding_model)

        # top_k 값 설정
        self.default_top_k = retrieval_config["top_k"]

        # 검색 방식 설정
        self.search_method = retrieval_config["search_method"]

        # 부분 일치 설정 (ChromaDB는 기본적으로 정확히 일치하는 필터만 지원하므로, 부분 일치를 위해서는 검색 결과를 더 많이 가져와서 필터링 후 top_k만큼 반환)
        self.fetch_k_base = openai_config["fetch_k"]  # 부분 일치 검색 활용

        # mmr 조절값 설정
        self.lambda_mult = openai_config.get["lambda_mult"]

        # Vector DB 생성
        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=persist_directory
        )
    # 정규화 함수 추가: "--filtter" 텍스트를 정규화하여 필터링 시 일관성을 유지
    def normalize_text(self, text: str) -> str:
        if not text: return ""
        text = unicodedata.normalize('NFC', str(text)) # 유니코드 정규화
        text = text.lower()                            # 소문자 통일
        text = re.sub(r'\s+', ' ', text)               # 공백 문자(스페이스, 탭 등)를 단일 공백으로 변환
        return text.strip()

    def search(self, query: str, top_k: int | None = None, filters: dict | None = None) -> list[SearchResult]:

        # CLI에서 호출할때는 yaml값 그대로, 코드에서 바꾸면 인자값으로 (yaml top_k바꾸면 바뀜)
        search_k = top_k if top_k is not None else self.default_top_k

        # filters 인자가 None이면 빈 딕셔너리로 초기화
        active_filters = filters or {}

        # 부분 일치 설정 (ChromaDB는 기본적으로 정확히 일치하는 필터만 지원하므로, 부분 일치를 위해서는 검색 결과를 더 많이 가져와서 필터링 후 top_k만큼 반환)
        fetch_k = max(search_k * 10, self.fetch_k_base) # fetch_k_base는 yaml에서 설정한 값

        # Search_methood 조건별 작동 로직
        try: 
            if self.search_method == "mmr":
                # MMR 검색 (다양성 고려)
                mmr_k = search_k
                mmr_fetch = fetch_k * 2 if active_filters else fetch_k
                
                docs = self.vectorstore.max_marginal_relevance_search(
                    query=query,
                    k=mmr_k,
                    fetch_k=mmr_fetch,
                    lambda_mult=self.lambda_mult,
                    filter=None
                )
                # Langchain의 MMR은 유사도 점수를 반환하지 않으므로 구조 통일을 위해 0.0으로 처리 (하드 코딩 X)
                docs_and_scores = [(doc, 0.0) for doc in docs]
            else:
                # 유사도 기반 검색 시 필터(--filters{})가 있으면  fetch_k + top_k 사용 / 없다면 top_k만 사용 -> 최적화
                search_limit = fetch_k if active_filters else search_k

                # 유사도 점수
                docs_and_scores = self.vectorstore.similarity_search_with_relevance_scores(
                    query=query,
                    k=search_limit,
                    filter= None
                )

        except Exception as e:       # ChromaDB에서 필터 조건이 잘못되었거나, 검색 중 오류가 발생하면 예외를 발생시켜 호출자에게 알림
            raise

        if not docs_and_scores:      # docs_and_scores가 비어있으면 (즉, 검색 결과가 없으면) 빈 리스트 반환
            return []

        # 부분 일치 필터링 
        results = []
        for doc, score in docs_and_scores:
            is_match = True

            if active_filters:
                for key, value in active_filters.items():
                    if value is None or str(value).strip() == "":    # 숫자 0은 정상 값으로 취급하고, None이거나 빈 문자열일 때만 건너뛰도록 수정
                        continue
                    
                    clean_filter_val = self.normalize_text(str(value)) # 필터 값 정규화
                    clean_metadata_val = self.normalize_text(str(doc.metadata.get(key, ""))) # 문서 메타데이터 값 정규화
                    
                    if clean_filter_val not in clean_metadata_val: # 부분 일치 여부 확인
                        is_match = False
                        break 

            if is_match:
                results.append(
                    SearchResult(
                    chunk_id=doc.metadata.get("chunk_id", ""),
                    doc_id=doc.metadata.get("doc_id", ""),
                    text=doc.page_content,
                    metadata=doc.metadata,
                    score=float(score)
                    )
                )

                if len(results) >= search_k: # 부분 일치 필터링 후 top_k만큼 결과 반환
                    break

        # 필터 미일치 시 안내 동작
        if not results:                                                        # docs_and_scores가 비어있으면 (즉, 검색 결과가 없으면) 안내 메시지 출력
            print(f"안내: '{active_filters}' 조건에 일치하는 문서를 찾을 수 없습니다.")   # 필터 조건에 일치하는 문서가 없음을 안내
            return []
                
        
        return results
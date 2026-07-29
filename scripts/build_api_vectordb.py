import yaml
import json
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

'''
# 생성 규칙
1. 청크 내용, 청킹 설정(`chunk_size`, `chunk_overlap`), 임베딩 모델, collection 이름이 바뀌면 기존 컬렉션을 삭제하고 다시 생성합니다. 청크 수만으로 최신 여부를 판단하지 않습니다.
2. Hybrid 검색 시 BM25와 벡터 검색의 연동을 위해 새로운 청크 생성 시 ChromaDB 재구축을 해야합니다. 
'''

BASE_DIR = Path(__file__).resolve().parent.parent

def build_db():
    load_dotenv() # 환경변수(.env)에서 OPENAI_API_KEY 로직 추가

    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY가 환경변수(.env)에 설정되지 않았습니다.") 
        return

    # 설정 및 데이터 로드
    yaml_path = BASE_DIR / "config/default.yaml" # 절대 경로로 지정
    with open(yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # yaml에 있는 청크 파일 경로 확인 (상대 경로라면 BASE_DIR 기준으로 절대 경로 변환)
    raw_chunk_path = config["paths"]["chunks"]
    chunk_path = Path(raw_chunk_path) if os.path.isabs(raw_chunk_path) else BASE_DIR / raw_chunk_path

    if not chunk_path.exists():
        print(f"청크 파일을 찾을 수 없습니다: {chunk_path}")
        return

    chunks = []
    with open(chunk_path, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    
    total_input_chunks = len(chunks)

    # 청크가 잘 로드 됐는지 출력으로 명시(확인 용도)
    print(f"JSONL 청크 파일 로드 완료: {total_input_chunks}개의 청크 확인")

    # OpenAI yaml 설정 확인
    openai_config = config["retrieval"]["profiles"]["openai"]
    embedding_model = openai_config["embedding_model"]
    persist_directory = openai_config["persist_directory"]
    collection_name = openai_config["collection_name"]

    chunking_config = config["chunking"]
    chunk_size = chunking_config["chunk_size"]
    chunk_overlap = chunking_config["chunk_overlap"]

    # 임베딩 모델 초기화
    embeddings = OpenAIEmbeddings(model=embedding_model)

    # Vector DB 초기화
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory
    )

    # 옵션 변경 감지 로직
    os.makedirs(persist_directory, exist_ok=True)               # persist_directory에 새로운 파일 제작
    state_file = Path(persist_directory) / "param_state.json"   # 옵션 값 저장을 위한 param_state.json 생성
    old_state = {}

    # 이미 param_state를 생성한 기록이 있다면 old_state로 가져와서 읽기
    if state_file.exists():
        with open(state_file, "r", encoding="utf-8") as f:
            old_state = json.load(f)

    # 옵션이 바뀌지 않았다고 우선 설정
    is_option_changed = False

    is_model_changed = old_state.get("embedding_model") and old_state.get("embedding_model") != embedding_model
    is_size_changed = old_state.get("chunk_size") and old_state.get("chunk_size") != chunk_size
    is_overlap_changed = old_state.get("chunk_overlap") and old_state.get("chunk_overlap") != chunk_overlap

    # 3가지 중 하나라도 바뀌었다면 is_option_changed가 True가 됨
    is_option_changed = is_model_changed or is_size_changed or is_overlap_changed

    if is_option_changed:
        print("Vector DB 환경 설정 변경이 감지되었습니다!")
        # 변경된 파라미터만 출력
        if is_model_changed: print(f" - 임베딩 모델: {old_state.get('embedding_model')} -> {embedding_model}")
        if is_size_changed: print(f" - 청크 크기: {old_state.get('chunk_size')} -> {chunk_size}")
        if is_overlap_changed: print(f" - 청크 중첩: {old_state.get('chunk_overlap')} -> {chunk_overlap}")
        print("충돌 방지를 위해 기존 DB를 초기화하고 처음부터 재구축합니다.")

        # 기존 Vector DB 삭제
        vectorstore.delete_collection()

        # 재 구축
        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=persist_directory
        )

    # 로깅: 적재 전 DB 청크 수 확인
    try:
        before_count = vectorstore._collection.count()
        print(f"적재 전 Chroma DB 청크 수: {before_count}개")
    except Exception as exc:
        raise RuntimeError(f"기존 Vector DB 조회에 실패했습니다.") from exc

    # 입력 받은 청크 데이터를 문서로 묶기 (doc_id)
    incoming_docs = {}
    for idx, chunk in enumerate(chunks):
        meta = chunk.get("metadata", {}).copy()
        chunk_id = chunk.get("chunk_id", f"chunk_{idx}")
        doc_id = chunk.get("doc_id", meta.get("doc_id", ""))
        
        meta["chunk_id"] = chunk_id
        meta["doc_id"] = doc_id
        
        if doc_id not in incoming_docs:
            incoming_docs[doc_id] = []
            
        incoming_docs[doc_id].append({
            "id": chunk_id,
            "text": chunk["text"],
            "metadata": meta
        })

    # 통계를 위한 카운터
    total_existing_chunks_processed = 0 # 기존 전체 청크 수
    stat_unchanged = 0                  # 변경 되지 않은 청크 (유지)
    stat_modified = 0                   # 변경 된 청크
    stat_deleted = 0                    # 삭제된 청크
    stat_added = 0                      # 추가된 청크

    # 없어진 정보를 삭제하는 로직 추가
    if not is_option_changed and before_count > 0:              # 완전히 DB를 삭제한 것이 아닌 기존 DB가 한개라도 남아 있다면
        all_db_data = vectorstore.get(include=["metadatas"])    # 기존 DB의 벡터데이터를 긁어옴
        all_db_doc_ids = set()

        for meta in all_db_data.get("metadatas", []):           # 메타데이터에서 doc_id 만 모아서 set으로 저장
            if meta and "doc_id" in meta:
                all_db_doc_ids.add(meta["doc_id"])
        
        incoming_doc_ids = set(incoming_docs.keys())            # 새로 들어온 데이터도 doc_id 만 따로 set으로 저장
        completely_deleted_doc_ids = all_db_doc_ids - incoming_doc_ids  # 원래 있던 doc_id에서 새로들어온 doc_id를 뺌, 그럼 통째로 사라진 파일에 대해 파악 가능
        
        for d_id in completely_deleted_doc_ids:                 # 사라진 문서들의 doc_id를 돌며 삭제 
            # 삭제 전, 해당 문서의 청크 개수만 가볍게 조회 후 삭제
            chunks_to_delete = vectorstore.get(where={"doc_id": d_id}, include=[])
            deleted_chunks_count = len(chunks_to_delete.get("ids", []))
            
            print(f"더 이상 존재하지 않는 문서 삭제 중 (doc_id: {d_id}, 청크 {deleted_chunks_count}개)")
            vectorstore.delete(where={"doc_id": d_id})          # 사라진 문서는 삭제 처리
            
            total_existing_chunks_processed += deleted_chunks_count # 삭제된 청크만큼 청크 status에 추가
            stat_deleted += deleted_chunks_count                    # 삭제된 청크 갯수 추가
        
    # 실제 DB 작업에 쓰일 리스트 모음
    to_upsert_texts = []
    to_upsert_metadatas = []
    to_upsert_ids = []
    to_delete_ids = []

    # 기존 DB 데이터와 새로운 청크 파일 비교 분석
    for doc_id, new_chunks in incoming_docs.items():
        
        try:                                     
            existing = vectorstore.get(where={"doc_id": doc_id}, include=["documents", "metadatas"])
            existing_ids = existing.get("ids", [])
            existing_docs = existing.get("documents", [])
            existing_metadatas = existing.get("metadatas", [])
        except Exception as exc:
            raise RuntimeError(f"기존 Vector DB 조회에 실패했습니다: {doc_id}") from exc

        # 기존 DB에 있던 청크 데이터 수 += (유지, 수정, 삭제 청크 수)
        total_existing_chunks_processed += len(existing_ids)

        # 빠른 비교를 위해 기존 데이터를 딕셔너리로 맵핑
        existing_map = {}
        for eid, edoc, emeta in zip(existing_ids, existing_docs, existing_metadatas):
            existing_map[eid] = {"text": edoc, "metadata": emeta}

        # 새 파일에 존재하는 청크 ID들을 기록 (삭제된 청크 구별용)
        seen_new_ids = set()

        for nc in new_chunks:
            nid = nc["id"]
            ntext = nc["text"]
            nmeta = nc["metadata"]
            seen_new_ids.add(nid)

            if nid in existing_map:
                # 기존에 있던 청크인 경우: 본문이나 메타데이터가 바뀌었는지 검사
                if existing_map[nid]["text"] != ntext or existing_map[nid]["metadata"] != nmeta:
                    stat_modified += 1
                    to_upsert_texts.append(ntext)
                    to_upsert_metadatas.append(nmeta)
                    to_upsert_ids.append(nid)
                else:
                    stat_unchanged += 1
            else:
                # 완전히 처음 보는 새로운 청크인 경우: 추가 대상으로 등록
                stat_added += 1
                to_upsert_texts.append(ntext)
                to_upsert_metadatas.append(nmeta)
                to_upsert_ids.append(nid)

        # 기존 DB에는 있었으나, 새 파일에서는 없어진 청크인 경우: 삭제 대상으로 등록
        for eid in existing_map:
            if eid not in seen_new_ids:
                stat_deleted += 1
                to_delete_ids.append(eid)

    # 정보가 사라진 청크 데이터 삭제
    if to_delete_ids:
        print(f"더 이상 존재하지 않는 구버전 청크 {len(to_delete_ids)}개 삭제 중...")
        vectorstore.delete(ids=to_delete_ids)

    # 추가 및 정보가 변경된 청크 저장
    total_upsert_count = len(to_upsert_ids)
    if total_upsert_count > 0:
        batch_size = openai_config.get("batch_size", 1000)
        print(f"추가(변경)된 청크 {total_upsert_count}개 저장 중")
        
        for i in range(0, total_upsert_count, batch_size):
            batch_texts = to_upsert_texts[i : i + batch_size]
            batch_metadatas = to_upsert_metadatas[i : i + batch_size]
            batch_ids = to_upsert_ids[i : i + batch_size]

            current_batch_num = (i // batch_size) + 1
            total_batches = (total_upsert_count + batch_size - 1) // batch_size
            
            print(f"[{current_batch_num}/{total_batches}] {i} ~ {min(i + batch_size, total_upsert_count)}번째 대상 처리 중")
            vectorstore.add_texts(
                texts=batch_texts,
                metadatas=batch_metadatas,
                ids=batch_ids
            )
    else:
        print("업데이트할 내용이 없습니다. VectorDB 내 모든 데이터가 이미 최신 상태입니다.")

    with open(state_file, "w", encoding="utf-8") as f:      # 업데이트 된 내용을 param_state.json파일에 최신화
        json.dump({                                         # 들어갈 내용
            "embedding_model": embedding_model,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap
        }, f, indent=4)                                     # 들여쓰기 (하드코딩 사유: 들여쓰기라 굳이..?)

    after_count = vectorstore._collection.count()

    print("== VectorDB 최신화 완료 ==")
    print(f" 기존 청크 수: {total_existing_chunks_processed}개")
    print(f" 최신화 후 최종 청크 수: {after_count}개")
    print("-"*40)
    print(f" == 세부 처리 결과 ==")
    print(f" - 유지 : {stat_unchanged}개")
    print(f" - 수정 : {stat_modified}개")
    print(f" - 삭제 : {stat_deleted}개")
    print(f" - 추가 : {stat_added}개")
    print("-"*40)

if __name__ == "__main__":
    build_db()
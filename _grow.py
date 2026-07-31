import os, sqlite3, chromadb
p='/tmp/growtest/chroma.sqlite3'
WATCH=['migrations','acquire_write','maintenance_log','embeddings_queue','max_seq_id','segments','collections']
def snap():
    con=sqlite3.connect(f'file:{p}?mode=ro',uri=True)
    d={t: con.execute(f'select count(*) from {t}').fetchone()[0] for t in WATCH}
    con.close()
    d['파일크기']=os.path.getsize(p)
    return d
base=snap()
print('열기 전  ', base)
for i in (1,5,10):
    while True:
        c=chromadb.PersistentClient(path='/tmp/growtest')
        col=c.get_collection('bidmate_openai'); col.count()
        del col, c
        import chromadb.api.shared_system_client as ssc
        ssc.SharedSystemClient.clear_system_cache()
        break
now=snap()
print('여러 번 연 뒤', now)
print()
for k in base:
    mark = '증가' if now[k]>base[k] else '그대로'
    print(f'  {k:<14} {base[k]:>12,} -> {now[k]:>12,}  {mark}')

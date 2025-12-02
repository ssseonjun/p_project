import chromadb

print("=== Business DB ===")
client_biz = chromadb.PersistentClient(path="chromaDB/chroma_business/finbert")
biz_cols = client_biz.list_collections()
for c in biz_cols:
    print(" -", c.name)

print("\n=== Wiki DB ===")
client_wiki = chromadb.PersistentClient(path="chromaDB/chroma_wiki/finbert")
wiki_cols = client_wiki.list_collections()
for c in wiki_cols:
    print(" -", c.name)
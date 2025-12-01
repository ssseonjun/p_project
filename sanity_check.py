import chromadb
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

client = chromadb.PersistentClient(path="chromaDB/chroma_business")
coll_sbert = client.get_collection("business_sbert")

# 모든 데이터 가져오기 (테스트라면 n_results 적당히 줄여도 됨)
res = coll_sbert.get(include=["embeddings", "metadatas", "documents"])

embs = np.array(res["embeddings"])          # (N, d)
metas = res["metadatas"]                    # list of dict
symbols = [m["symbol"] for m in metas]
companies = [m["company_name"] for m in metas]

# 예시: 2D t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
Z = tsne.fit_transform(embs)

df_vis = pd.DataFrame({
    "x": Z[:,0],
    "y": Z[:,1],
    "symbol": symbols,
    "company": companies,
    # "sector": sectors,  # 있으면 추가
})

plt.figure()
plt.scatter(df_vis["x"], df_vis["y"])  # 색깔은 나중에 sector별로
plt.show()
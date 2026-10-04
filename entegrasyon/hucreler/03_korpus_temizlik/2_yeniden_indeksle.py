import shutil, os
from scripts.indexing.indexer import ensure_index
from scripts.retrieval.retriever import load_retriever

try:
    del retriever
except NameError:
    pass
import gc; gc.collect()

QD = str(settings.qdrant_path)
if os.path.isdir(QD):
    shutil.rmtree(QD)
    print("eski indeks silindi")

print("temizlenmis korpus indeksleniyor (15 dk)")
rapor = ensure_index(settings, force_recreate=True)
print("indeks:", rapor)

retriever = load_retriever(settings)
ornek = retriever.invoke("akut miyokard infarktusunde en erken yukselen belirtec")
print(f"\nretrieval: {len(ornek)} belge")
print(ornek[0].page_content[:300].replace("\n", " "))

import os, sys, glob, json, zipfile, shutil, unicodedata, collections, torch
from pathlib import Path
from google.colab import drive

os.chdir("/content")
drive.mount("/content/drive", force_remount=False)

REPO = "/content/agentic_rag_repo"
if not os.path.isdir(f"{REPO}/agentic_rag"):
    shutil.rmtree(REPO, ignore_errors=True)
    kod = os.system("git clone --branch agentic-rag --depth 1 "
                    "https://github.com/okaybuyukdeveci/Turkish-Medical-LLM-Fine-Tuning-and-Agentic-RAG.git "
                    + REPO)
    assert kod == 0 and os.path.isdir(f"{REPO}/agentic_rag"), f"klonlama basarisiz (kod {kod})"
PROJ = f"{REPO}/agentic_rag"
DATA = f"{PROJ}/data"
sys.path.insert(0, PROJ)
os.chdir(PROJ)

def anahtar(ad):
    d = unicodedata.normalize("NFD", ad)
    return "".join(c for c in d if not unicodedata.combining(c)).casefold()

gruplar = collections.defaultdict(list)
for d in Path(DATA).iterdir():
    if d.is_dir(): gruplar[anahtar(d.name)].append(d)
silinen = 0
for ds in gruplar.values():
    if len(ds) < 2: continue
    ds.sort(key=lambda p: len(p.name))
    for d in ds[1:]:
        silinen += len(list(d.rglob("*.md"))); shutil.rmtree(d)
kalan = glob.glob(f"{DATA}/**/*.md", recursive=True)
print(f"korpus: {len(kalan)} markdown, {len(list(Path(DATA).iterdir()))} ders"
      + (f" (Unicode kopyasi silindi: {silinen})" if silinen else ""))

ZIP_PATH = "/content/drive/MyDrive/tus_data/TUS_Dataset_Final-20261002T154401Z-1-001.zip"
HEDEF = "/content/tus_veri"
if not glob.glob(f"{HEDEF}/**/test_cevap.jsonl", recursive=True):
    with zipfile.ZipFile(ZIP_PATH) as z: z.extractall(HEDEF)
DATA_DIR = str(Path(glob.glob(f"{HEDEF}/**/test_cevap.jsonl", recursive=True)[0]).parent)
test_rows = [{"question": r["soru"], "letter": r["dogru_cevap"],
              "group": (r.get("meta") or {}).get("group", "?")}
             for r in (json.loads(l) for l in open(f"{DATA_DIR}/test_cevap.jsonl", encoding="utf-8"))]
print(f"test: {len(test_rows)} soru")

from scripts.settings import RagSettings
from scripts.indexing.indexer import ensure_index
from scripts.retrieval.retriever import load_retriever

settings = RagSettings(embedding_device="cuda" if torch.cuda.is_available() else "cpu")
settings.validate()
print(f"gomme: {settings.embedding_model} ({settings.embedding_device}) | k={settings.retrieval_k}")

print("\nindeks kuruluyor (5-15 dk)")
rapor = ensure_index(settings, force_recreate=silinen > 0)
print("indeks:", rapor)

retriever = load_retriever(settings)
ornek = retriever.invoke("akut miyokard infarktusunde en erken yukselen belirtec")
print(f"\nretrieval denemesi: {len(ornek)} belge")
print("ilk belge:", ornek[0].page_content[:300].replace("\n", " "))

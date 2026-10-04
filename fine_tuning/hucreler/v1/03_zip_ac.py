import zipfile, glob, os
from pathlib import Path

ZIP_PATH = "/content/drive/MyDrive/tus_data/TUS_Dataset_Final-20261002T154401Z-1-001.zip"
HEDEF    = "/content/tus_veri"

if not glob.glob(f"{HEDEF}/**/egitim.jsonl", recursive=True):
    with zipfile.ZipFile(ZIP_PATH) as z:
        z.extractall(HEDEF)

bulunan = glob.glob(f"{HEDEF}/**/egitim.jsonl", recursive=True)
assert bulunan, f"egitim.jsonl bulunamadi. Cikan icerik: {os.listdir(HEDEF)[:10]}"
DATA_DIR = str(Path(bulunan[0]).parent)
print("DATA_DIR:", DATA_DIR)
for f in ("egitim.jsonl", "test_cevap.jsonl", "test_aciklamali.jsonl"):
    p = Path(DATA_DIR) / f
    assert p.exists(), f"eksik dosya: {f}"
    print(f"  {f}: {sum(1 for _ in open(p, encoding='utf-8'))} kayit")

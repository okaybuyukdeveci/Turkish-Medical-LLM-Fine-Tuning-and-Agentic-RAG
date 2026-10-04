import os, re, sys, glob, zipfile, shutil
from pathlib import Path

ZIP = "/content/drive/MyDrive/tus_data/sozlukler.zip"
SOZ = "/content/sozlukler"
if not os.path.isdir(SOZ):
    with zipfile.ZipFile(ZIP) as z: z.extractall(SOZ)
sys.path.insert(0, SOZ)

# v3_temizlik sozlukleri ../dataset_v3/inceleme/ altinda arar — ayni duzeni kur
Path(f"{SOZ}/dataset_v3/inceleme").mkdir(parents=True, exist_ok=True)
for f in ("deasc.pkl", "sozluk.pkl"):
    h = f"{SOZ}/dataset_v3/inceleme/{f}"
    if not os.path.exists(h): shutil.copy(f"{SOZ}/{f}", h)
Path(f"{SOZ}/scripts").mkdir(exist_ok=True)
shutil.copy(f"{SOZ}/v3_temizlik.py", f"{SOZ}/scripts/v3_temizlik.py")
sys.path.insert(0, f"{SOZ}/scripts")
from v3_temizlik import turkce_yap

YEDEK = "/content/data_ham"
if not os.path.isdir(YEDEK):
    shutil.copytree(DATA, YEDEK)
    print("ham korpus yedeklendi:", YEDEK)

KUNYE = re.compile(
    r"TUSDATA|www\.|https?://|0 ?\(\d{3}\)|444 ?\d{2} ?\d{2}|"
    r"Mah\.|Cad\.|Cadde|Sok\.|Bulvar|Apt\.|Kat ?:|No ?: ?\d|Daire ?:|"
    r"TUS HAZIRLIK|sadece TUS kazandir|BILIMSEL TOPLANTI|Baski-Cilt|"
    r"yayım hakları|5846 sayılı|Grafik - Mizampaj|Sayfa Tasarimi", re.I)
RESIM = re.compile(r"^!\[\]\(images/[^)]*\)\s*$")

def temizle(metin):
    satirlar = metin.split("\n")
    cikti, atilan_kunye, atilan_resim, bos_sikis = [], 0, 0, 0
    onceki_bos = False
    for s in satirlar:
        if KUNYE.search(s):
            atilan_kunye += 1; continue
        if RESIM.match(s):
            atilan_resim += 1; continue
        if not s.strip():
            if onceki_bos: bos_sikis += 1; continue
            onceki_bos = True
        else:
            onceki_bos = False
        cikti.append(s)
    duzeltilmis = turkce_yap("\n".join(cikti), temkinli=True)
    return duzeltilmis, atilan_kunye, atilan_resim, bos_sikis

k_top = r_top = b_top = 0
once_kr = sonra_kr = 0
for f in sorted(glob.glob(f"{YEDEK}/**/*.md", recursive=True)):
    ham = Path(f).read_text(encoding="utf-8", errors="replace")
    yeni, k, r, b = temizle(ham)
    hedef = Path(DATA) / Path(f).relative_to(YEDEK)
    hedef.parent.mkdir(parents=True, exist_ok=True)
    hedef.write_text(yeni, encoding="utf-8")
    k_top += k; r_top += r; b_top += b
    once_kr += len(ham); sonra_kr += len(yeni)
    print(f"  {Path(f).name[:40]:<42} {len(ham):>8} -> {len(yeni):>8}")

print(f"\nkunye satiri atildi : {k_top:,}")
print(f"resim yer tutucu    : {r_top:,}")
print(f"fazla bos satir     : {b_top:,}")
print(f"karakter            : {once_kr:,} -> {sonra_kr:,}  (%{100*(once_kr-sonra_kr)/once_kr:.1f} kucuildu)")

ornek = Path(sorted(glob.glob(f"{DATA}/**/*.md", recursive=True))[0]).read_text(encoding="utf-8")
print("\ntemizlenmis dosyanin ilk 25 satiri:")
print("\n".join(ornek.split("\n")[:25]))

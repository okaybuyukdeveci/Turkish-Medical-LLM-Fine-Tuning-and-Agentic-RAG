import json, re
from google.colab import files

try:
    base_outs, ft_outs, test_acc
except NameError:
    raise SystemExit("Oturum dusmus — 01-11 arasi hucreleri yeniden calistir, sonra buraya don.")

def sik_metni(soru, harf):
    m = re.search(rf"(?m)^{harf}\) (.+)$", soru)
    return m.group(1).strip() if m else ""

kayitlar = []
for r, bo, fo in zip(test_acc, base_outs, ft_outs):
    bp, fp = parse_loose(bo), parse_loose(fo)
    g = r["letter"]
    kayitlar.append({
        "soru": r["question"],
        "kaynak": r.get("group", "?"),
        "dogru_harf": g,
        "dogru_metin": sik_metni(r["question"], g),
        "taban_harf": bp,
        "taban_metin": sik_metni(r["question"], bp) if bp else "",
        "taban_cikti": bo.strip()[:400],
        "ft_harf": fp,
        "ft_metin": sik_metni(r["question"], fp) if fp else "",
        "ft_cikti": fo.strip()[:400],
        "taban_dogru": bp == g,
        "ft_dogru": fp == g,
        "durum": ("ikisi_de_dogru" if bp == g and fp == g else
                  "ikisi_de_yanlis" if bp != g and fp != g else
                  "ft_kazandi" if fp == g else "ft_kaybetti"),
    })

from collections import Counter
print("durum dagilimi:", dict(Counter(k["durum"] for k in kayitlar)))
print("ft yanlis toplam:", sum(not k["ft_dogru"] for k in kayitlar))

with open("/content/hata_analizi.json", "w", encoding="utf-8") as f:
    json.dump(kayitlar, f, ensure_ascii=False, indent=1)
files.download("/content/hata_analizi.json")

"""sonuclar/*.json dosyalarindan OZET.md tablosunu uretir."""
import json
from pathlib import Path
from collections import defaultdict

KOK = Path(__file__).resolve().parent
dosyalar = sorted((KOK / "sonuclar").glob("*.json"))
if not dosyalar:
    (KOK / "sonuclar" / "OZET.md").write_text("# Sonuçlar\n\nHenüz koşu yok.\n", encoding="utf-8")
    raise SystemExit("sonuç yok")

kosular = []
for f in dosyalar:
    d = json.loads(f.read_text(encoding="utf-8"))
    kosular.append({**d, "dosya": f.name, "tarih": f.stem.split("__")[-1]})

veriler = sorted({k["veri"] for k in kosular})
konfigler = sorted({k["konfig"] for k in kosular})
# ayni veri+konfig icin en yeni kosu
son = {}
for k in kosular:
    son[(k["veri"], k["konfig"])] = max(son.get((k["veri"], k["konfig"]), k), k,
                                        key=lambda x: x["tarih"])

s = ["# Benchmark sonuçları", "",
     "Otomatik üretildi — `python benchmark/ozet.py`. Elle düzenleme.", "",
     "## Doğruluk", "",
     "| Veri seti | Soru | " + " | ".join(konfigler) + " |",
     "|---|---:| " + " | ".join("---:" for _ in konfigler) + " |"]
for v in veriler:
    n = next((k["soru_sayisi"] for k in kosular if k["veri"] == v), "?")
    hucre = []
    for c in konfigler:
        k = son.get((v, c))
        hucre.append(f"{k['dogruluk']:.1%}" if k else "—")
    s.append(f"| `{v}` | {n} | " + " | ".join(hucre) + " |")

s += ["", "## Koşu kaydı", "",
      "| Tarih | Veri | Konfig | Doğruluk | Cevapsız | Süre | Cihaz |",
      "|---|---|---|---:|---:|---:|---|"]
for k in sorted(kosular, key=lambda x: (x["tarih"], x["veri"]), reverse=True):
    s.append(f"| {k['tarih'][:4]}-{k['tarih'][4:6]}-{k['tarih'][6:]} | `{k['veri']}` | "
             f"`{k['konfig']}` | {k['dogruluk']:.1%} | {k['cevapsiz']} | "
             f"{k['sure_sn']/60:.0f} dk | {k['cihaz']} |")

s += ["", "## Okurken dikkat", "",
      "**Yıllar arası yüzde karşılaştırması ölçüm değildir.** 2025'te %85, 2026'da %82",
      "çıkması modelin kötüleştiğini göstermez — soru zorluğu farklı olabilir. Her veri",
      "setini kendi içinde okuyun; aynı set üzerinde iki konfigürasyonu karşılaştırmak",
      "anlamlıdır, iki farklı seti karşılaştırmak değil.", "",
      "Sonuç dosyaları soru bazında tahminleri tutuyor; aynı set üzerindeki iki koşu",
      "paired McNemar ile karşılaştırılabilir.", ""]
(KOK / "sonuclar" / "OZET.md").write_text("\n".join(s), encoding="utf-8")
print(f"{len(kosular)} koşu → sonuclar/OZET.md")

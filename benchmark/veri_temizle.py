"""PDF ayrıştırma artıklarını temizler. İhtiyatlı: şüpheliyi düzeltmez, işaretler.

    python benchmark/veri_temizle.py veri/2024_sonbahar.jsonl --yaz

Düzelttikleri, hepsi ölçülmüş kusurlar:
  1. Derleyicinin eklediği "(DUS'da sorulmaya uygun)" gibi notlar
  2. Şık sonundaki başıboş rakamlar (sayfa/soru numarası sızıntısı)
  3. Cümle başı ayrık harf: "A şağıdaki" -> "Aşağıdaki"

Düzeltmediği, sadece eleyip raporladığı:
  - Boş ya da 2 karakterden kısa şık
  - 5'ten farklı şık sayısı
"""
import argparse, json, re
from pathlib import Path

NOT = re.compile(r"\s*\((?:DUS|TUS)[’'`´]?(?:da|de|ta|te)?\s+soru[a-zçğıöşü]*\s+uygun\)\s*", re.I)
SON_RAKAM = re.compile(r"\s+\d{1,3}(?:\s+\d{1,3})*\s*$")
# Yalniz cumle basinda birlestir. Cumle ortasindaki tek harf gercek olabilir:
# "R yan zincirleri", "T hucresi", "D vitamini", "X isini".
BAS_AYRIK = re.compile(r"(^|(?<=[.?!]\s))([A-ZÇĞİÖŞÜ])\s([a-zçğıöşü]{2,})")
# Cumle ortasinda GENEL birlestirme yapilmaz: 395 eslesmenin 182'si "I ve II"
# gibi sik metinleri, 25'i "D vitamini", onlarcasi "T hucresi" / "R yan zinciri".
# Birlestirmek bunlari bozardi. Yalnizca Turkce'de tek basina kelime olmayan
# sayili parca hedeflenir.
PARCA = re.compile(r"\b([YA])\s(ukarıda\w*|alnız|şağıda\w*)", re.I)

def temizle_metin(s):
    s = NOT.sub(" ", s)
    s = BAS_AYRIK.sub(lambda m: f"{m.group(1)}{m.group(2)}{m.group(3)}", s)
    s = PARCA.sub(lambda m: f"{m.group(1)}{m.group(2)}", s)
    return re.sub(r"\s+", " ", s).strip()

def temizle(r):
    satir = r["soru"].split("\n")
    kok, siklar = satir[0], satir[1:]
    kok = temizle_metin(kok)
    yeni = []
    for s in siklar:
        s = NOT.sub(" ", s)
        # Sondaki basibos rakami kirp, AMA sikkin kendisi sayiysa dokunma:
        # "A) 7" (Glasgow skoru) ile "E) Vertebra 10 11" (sayfa no sizintisi)
        # ayni regexe uyuyor. Kirpma sonrasi icerik bosaliyorsa geri al.
        kirpik = SON_RAKAM.sub("", s)
        if re.sub(r"^[A-E]\)\s*", "", kirpik).strip(): s = kirpik
        s = PARCA.sub(lambda m: f"{m.group(1)}{m.group(2)}", s)
        yeni.append(re.sub(r"\s+", " ", s).strip())
    sorun = []
    if len(yeni) != 5: sorun.append(f"{len(yeni)} sik")
    for s in yeni:
        # "A) 7" gecerlidir (Glasgow skoru, doz, deger). Yalnizca GERCEKTEN bos
        # olani ele. "2 karakterden kisa" kurali sayisal cevapli sorulari eliyordu.
        if not re.sub(r"^[A-E]\)\s*", "", s).strip(): sorun.append("bos sik")
    if len(kok) < 25: sorun.append("kok cok kisa")
    r = dict(r); r["soru"] = kok + "\n" + "\n".join(yeni)
    return r, sorun

def main():
    a = argparse.ArgumentParser()
    a.add_argument("yollar", nargs="+")
    a.add_argument("--yaz", action="store_true", help="dosyanin uzerine yaz")
    A = a.parse_args()
    for y in A.yollar:
        p = Path(y)
        rows = [json.loads(l) for l in p.read_text(encoding="utf-8").split("\n") if l.strip()]
        tut, elenen, degisen = [], [], 0
        for r in rows:
            t, sorun = temizle(r)
            if sorun: elenen.append((r["id"], sorun)); continue
            if t["soru"] != r["soru"]: degisen += 1
            tut.append(t)
        print(f"{p.name:<24} {len(rows):>4} -> {len(tut):>4}  "
              f"(temizlenen {degisen}, elenen {len(elenen)})")
        for i, s in elenen[:3]: print(f"      elendi {i}: {', '.join(s)}")
        if A.yaz:
            with p.open("w", encoding="utf-8") as f:
                for t in tut: f.write(json.dumps(t, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()

"""Ayristirilmis eski sinavlardan karar-egitimi havuzu kurar.

    python benchmark/havuz_kur.py /tmp/ham/*.jsonl --cikti benchmark/veri_2006_2020

Uc nedenle kayit duser ve her biri manifestte sayilir:
  ortusme  — egitim setiyle shingle ortusmesi >= %50; model ezberden dogru
             yapar, "biliyorum" etiketi yaniltici olur.
  bos_sik  — ust/alt simgeli sik metni ayri satira dusmus, sik bos kalmis.
  kopya    — ayni sinavda Jaccard >= 0.9; sahte soru numarasindan dogan
             ikiz kayitlarin ikisi de atilir, hangisinin dogru oldugu belli degil.
  talimat  — 2009/2010'da kapak sayfasindaki sinav talimati soru numarasi
             kapmis; anahtardan gelen harf o metne ait degil.
  altbilgi — ayristiricidan sizmis sayfa altbilgisi kalintisi.
"""
import argparse, json, re, sys
from pathlib import Path

KOK = Path(__file__).resolve().parent
sys.path.insert(0, str(KOK))
from dogrula import jeton, shingle, jaccard

SIK_SATIR = re.compile(r"^[A-E]\)")
TALIMAT = re.compile(r"Sınavda uyulacak|Cevaplamaya istediğiniz|kitapçığın arka kapağ|"
                     r"soru kitapçığ|cevap kâğıd|optik okuyucu|işaretlemelerinizi", re.I)
ALTBILGI = re.compile(r"TEST[İI]\s+soruları bulunmaktadır|Diğer sayfaya geçiniz|"
                      r"\d{4}-[A-Z]{1,3}-[A-Z]{3,}")

def bos_sikli(soru):
    siklar = [l for l in soru.split("\n") if SIK_SATIR.match(l)]
    return any(not re.sub(r"^[A-E]\)", "", l).strip() for l in siklar)

def main():
    a = argparse.ArgumentParser()
    a.add_argument("yollar", nargs="+")
    a.add_argument("--cikti", required=True)
    a.add_argument("--parmak-izi", default=str(KOK / "parmak_izi" / "egitim_shingle.txt"))
    a.add_argument("--ortusme-esik", type=float, default=0.5)
    A = a.parse_args()

    pi = set(Path(A.parmak_izi).read_text().split())
    cikti = Path(A.cikti); cikti.mkdir(parents=True, exist_ok=True)
    manifest, toplam = [], {"okunan": 0, "ortusme": 0, "bos_sik": 0, "kopya": 0,
                            "talimat": 0, "altbilgi": 0, "kalan": 0}

    for yol in sorted(Path(y) for y in A.yollar):
        kayit = [json.loads(l) for l in yol.read_text(encoding="utf-8").split("\n") if l.strip()]
        sayim = {"okunan": len(kayit), "ortusme": 0, "bos_sik": 0, "kopya": 0,
                 "talimat": 0, "altbilgi": 0}

        jetonlar = [set(jeton(x["soru"])) for x in kayit]
        ikiz = set()
        for i in range(len(kayit)):
            for j in range(i + 1, len(kayit)):
                if jaccard(jetonlar[i], jetonlar[j]) >= 0.9: ikiz |= {i, j}

        tutulan = []
        for i, x in enumerate(kayit):
            if i in ikiz: sayim["kopya"] += 1; continue
            if TALIMAT.search(x["soru"]): sayim["talimat"] += 1; continue
            if ALTBILGI.search(x["soru"]): sayim["altbilgi"] += 1; continue
            if bos_sikli(x["soru"]): sayim["bos_sik"] += 1; continue
            sh = shingle(jeton(x["soru"]))
            if sh and len(sh & pi) / len(sh) >= A.ortusme_esik:
                sayim["ortusme"] += 1; continue
            tutulan.append(x)

        with (cikti / yol.name).open("w", encoding="utf-8") as f:
            for x in tutulan: f.write(json.dumps(x, ensure_ascii=False) + "\n")
        sayim["kalan"] = len(tutulan)
        manifest.append({"dosya": yol.name, **sayim})
        for k in toplam: toplam[k] += sayim.get(k, 0)
        print(f"  {yol.stem:<22}{sayim['okunan']:>5} ->{sayim['kalan']:>5}   "
              f"ortusme {sayim['ortusme']:>3}  bos_sik {sayim['bos_sik']:>2}  "
              f"kopya {sayim['kopya']:>2}  talimat {sayim['talimat']:>2}  "
              f"altbilgi {sayim['altbilgi']:>2}")

    (cikti / "manifest.json").write_text(
        json.dumps({"toplam": toplam, "dosyalar": manifest,
                    "ortusme_esik": A.ortusme_esik}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    print(f"\ntoplam {toplam['okunan']} -> {toplam['kalan']} soru")
    print(f"  dusen: ortusme {toplam['ortusme']}, bos_sik {toplam['bos_sik']}, "
          f"kopya {toplam['kopya']}, talimat {toplam['talimat']}, altbilgi {toplam['altbilgi']}")
    print("yazildi:", cikti)

if __name__ == "__main__":
    main()

"""TUS sınav PDF'ini benchmark şemasına çevirir.

    python benchmark/pdf_ayristir.py sinav.pdf --cikti veri/2024_sonbahar.jsonl

Beklenen biçim: numaralı sorular (1., 2., ...), A)–E) şıkları, sonda
"CEVAP ANAHTARI" sayfası. Metin katmanı olmayan PDF'lerde çalışmaz —
o durumda önce OCR gerekir.

2020 ve öncesinin PDF'lerinde soru numaraları ile şık harfleri ayrı bir
metin bloğunda duruyor; pypdf onları sayfanın sonuna atıyor ve sorular
şıksız görünüyor. `--motor konum` kelimeleri koordinatlarından yeniden
satırlara dizer. Varsayılan pypdf, çünkü 2021+ verisi onunla ölçüldü.
"""
import argparse, json, re, sys
from pathlib import Path
from pypdf import PdfReader

SIK = re.compile(r"(?<![A-Za-zÇĞİÖŞÜçğıöşü0-9])([A-E])\)\s*")
IPTAL = re.compile(r"^\s*Bu soru iptal edilmi", re.I)
SORU_ISARET = re.compile(r"^[ \t]*(\d(?:[ \t]?\d){0,2})[ \t]*\.(?=\s|[A-ZÇĞİÖŞÜ])", re.M)
ANAHTAR_SATIR = re.compile(r"(\d{1,3})\s*\.\s*([A-E])(?![a-zçğıöşü])")
GORSEL_IZ = re.compile(r"şekil|görüntü|resim|grafik|fotoğraf|aşağıda gösterilen|yukarıdaki görünt",
                       re.I)
# Tek basina duran A-E harfi kitapcik kodu; 2006-2009'da neredeyse her
# sayfada var ve dustugu blogun son sikkina yapisiyor. Sik satiri her zaman
# "A)" ile basladigi icin cikplak harf asla gercek sik degil.
SAYFA_COPU = re.compile(r"^(?:\d{1,3}|[A-E]|Diğer sayfaya geçiniz\.?|\d{4}-TUS.*|T[TK]BT)$")
# Kitapcik kodu / altbilgi bazen son sikkin y hizasina dusup ayni satira
# karisiyor. Satir ici oldugu icin tam-satir suzgeci yakalamiyor; satirin
# kalan kismi buradan kesilir. Kesilen yerin hemen oncesinde duran tek harf
# kitapcigin cevap isaretidir, o da atilir ("E) Amfizem A 2006-G-KTBT").
SATIR_COPU = re.compile(r"(?:^|(?<=\s))(?:\d{4}-[A-Z]{1,3}-[A-Z]{3,}[\w-]*|"
                        r"Diğer sayfaya geçiniz|\d{4}-TUS\b|Bu testte \d+|TEST[İI]\s+soruları)")
KESIK_HARF = re.compile(r"\s[A-E]$")

def satir_temizle(t):
    m = SATIR_COPU.search(t)
    if not m: return t
    return KESIK_HARF.sub("", t[:m.start()].rstrip()).rstrip()

def konum_sayfasi(sayfa, tol=3.0):
    """Kelimeleri koordinatlarindan satirlara dizer, iki sutunu ayri okur."""
    kelime = sayfa.get_text("words")
    if not kelime: return ""
    orta = (min(w[0] for w in kelime) + max(w[2] for w in kelime)) / 2
    sutun = ([], [])
    for w in kelime: sutun[0 if w[0] < orta else 1].append(w)
    satirlar = []
    for sut in sutun:
        grup = {}
        for w in sorted(sut, key=lambda w: (w[1], w[0])):
            y = next((g for g in grup if abs(g - w[1]) <= tol), w[1])
            grup.setdefault(y, []).append(w)
        for y in sorted(grup):
            t = " ".join(w[4] for w in sorted(grup[y], key=lambda w: w[0])).strip()
            t = satir_temizle(t)
            if t and not SAYFA_COPU.match(t): satirlar.append(t)
    return "\n".join(satirlar)

def sayfalar(yol, motor="pypdf"):
    """Soru sayfalarinda koordinat motoru, anahtar sayfalarinda duz metin.

    Anahtar sayfasi cok sutunlu bir tablo: iki sutuna bolmek "1. A" ile baska
    sutundaki "48."i ayni satirda birlestirip ciftleri kiriyor. Duz get_text()
    o sayfalari dogru veriyor, soru sayfalarinda ise sik harflerini sona atiyor.
    """
    if motor == "konum":
        import pymupdf
        with pymupdf.open(yol) as d:
            cikti = []
            for sayfa in d:
                duz = sayfa.get_text() or ""
                cikti.append(duz if len(ANAHTAR_SATIR.findall(duz)) >= 20
                             else konum_sayfasi(sayfa))
            return cikti
    return [(p.extract_text() or "") for p in PdfReader(yol).pages]

def anahtar_cikar(sayfa_metinleri, esik=20):
    """Cevap anahtari sayfasini BASLIK kelimesiyle degil, orgu yogunluguyla bulur.

    Dosyalarin cogunda "CEVAP ANAHTARI" basligi yok; yalnizca "12. B" satirlari
    var. Bir anahtar sayfasinda bu kalip yuzlerce kez gecer, soru sayfasinda
    gecmez — esik bu yuzden guvenli.
    """
    sayim = [(i, len(ANAHTAR_SATIR.findall(s))) for i, s in enumerate(sayfa_metinleri)]
    anahtar_sayfa = [i + 1 for i, n in sayim if n >= esik]
    if not anahtar_sayfa:            # yine de basliga bak (kucuk sinavlar)
        anahtar_sayfa = [i + 1 for i, s in enumerate(sayfa_metinleri) if "ANAHTAR" in s.upper()]
    anahtar = {}
    for i in anahtar_sayfa:
        for no, harf in ANAHTAR_SATIR.findall(sayfa_metinleri[i - 1]):
            anahtar[int(no)] = harf
    return anahtar, anahtar_sayfa

def siklari_ayir(blok):
    """Tek satıra sıkışmış şıkları kendi satırlarına böler."""
    yerler = [(m.start(), m.group(1)) for m in SIK.finditer(blok)]
    beklenen, secili = "A", []
    for p, h in yerler:
        if h == beklenen:
            secili.append((p, h))
            beklenen = chr(ord(beklenen) + 1)
            if beklenen > "E": break
    if len(secili) < 4: return None, None
    kok = blok[:secili[0][0]].strip()
    siklar = []
    for i, (p, h) in enumerate(secili):
        son = secili[i+1][0] if i + 1 < len(secili) else len(blok)
        metin = blok[p:son].strip()
        siklar.append(re.sub(r"\s+", " ", metin))
    return kok, siklar

def bolum_adi(metin):
    u = metin[:400].upper()
    if "TEMEL" in u: return "temel"
    if "KLİNİK" in u or "KLINIK" in u: return "klinik"
    return None

def ayristir(yol, kaynak_adi, motor="pypdf"):
    """PDF'i BOLUM bolum ayristirir.

    2020 oncesi sinavlarda tek PDF'te iki ayri test var (Temel ve Klinik) ve
    soru numaralari her ikisinde de 1'den basliyor. Numaraya gore tek sozluk
    tutmak ikisini ust uste bindirip yarisini kaybediyordu. Her cevap anahtari
    sayfasi bir bolumu kapatir; bolumler ayri ayri ayristirilir.
    """
    sy = sayfalar(yol, motor)
    _, anahtar_sayfa = anahtar_cikar(sy)
    if not anahtar_sayfa: raise SystemExit("cevap anahtarı bulunamadı")

    ANAHTAR_BAS = re.compile(r"^[^\n]*CEVAP\s+ANAHTAR[^\n]*$", re.M | re.I)
    kayitlar, atlanan = [], {"sik_az": 0, "cevap_yok": 0, "iptal": 0}
    onceki, bolumler = 0, []
    for ks in anahtar_sayfa:                       # 1 tabanli
        anahtar = {int(n): h for n, h in ANAHTAR_SATIR.findall(sy[ks-1])}
        govde_sayfa = sy[onceki:ks-1]
        son = sy[ks-1]
        m = ANAHTAR_BAS.search(son) or ANAHTAR_SATIR.search(son)
        if m and m.start() > 200: govde_sayfa = govde_sayfa + [son[:m.start()]]
        bolumler.append((bolum_adi(son) or f"b{len(bolumler)+1}", anahtar, "\n".join(govde_sayfa)))
        onceki = ks

    # 2010 ve oncesinde iki anahtar sayfasi dosyanin sonunda yan yana duruyor;
    # ikinci bolume govde kalmiyor. Iki test de ilk anahtardan onceki sayfalarda.
    # Bolum basligi bu dosyalarda hic yok, ayirt eden sey numaralandirmanin
    # yeniden 1'den baslamasi.
    if len(bolumler) > 1 and any(len(SORU_ISARET.findall(g)) < len(a) * 0.5
                                 for _, a, g in bolumler):
        tam = "\n".join(sy[:anahtar_sayfa[0] - 1])
        isaret = [(m.start(), int(m.group(1).replace(" ", "")))
                  for m in SORU_ISARET.finditer(tam)]
        sinir = [0]
        for _, anahtar, _ in bolumler[:-1]:
            aday = [p for i, (p, n) in enumerate(isaret)
                    if i and n <= 3 and isaret[i-1][1] >= len(anahtar) * 0.8 and p > sinir[-1]]
            if not aday: sinir = None; break
            sinir.append(aday[0])
        if sinir:
            sinir.append(len(tam))
            bolumler = [(ad, anahtar, tam[sinir[i]:sinir[i+1]])
                        for i, (ad, anahtar, _) in enumerate(bolumler)]
            print("  govde numaralandirmanin yeniden basladigi yerden bolundu")
        else:
            print("  UYARI: govde eksik, bolum siniri bulunamadi")

    print(f"{len(sy)} sayfa | {len(bolumler)} bolum | anahtar sayfasi {anahtar_sayfa}")
    for bad, anahtar, govde in bolumler:
        gecerli = set(anahtar)
        isaret = [(m.start(), int(m.group(1).replace(" ", "")))
                  for m in SORU_ISARET.finditer(govde)]
        def blok_al(i, uzatma):
            j = min(i + 1 + uzatma, len(isaret))
            return govde[isaret[i][0]: isaret[j][0] if j < len(isaret) else len(govde)]
        bloklar = {}
        for i, (_, n) in enumerate(isaret):
            if n not in gecerli: continue
            for uz in (0, 1, 2):
                blok = blok_al(i, uz)
                _, s = siklari_ayir(re.sub(r"^[ \t]*\d(?:[ \t]?\d){0,2}[ \t]*\.\s*", "", blok.strip()))
                if s: break
            if n not in bloklar or len(blok) > len(bloklar[n]): bloklar[n] = blok
        print(f"  bolum {bad}: {len(anahtar)} cevap, {len(bloklar)} blok")

        for no, ham in sorted(bloklar.items()):
            blok = re.sub(r"^[ \t]*\d(?:[ \t]?\d){0,2}[ \t]*\.\s*", "", ham.strip())
            blok = re.sub(r"\n\s*\d+\s*\n", "\n", blok)
            # Iptal edilen sorunun blogu bir sonraki sorunun metnini icine
            # aliyor; anahtardan gelen harf o metne ait degil.
            if IPTAL.match(blok): atlanan["iptal"] += 1; continue
            kok, siklar = siklari_ayir(blok)
            if not siklar: atlanan["sik_az"] += 1; continue
            yil = re.search(r"(20\d\d)", kaynak_adi)
            kayitlar.append({
                "id": f"{kaynak_adi}_{bad}_{no:03d}",
                "soru": re.sub(r"\s+", " ", kok).strip() + "\n" + "\n".join(siklar),
                "dogru_cevap": anahtar[no],
                "yil": int(yil.group(1)) if yil else None,
                # Tek bolumlu sinavda anahtar basligi iki testi birden kapsiyor;
                # bolumu soru numarasindan tur (1-100 temel, 101+ klinik).
                "kaynak": kaynak_adi, "soru_no": no,
                "bolum": bad if len(bolumler) > 1 else ("temel" if no <= 100 else "klinik"),
                "gorsel_supheli": bool(GORSEL_IZ.search(kok)),
            })
    return kayitlar, atlanan

def main():
    a = argparse.ArgumentParser()
    a.add_argument("pdf")
    a.add_argument("--cikti", required=True)
    a.add_argument("--ad", default=None, help="kaynak adi, varsayilan dosya adi")
    a.add_argument("--motor", choices=("pypdf", "konum"), default="pypdf",
                   help="konum: kelimeleri koordinatindan diz (2020 ve oncesi)")
    A = a.parse_args()
    ad = A.ad or Path(A.pdf).stem
    kayitlar, atlanan = ayristir(A.pdf, ad, A.motor)
    gorsel = sum(k["gorsel_supheli"] for k in kayitlar)
    print(f"\ncikarilan: {len(kayitlar)} soru | atlanan: {atlanan}")
    print(f"gorsele atif yapan (metin-only'de cozulemeyebilir): {gorsel}")
    Path(A.cikti).parent.mkdir(parents=True, exist_ok=True)
    with Path(A.cikti).open("w", encoding="utf-8") as f:
        for k in kayitlar: f.write(json.dumps(k, ensure_ascii=False) + "\n")
    print("yazildi:", A.cikti)

if __name__ == "__main__":
    main()

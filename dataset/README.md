# Veri seti

TUS hazırlık kitaplarının PDF'lerinden çıkarılmış, çoktan seçmeli soru +
açıklama biçimine getirilmiş eğitim verisi.

## Veri depoda değil

Eğitim dosyaları (41 MB) telif nedeniyle bu depoda tutulmuyor. Bu klasör veriyi
**üreten ve doğrulayan kodu** içeriyor. Veri dosyalarının konumu için depo sahibine
başvurun.

Beklenen yerleşim (Colab defterleri bunu varsayar):

```
MyDrive/tus_data/TUS_Dataset_Final.zip
  veri/egitim.jsonl            23.957 kayıt
  veri/test_cevap.jsonl           316 soru  (yalnızca doğru şık)
  veri/test_aciklamali.jsonl      348 soru  (şık + açıklama)
  veri/test_cikarilan.jsonl       174 soru  (sızıntı/kopya, kullanılmaz)
  veri/elenen/*.jsonl            1.043 kayıt, nedeniyle birlikte
  veri/istatistik.json
```

## Kayıt biçimi

```json
{
  "id": "a1b2c3d4e5f6",
  "messages": [
    {"role": "user", "content": "soru metni\nA) ...\nE) ..."},
    {"role": "assistant", "content": "<think>\naçıklama\n</think>\n\nDoğru cevap: C"}
  ],
  "meta": {"kaynak": "...", "ders": "..."}
}
```

## Sayılar

| | |
|---|---|
| İncelenen kayıt | 24.945 (%100, tek tek) |
| Eğitime giren | 23.957 |
| Elenen | 1.043 |
| Test seti | 664 (316 cevaplı + 348 açıklamalı) |
| Eğitim–test sızıntısı | 0 |

Elenme nedenleri `istatistik.json` içinde; en büyük kalemler şüpheli cevap anahtarı
(428), bağlam gerektiren soru (71), şık kaybı (155), bozuk soru metni (101).

## Sızıntı kontrolü hakkında

İlk kontrolde tam eşleşme (normalize edilmiş metin) kullanıldı ve "sızıntı yok"
sonucu çıktı. **Bu yanlıştı.** Tek karakterlik farklar (`tarafından` / `tarafindan`)
tam eşleşmeyi atlatıyordu. Jaccard benzerliği ≥ 0,8 ile tekrar bakıldığında 100
test sorusunun eğitim setinde eşi olduğu görüldü; bunlar çıkarıldı. Sonraki
kontroller Jaccard ile yapılmalı — `kirlilik_tara.py`.

## Hat

| Dosya | İş |
|---|---|
| `pipeline/v3_temizlik.py` | Metin onarımı: ASCII Türkçe → Türkçe, yapışık kelime ayırma, LaTeX temizliği |
| `pipeline/v3_uret.py` | Ham OCR çıktısı → SFT kayıtları; elenenleri nedeniyle ayırır |
| `pipeline/v3_dogrula.py` | Biçim ve tutarlılık doğrulaması |
| `pipeline/incele.py` | Kayıt kayıt inceleme aracı (append-only karar dosyası) |
| `pipeline/eval_temizle.py` `eval_incele.py` `eval_uygula.py` | Test setinin ayrı hattı |
| `pipeline/anahtar_denetim.py` | Cevap anahtarı ile açıklamanın çelişmesini arar |
| `pipeline/yazim_denetim.py` | Yeniden yazılmış açıklamalarda kaynakta olmayan sayı/kısaltma arar |
| `pipeline/kirlilik_tara.py` | Eğitim–test örtüşmesi (Jaccard) |

## Temizlikte öğrenilenler

İki hata pahalıya mal oldu, ikisi de kodda düzeltildi:

**Yapışık kelime ayırma deasciification'dan önce çalışıyordu.** Elle onarılmış
3.800 açıklama bölünerek bozuldu. Sıra düzeltildi; elle onarılmış kayıtlarda
ayırma hiç çalışmıyor.

**Deasciifier kelime içi büyük harfleri yok ediyordu** — IgM → Igm, cAMP → camp,
pH → ph, NaCl → Nacl. 492 kayıtta tıbbi kısaltmalar bozulmuştu. `v3_temizlik.py`
içindeki koruma:

```python
if any(c.isupper() for c in w[1:]) and not w.isupper():
    return w
```

Bu kalıp ASCII Türkçe kelimelerde oluşmaz, dolayısıyla güvenli.

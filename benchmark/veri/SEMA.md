# Benchmark veri seti şeması

Her satır bir JSON nesnesi (JSONL). Dosya adı `<yil>_<kaynak>.jsonl`,
örnek: `2025_tus.jsonl`, `2026_deneme.jsonl`.

## Zorunlu alanlar

| Alan | Tip | Açıklama |
|---|---|---|
| `id` | string | Set içinde benzersiz |
| `soru` | string | Soru kökü **ve** şıklar. Şıklar `A)` … `E)` biçiminde, her biri kendi satırında |
| `dogru_cevap` | string | Tek harf, `A`–`E` |

## Opsiyonel alanlar

| Alan | Tip | Açıklama |
|---|---|---|
| `aciklama` | string | Referans açıklama. Doğruluk ölçümünde kullanılmaz |
| `ders` | string | Anatomi, Farmakoloji, … Ders bazında kırılım için |
| `yil` | int | Sınav yılı |
| `kaynak` | string | Hangi sınav/soru bankası |

## Örnek

```json
{"id": "2025_014",
 "soru": "Aşağıdaki tularemi türlerinden hangisinde lenfadenopati görülmez?\nA) Tifoidal\nB) Orofaringeal\nC) Oküloglandüler\nD) Glandüler\nE) Gastrointestinal",
 "dogru_cevap": "A",
 "ders": "Mikrobiyoloji",
 "yil": 2025}
```

## `dogrula.py` neyi reddeder

- Zorunlu alan eksik, `dogru_cevap` A–E dışında
- 4'ten az şık
- Set içinde kopya soru (Jaccard ≥ 0,9)
- Eğitim setiyle leakage (Jaccard ≥ 0,7) — parmak izi dosyasına karşı
- Doğru cevap harfi dağılımı aşırı dengesiz (bir harf > %40)

İlk dördü PR'ı kırar, sonuncusu uyarı verir.

## Neden bu kontroller

Leakage kontrolünü ilk sefer `exact match` ile yapmıştık ve "leakage yok"
sonucu almıştık. **Yanlıştı** — tek karakterlik farklar (`tarafından` /
`tarafindan`) kontrolü atlatıyordu. Jaccard'a geçince 316 soruluk sette
100 sorunun eğitim setinde eşi olduğu çıktı. O yüzden bu kontrol CI'da.

# Ölçümler

Hepsi aynı 316 soruluk temiz test setinde. Anlamlılık testleri McNemar (exact,
iki yanlı), eşli — yani aynı sorular üzerinde aynı modelin iki yapılandırması
karşılaştırıldı.

## Ana tablo

| Yapılandırma | Doğruluk |
|---|---|
| Qwen3.5-4B ham | 62,3% |
| + SFT v1 | 70,9% |
| + SFT v2 | 71,8% |
| v2 + düz RAG (k=6, bağlam 6000) | 83,9% |
| v2 + hibrit + rerank | 85,1% |
| **v1 + hibrit + rerank** | **88,6%** |
| v1 + hibrit + rerank, temizlenmiş korpus | 87,0% |

## Anlamlılık

| Karşılaştırma | Fark | lehte / aleyhte | p |
|---|---|---|---|
| ham → SFT v1 | +8,5 | — | 0,003 |
| v2: RAG yok → düz RAG | +11,4 | 56 / 20 | < 0,0001 |
| v2: RAG yok → hibrit+rerank | +12,7 | 57 / 17 | < 0,0001 |
| v2: düz RAG → hibrit+rerank | +1,3 | 16 / 12 | 0,57 |
| v1 → v2 (hibrit+rerank, aynı belgeler) | −3,5 | 2 / 13 | **0,007** |
| v1: ham → temiz korpus | −1,6 | 0 / 5 | 0,06 |
| v2: ham → temiz korpus | −0,6 | 2 / 4 | 0,69 |

v1'de düz RAG → hibrit+rerank adımı +5 puan ve anlamlıydı (p = 0,007); v2'de
aynı adım etkisini kaybediyor. İki ayrı işaret aynı yöne bakıyor: v2 iyi bağlamı
değerlendiremiyor.

## Yanıt süresi

A100-40GB, tek soru, model ve reranker bellekte, toplu işlem yok:

| Aşama | 64 token | 300 token |
|---|---|---|
| yoğun getirme | 0,12 sn | 0,12 sn |
| BM25 | 0,15 sn | 0,15 sn |
| yeniden sıralama | 0,26 sn | 0,25 sn |
| üretim | 5,70 sn | 11,32 sn |
| **toplam (medyan)** | **6,23 sn** | **11,82 sn** |
| p95 | 6,51 sn | 27,41 sn |

Getirme toplamın %8'i — hibrit + rerank'in gecikme bedeli yarım saniye.

Üretim hızı 11 token/sn, A100'de 4B model için **düşük**. Flash Attention 2
yüklü değil, LoRA adapter taban ağırlıklara birleştirilmemiş (`merge_and_unload`)
ve iki adapter aynı anda yüklü. Bu sayı optimize edilmiş bir alt sınır değil.

## Denenip elenen yöntemler

Hepsi ölçüldü, hiçbiri kullanılmıyor:

| Yöntem | Sonuç | Neden bırakıldı |
|---|---|---|
| Modelin kendi arama sorgusunu üretmesi | 73,4% ve 78,5% | Düz RAG'in (83,9%) altında. E5 gömmesi tam cümlede daha iyi çalışıyor; soru metninin kendisi daha iyi bir sorgu. |
| k ve bağlam uzunluğu taraması | 85,4% | p = 0,38 |
| Tek başına yeniden sıralama | — | p = 0,17 |
| Tek başına hibrit arama | — | p = 0,50 |
| Korpus temizliği | 87,0% | −1,6 puan, 5 bozulma 0 düzelme |
| Cevaplamada v2 adapter | 85,1% | −3,5 puan, p = 0,007 |

## v2 neden RAG altında geriliyor

13 gerileyen soru üzerinde teşhis:

| | gerileyen 13 | tüm 316 |
|---|---|---|
| v2'nin RAG'li cevabı = RAG'siz cevabı | %23 | %72 |
| doğru şıkkın bağlamda bulunması | %80 | %71 |

İlk satır belirleyici: normalde v2 RAG'li ve RAG'siz aynı cevabı veriyor. Gerilediği
sorularda cevabını değiştirmiş — ve yanlışa değiştirmiş. Yani bağlamı yok saymıyor,
bağlam tarafından yanıltılıyor.

İkinci satır getirmeyi aklıyor: kanıt bağlamda vardı, ortalamadan daha güçlü biçimde.
Çıktı uzunlukları iki adapter'da aynı, yani biçim/kesilme sorunu da değil.

Gözlem (iddia değil): incelenen beş örneğin üçü olumsuz soruydu ("hangisinde
görülmez", "hangisi değildir") ve v2 her üçünde bağlamda **listelenen** bir şıkkı
seçmişti. Muhtemel mekanizma, v2'ye eklenen sorgu üretme görevinin modele
"metindeki göze çarpan terimleri çıkar" alışkanlığı kazandırması. 13 soruda 3
örnek; doğrulanmadı.

## Hata analizi (SFT v1, RAG öncesi)

88 yanlış cevap incelendi: biçim yanlılığı yok, harf yanlılığı yok, yalnızca 1
bozuk soru. Hatalar ince taneli olgu eksikliği — modelin konuyu bilip ayrıntıyı
bilmemesi. RAG'in SFT'den daha çok fayda sağlamasının nedeni bu.

## Ölçüm hijyeni

- `repetition_penalty` doğruluk ölçümünde **kullanılmıyor**. HF cezayı prompt
  tokenlarına da uyguluyor, bu da modeli şık metnini kopyalamaktan caydırıyor ve
  doğruluğu yapay olarak düşürüyor (68,4% yerine gerçek 70,9%).
- Tahminler soru bazında kaydediliyor (`MyDrive/tus_sft/rag_sonuc/*.json`), böylece
  sonraki ölçümler eşli karşılaştırma yapabiliyor. İlk turda bu yapılmamıştı ve
  v1 ile eşli test imkânsız hale gelmişti.
- Üretim `do_sample=False` — deterministik. Yine de batch dolgusu bf16'da birebir
  aynı sayıları vermiyor; iki ayrı oturumda ±2 soruluk fark normal.

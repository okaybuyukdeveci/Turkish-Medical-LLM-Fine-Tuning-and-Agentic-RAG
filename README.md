# Türkçe Medikal LLM — İnce Ayar ve Agentic RAG

TUS (Tıpta Uzmanlık Sınavı) sorularını çözen bir sistem. İki parçadan oluşuyor ve
ikisi birlikte çalışıyor: ince ayarlanmış bir dil modeli ve ders kitaplarından
getirme yapan bir RAG katmanı.

## Depo haritası

| Klasör | İçerik | Sahibi |
|---|---|---|
| `agentic_rag/` | LangGraph tabanlı RAG sistemi, Qdrant indeksleme, korpus | RAG ekibi |
| `dataset/` | Eğitim verisinin üretim ve temizlik hattı | bu dal |
| `fine_tuning/` | Qwen3.5-4B LoRA ince ayarı (v1 ve v2) | bu dal |
| `entegrasyon/` | İnce ayarlı modelin RAG'e bağlanması ve ölçümler | bu dal |
| `benchmark/` | Gerçek ÖSYM sınavlarında tekrarlanabilir ölçüm hattı | bu dal |

`agentic_rag/` varsayılan olarak Together API üzerinden GLM-5.3-Flash kullanır.
`entegrasyon/` altındaki defterler onun yerine kendi ince ayarlı modelimizi koyar;
Together anahtarı gerektirmez.

## Sonuç özeti

2021-2025 arası **yedi gerçek ÖSYM sınavı, 1.471 soru**. Eğitim setiyle fazla
örtüşen iki sınav ortalamadan çıkarıldı.

| Yapılandırma | Doğruluk |
|---|---|
| Qwen3.5-4B ham | 63,0% |
| + SFT (v1) | 73,8% |
| + her soruda RAG | 78,8% |
| **+ kapılı RAG** | **80,3%** |
| oracle (kusursuz kapı) | 86,4% |

**Kapılı RAG:** arama her soruda değil, yalnızca model cevabından emin değilken
çalışıyor — soruların ~%45'inde. Her soruda aramaya göre hem daha doğru hem
yarı maliyetli.

Kendi 316 soruluk iç test setimizde bu sistem %88,6 yapıyor, ama o sayı
**başlık olarak kullanılmıyor**: set RAG korpusuyla aynı yayınevi ekosisteminden
geliyor ve gerçek sınav performansını abartıyor.

Yanıt süresi (A100, tek soru, model bellekte): kısa cevap 6,2 sn, tam açıklama 11,8 sn.
Getirme bunun yalnızca %8'i.

Sınav bazında tablo, anlamlılık testleri ve denenip **işe yaramayan** yöntemler
(v2 adapter, modelin ürettiği sorgu, v3 karar adapteri):
[`benchmark/SONUCLAR.md`](benchmark/SONUCLAR.md) · iç test setinin ayrıntısı:
[`entegrasyon/sonuclar/OLCUMLER.md`](entegrasyon/sonuclar/OLCUMLER.md)

## Nereden başlamalı

- Veriyi anlamak için: `dataset/README.md`
- Modeli yeniden eğitmek için: `fine_tuning/README.md`
- Sistemi çalıştırmak ve ölçmek için: `entegrasyon/README.md`
- RAG katmanının kendisi için: `agentic_rag/README.md`
- Gerçek sınavlarda ölçmek için: `benchmark/KURULUM.md`

## Bilinen sorun

`agentic_rag/data/` altındaki 24 markdown dosyası depoda **iki farklı Unicode
kodlamasıyla** duruyor (NFC ve NFD). macOS bunları tek dosya gibi gösterir, Linux
iki ayrı dosya olarak görür — Colab'da indeksleme korpusu ikiye katlar
(28.539 yerine 57.078 chunk). `entegrasyon/hucreler/01_rag_olcum/2_hazirlik.py`
bunu klasör adlarını birleştirici işaretlerden arındırarak gruplayıp çözüyor.
Kalıcı çözüm depodaki kopya klasörlerin silinmesi; bu dalda onların dosyalarına
dokunulmadı.

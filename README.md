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

`agentic_rag/` varsayılan olarak Together API üzerinden GLM-5.3-Flash kullanır.
`entegrasyon/` altındaki defterler onun yerine kendi ince ayarlı modelimizi koyar;
Together anahtarı gerektirmez.

## Sonuç özeti

316 soruluk temiz test setinde, hepsi aynı set üzerinde ölçüldü:

| Yapılandırma | Doğruluk |
|---|---|
| Qwen3.5-4B ham | 62,3% |
| + SFT v1 | 70,9% |
| + düz RAG | 83,9% |
| + hibrit getirme + yeniden sıralama | **88,6%** |

Yanıt süresi (A100, tek soru, model bellekte): kısa cevap 6,2 sn, tam açıklama 11,8 sn.
Getirme bunun yalnızca %8'i.

Ayrıntılı tablo, anlamlılık testleri ve denenip **işe yaramayan** yöntemler:
[`entegrasyon/sonuclar/OLCUMLER.md`](entegrasyon/sonuclar/OLCUMLER.md)

## Nereden başlamalı

- Veriyi anlamak için: `dataset/README.md`
- Modeli yeniden eğitmek için: `fine_tuning/README.md`
- Sistemi çalıştırmak ve ölçmek için: `entegrasyon/README.md`
- RAG katmanının kendisi için: `agentic_rag/README.md`

## Bilinen sorun

`agentic_rag/data/` altındaki 24 markdown dosyası depoda **iki farklı Unicode
kodlamasıyla** duruyor (NFC ve NFD). macOS bunları tek dosya gibi gösterir, Linux
iki ayrı dosya olarak görür — Colab'da indeksleme korpusu ikiye katlar
(28.539 yerine 57.078 chunk). `entegrasyon/hucreler/01_rag_olcum/2_hazirlik.py`
bunu klasör adlarını birleştirici işaretlerden arındırarak gruplayıp çözüyor.
Kalıcı çözüm depodaki kopya klasörlerin silinmesi; bu dalda onların dosyalarına
dokunulmadı.

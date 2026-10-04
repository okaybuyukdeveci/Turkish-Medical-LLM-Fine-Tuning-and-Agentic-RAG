import json, shutil, os
from pathlib import Path
from google.colab import files

KAYNAK = "/content/drive/MyDrive/tus_sft/qwen35_4b/final_adapter"
SONUC  = "/content/drive/MyDrive/tus_sft/qwen35_4b/sonuc.json"
PAKET  = "/content/model"

assert os.path.isdir(KAYNAK), f"adapter bulunamadi: {KAYNAK}"
shutil.rmtree(PAKET, ignore_errors=True)
shutil.copytree(KAYNAK, PAKET)
if os.path.exists(SONUC):
    shutil.copy(SONUC, f"{PAKET}/sonuc.json")

KULLANIM = """# İnce ayarlı TUS modeli

Qwen3.5 4B üzerine eğitilmiş LoRA adapter. Temel model dahil değildir —
yüklerken `Qwen/Qwen3.5-4B` indirilir, adapter onun üzerine bindirilir.

## Yükleme

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    "Qwen/Qwen3.5-4B", max_seq_length=2048, dtype="bfloat16",
)
model.load_adapter("model")
FastLanguageModel.for_inference(model)
```

## Soru sorma

```python
SYSTEM = ("Sen bir tıp uzmanısın. Aşağıdaki TUS sorusunu çöz. "
          "İlk satıra 'Doğru cevap: <harf>) <şık metni>' yaz, "
          "ardından kısa bir açıklama ver.")

soru = "Aşağıdakilerden hangisi...?\\nA) ...\\nB) ...\\nC) ...\\nD) ...\\nE) ..."
msgs = [{"role": "system", "content": SYSTEM},
        {"role": "user", "content": soru}]
p = tokenizer.apply_chat_template(msgs, tokenize=False,
                                  add_generation_prompt=True, enable_thinking=False)
ids = tokenizer(p, return_tensors="pt", add_special_tokens=False).to(model.device)
out = model.generate(**ids, max_new_tokens=400, do_sample=False, repetition_penalty=1.1)
print(tokenizer.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True))
```

`repetition_penalty=1.1` önemli — onsuz model bazı cümleleri tekrarlıyor.

## Sonuçlar

`sonuc.json` dosyasında: taban %62,3 → ince ayar %70,9 (+8,5 puan, p=0,003),
316 soruluk kontrol edilmiş test setinde.
"""
Path(f"{PAKET}/KULLANIM.md").write_text(KULLANIM, encoding="utf-8")

print("paket icerigi:")
for f in sorted(os.listdir(PAKET)):
    mb = os.path.getsize(f"{PAKET}/{f}") / 1e6
    print(f"  {f:<34} {mb:7.2f} MB")

shutil.make_archive("/content/model", "zip", PAKET)
print(f"\nzip: {os.path.getsize('/content/model.zip')/1e6:.1f} MB")
files.download("/content/model.zip")

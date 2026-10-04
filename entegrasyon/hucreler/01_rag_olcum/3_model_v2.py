import re, torch
from pathlib import Path
from unsloth import FastLanguageModel
from peft import PeftModel

MODEL_NAME  = "Qwen/Qwen3.5-4B"
ADAPTER_V2  = "/content/drive/MyDrive/tus_sft/qwen35_4b_v2/final_adapter"
ADAPTER_V1  = "/content/drive/MyDrive/tus_sft/qwen35_4b/final_adapter"
MAX_SEQ_LEN, SEED = 4096, 42

assert Path(ADAPTER_V2).is_dir(), f"v2 adapter yok: {ADAPTER_V2}"

SYSTEM_SFT = ("Sen bir tıp uzmanısın. Aşağıdaki TUS sorusunu çöz. "
              "İlk satıra 'Doğru cevap: <harf>) <şık metni>' yaz, ardından kısa bir açıklama ver.")

SYSTEM_RAG = ("Sen bir tıp uzmanısın. Aşağıda bir TUS sorusu ve ders kitaplarından alınmış "
              "ilgili bölümler var. Kaynak metni kullanarak soruyu çöz. "
              "İlk satıra 'Doğru cevap: <harf>) <şık metni>' yaz, ardından kısa bir açıklama ver. "
              "Kaynak metin soruyu cevaplamıyorsa kendi tıbbi bilgini kullan.")

model, tokenizer = FastLanguageModel.from_pretrained(
    MODEL_NAME, max_seq_length=MAX_SEQ_LEN, dtype=torch.bfloat16, load_in_4bit=False)
tok = getattr(tokenizer, "tokenizer", tokenizer)
if tok.pad_token is None: tok.pad_token = tok.eos_token

model = PeftModel.from_pretrained(model, ADAPTER_V2, adapter_name="v2")
AKTIF = "v2"

V1_VAR = Path(ADAPTER_V1).is_dir()
if V1_VAR:
    try:
        model.load_adapter(ADAPTER_V1, adapter_name="v1")
        print("v1 adapter da yuklendi — esli karsilastirma yapilabilir")
    except Exception as e:
        V1_VAR = False
        print("v1 adapter yuklenemedi, sadece v2 olculecek:", type(e).__name__, e)
else:
    print("v1 adapter Drive'da bulunamadi, sadece v2 olculecek")

def adapter_sec(ad):
    global AKTIF
    model.set_adapter(ad); AKTIF = ad
    print(f"aktif adapter: {ad}")

FastLanguageModel.for_inference(model)
model.set_adapter("v2")
print("model hazir | aktif adapter: v2")

def prompt(sistem, kullanici):
    msgs = [{"role": "system", "content": sistem}, {"role": "user", "content": kullanici}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)

SON = re.compile(r"Doğru cevap\s*:?\s*\(?([A-E])\)?")
HER = re.compile(r"(?:Doğru\s*)?(?:cevap|yanıt)\s*[:\-]?\s*\(?([A-E])\)?", re.I)
def parse_cevap(t):
    c = re.sub(r"<think>.*?</think>", "", t or "", flags=re.S).strip()
    m = SON.search(c) or HER.search(c) or re.search(r"\b([A-E])\)", c)
    return m.group(1).upper() if m else None

@torch.no_grad()
def uret(prompts, max_new_tokens=96, bs=8, tekrar_cezasi=1.0, etiket=""):
    tok.padding_side = "left"
    outs = []
    for i in range(0, len(prompts), bs):
        b = prompts[i:i+bs]
        enc = tok(b, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        g = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                           repetition_penalty=tekrar_cezasi, pad_token_id=tok.pad_token_id)
        outs += tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"  {etiket}{min(i+bs,len(prompts))}/{len(prompts)}", end="\r")
    print()
    return outs

def dogruluk(outs, rows, etiket=""):
    p = [parse_cevap(o) for o in outs]
    acc = sum(a == r["letter"] for a, r in zip(p, rows)) / len(rows)
    yok = sum(a is None for a in p)
    print(f"{etiket}dogruluk {acc:.1%} | cevapsiz {yok}/{len(rows)}")
    return acc, p

MAX_BAGLAM = 6000
def baglam_kur(docs, limit=MAX_BAGLAM):
    parcalar, toplam = [], 0
    for i, d in enumerate(docs, 1):
        t = d.page_content.strip()
        kaynak = (d.metadata or {}).get("source", "?")
        blok = f"[Kaynak {i} — {Path(kaynak).stem}]\n{t}"
        if toplam + len(blok) > limit:
            kalan = limit - toplam
            if kalan > 400: parcalar.append(blok[:kalan])
            break
        parcalar.append(blok); toplam += len(blok)
    return "\n\n".join(parcalar)

print("\n[referans] RAG'siz olcum — v2")
sade_outs = uret([prompt(SYSTEM_SFT, r["question"]) for r in test_rows], 64, bs=16, etiket="[sade] ")
sade_acc, sade_pred = dogruluk(sade_outs, test_rows, "[sade v2] ")
print("beklenen ~71,8% (SFT v2 olcumu) | v1: 70,9%")

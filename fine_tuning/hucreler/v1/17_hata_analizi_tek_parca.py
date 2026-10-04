import json, re, random, zipfile, glob, os, torch
from pathlib import Path
from collections import Counter
from google.colab import drive, files

drive.mount("/content/drive", force_remount=False)

ZIP_PATH = "/content/drive/MyDrive/tus_data/TUS_Dataset_Final-20261002T154401Z-1-001.zip"
ADAPTER  = "/content/drive/MyDrive/tus_sft/qwen35_4b/final_adapter"
HEDEF    = "/content/tus_veri"
MODEL_NAME, MAX_SEQ_LEN, SEED = "Qwen/Qwen3.5-4B", 2048, 42

SYSTEM_PROMPT = ("Sen bir tıp uzmanısın. Aşağıdaki TUS sorusunu çöz. "
                 "İlk satıra 'Doğru cevap: <harf>) <şık metni>' yaz, ardından kısa bir açıklama ver.")

if not glob.glob(f"{HEDEF}/**/egitim.jsonl", recursive=True):
    with zipfile.ZipFile(ZIP_PATH) as z: z.extractall(HEDEF)
DATA_DIR = str(Path(glob.glob(f"{HEDEF}/**/egitim.jsonl", recursive=True)[0]).parent)
assert os.path.isdir(ADAPTER), f"adapter yok: {ADAPTER}"

ANSWER = re.compile(r"Doğru cevap:\s*([A-E])")
test_acc = [{"question": r["soru"], "letter": r["dogru_cevap"],
             "group": (r.get("meta") or {}).get("group", "?")}
            for r in (json.loads(l) for l in open(f"{DATA_DIR}/test_cevap.jsonl", encoding="utf-8"))]
print(f"test: {len(test_acc)} soru | DATA_DIR: {DATA_DIR}")

from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    MODEL_NAME, max_seq_length=MAX_SEQ_LEN, dtype=torch.bfloat16, load_in_4bit=False)
tok = getattr(tokenizer, "tokenizer", tokenizer)
if tok.pad_token is None: tok.pad_token = tok.eos_token

def build_prompt(q):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": q}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)

BASLIK = re.compile(r"^\s*Doğru cevap\s*:?\s*\(?([A-E])\)?")
HERHANGI = re.compile(r"(?:Doğru\s*)?(?:cevap|yanıt|answer)\s*[:\-]?\s*\(?([A-E])\)?", re.I)
def parse_loose(t):
    c = re.sub(r"<think>.*?</think>", "", t, flags=re.S).strip()
    m = BASLIK.match(c) or HERHANGI.search(c) or re.search(r"\b([A-E])\)", c)
    return m.group(1).upper() if m else None

@torch.no_grad()
def uret(rows, max_new_tokens, bs=16, etiket=""):
    FastLanguageModel.for_inference(model)
    tok.padding_side = "left"
    outs = []
    for i in range(0, len(rows), bs):
        b = rows[i:i+bs]
        enc = tok([build_prompt(r["question"]) for r in b], return_tensors="pt",
                  padding=True, add_special_tokens=False).to(model.device)
        g = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                           pad_token_id=tok.pad_token_id)
        outs += tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"  {etiket}{min(i+bs,len(rows))}/{len(rows)}", end="\r")
    acc = sum(parse_loose(o) == r["letter"] for o, r in zip(outs, rows)) / len(rows)
    print(f"\n  {etiket}dogruluk {acc:.1%}")
    return outs, acc

print("\n[1/2] taban model olculuyor (uzun, ~12 dk)")
base_outs, base_acc = uret(test_acc, 512, etiket="[taban] ")

print("\n[2/2] adapter yukleniyor")
from peft import PeftModel
model = PeftModel.from_pretrained(model, ADAPTER)
ft_outs, ft_acc = uret(test_acc, 64, etiket="[ft] ")

def sik(soru, harf):
    m = re.search(rf"(?m)^{harf}\) (.+)$", soru)
    return m.group(1).strip() if m else ""

kayitlar = []
for r, bo, fo in zip(test_acc, base_outs, ft_outs):
    bp, fp, g = parse_loose(bo), parse_loose(fo), r["letter"]
    kayitlar.append({"soru": r["question"], "kaynak": r.get("group", "?"),
        "dogru_harf": g, "dogru_metin": sik(r["question"], g),
        "taban_harf": bp, "taban_metin": sik(r["question"], bp) if bp else "",
        "ft_harf": fp, "ft_metin": sik(r["question"], fp) if fp else "",
        "ft_cikti": fo.strip()[:400], "taban_dogru": bp == g, "ft_dogru": fp == g,
        "durum": ("ikisi_de_dogru" if bp == g and fp == g else
                  "ikisi_de_yanlis" if bp != g and fp != g else
                  "ft_kazandi" if fp == g else "ft_kaybetti")})

print("\ndurum:", dict(Counter(k["durum"] for k in kayitlar)))
print(f"taban {base_acc:.1%} | ft {ft_acc:.1%}")
with open("/content/hata_analizi.json", "w", encoding="utf-8") as f:
    json.dump(kayitlar, f, ensure_ascii=False, indent=1)
files.download("/content/hata_analizi.json")

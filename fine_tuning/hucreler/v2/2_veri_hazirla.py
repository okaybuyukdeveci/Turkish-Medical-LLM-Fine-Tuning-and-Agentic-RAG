import os, re, json, glob, zipfile, random, unicodedata, torch
from pathlib import Path
from google.colab import drive
drive.mount("/content/drive", force_remount=False)

MODEL_NAME = "Qwen/Qwen3.5-4B"
ZIP_PATH   = "/content/drive/MyDrive/tus_data/TUS_Dataset_Final-20261002T154401Z-1-001.zip"
HEDEF, SEED = "/content/tus_veri", 42
KARMA_ORAN = 0.13          # egitim setinin ~%13'u talimat ornegi
MAX_SEQ_LEN = 2048

if not glob.glob(f"{HEDEF}/**/egitim.jsonl", recursive=True):
    with zipfile.ZipFile(ZIP_PATH) as z: z.extractall(HEDEF)
DATA_DIR = str(Path(glob.glob(f"{HEDEF}/**/egitim.jsonl", recursive=True)[0]).parent)

CEV  = re.compile(r"^<think>\n(.*)\n</think>\n\nDoğru cevap: ([A-E])$", re.S)
SIK  = re.compile(r"^([A-E])\) (.+)$", re.M)
def nrm(s):
    s = unicodedata.normalize("NFD", s.lower()).translate(str.maketrans("çğıöşü", "cgiosu"))
    return re.sub(r"[^a-z0-9 ]", " ", s)

ham = [json.loads(l) for l in open(f"{DATA_DIR}/egitim.jsonl", encoding="utf-8")]
guclu, zayif = [], []
for r in ham:
    m = CEV.match(r["messages"][1]["content"])
    if not m: continue
    acik, harf = m.group(1), m.group(2)
    if len(acik.strip()) < 40: continue
    sk = dict(SIK.findall(r["messages"][0]["content"]))
    if list(sk) not in (list("ABCDE"), list("ABCD")) or harf not in sk: continue
    kel = [w for w in nrm(sk[harf]).split() if len(w) >= 5]
    (zayif if (kel and not any(w in nrm(acik) for w in kel)) else guclu).append(
        {"question": r["messages"][0]["content"], "letter": harf, "aciklama": acik,
         "sik": sk[harf], "group": (r.get("meta") or {}).get("group", "?")})
print(f"toplam {len(ham)} -> guclu {len(guclu)} | ayiklanan zayif {len(zayif)}")

def cevap_once(r):
    return {"gorev": "tus", "question": r["question"],
            "answer": f"Doğru cevap: {r['letter']}) {r['sik']}\n\nAçıklama: {r['aciklama'][:3000]}"}
tus_rows = [cevap_once(r) for r in guclu]

from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    MODEL_NAME, max_seq_length=MAX_SEQ_LEN, dtype=torch.bfloat16, load_in_4bit=False)
tok = getattr(tokenizer, "tokenizer", tokenizer)
if tok.pad_token is None: tok.pad_token = tok.eos_token
FastLanguageModel.for_inference(model)

def prompt(sis, kul):
    return tok.apply_chat_template([{"role":"system","content":sis},{"role":"user","content":kul}],
                                   tokenize=False, add_generation_prompt=True, enable_thinking=False)

@torch.no_grad()
def uret(prompts, mnt=64, bs=16, etiket=""):
    tok.padding_side = "left"; outs = []
    for i in range(0, len(prompts), bs):
        b = prompts[i:i+bs]
        enc = tok(b, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        g = model.generate(**enc, max_new_tokens=mnt, do_sample=False, pad_token_id=tok.pad_token_id)
        outs += tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"  {etiket}{min(i+bs,len(prompts))}/{len(prompts)}", end="\r")
    print(); return outs

SIS_SORGU = ("Sen bir tıp kütüphanecisisin. Sana bir TUS sorusu verilecek. Soruyu CEVAPLAMA. "
             "Yalnızca bu soruyu çözmek için ders kitabında aranması gereken tıbbi anahtar "
             "kelimeleri tek satırda, virgülle ayırarak yaz.")
SIS_TERIM = ("Sen bir tıp uzmanısın. Sana bir tıbbi terim verilecek. "
             "Terimi iki cümleyle, sade Türkçeyle açıkla. Soru sorma, liste yapma.")

random.Random(SEED).shuffle(guclu)
n_her = int(len(tus_rows) * KARMA_ORAN / 2)
print(f"\nkarma veri uretiliyor: {n_her} sorgu + {n_her} terim")

s_kaynak = guclu[:n_her]
s_out = uret([prompt(SIS_SORGU, r["question"]) for r in s_kaynak], 48, etiket="[sorgu] ")
def temiz(t):
    s = re.sub(r"<think>.*?</think>", "", t or "", flags=re.S).strip().split("\n")[0]
    return re.sub(r"^(arama|sorgu|anahtar kelimeler)\s*[:\-]\s*", "", s.strip(' "\'`'), flags=re.I).strip()
sorgu_rows = [{"gorev": "sorgu", "sistem": SIS_SORGU, "question": r["question"], "answer": temiz(o)}
              for r, o in zip(s_kaynak, s_out)
              if 5 <= len(temiz(o)) <= 200 and not re.match(r"^[A-E][\)\.]", temiz(o))]

TERIM = re.compile(r"\b[A-ZÇĞİÖŞÜ][a-zçğıöşü]{5,}(?:\s+[a-zçğıöşü]{4,})?\b")
havuz = []
for r in guclu[n_her:n_her*4]:
    t = TERIM.findall(r["aciklama"][:400])
    if t: havuz.append(t[0])
havuz = list(dict.fromkeys(havuz))[:n_her]
t_out = uret([prompt(SIS_TERIM, f"{t} nedir?") for t in havuz], 120, etiket="[terim] ")
terim_rows = [{"gorev": "terim", "sistem": SIS_TERIM, "question": f"{t} nedir?",
               "answer": re.sub(r"<think>.*?</think>", "", o, flags=re.S).strip()}
              for t, o in zip(havuz, t_out) if 40 <= len(o.strip()) <= 700]

print(f"\nkarma: sorgu {len(sorgu_rows)} | terim {len(terim_rows)}")
print("ornek sorgu :", sorgu_rows[0]["answer"][:90])
print("ornek terim :", terim_rows[0]["question"], "->", terim_rows[0]["answer"][:90])

egitim = tus_rows + sorgu_rows + terim_rows
random.Random(SEED).shuffle(egitim)
nv = max(1, int(len(egitim) * 0.02))
val_rows, train_rows = egitim[:nv], egitim[nv:]
print(f"\nTOPLAM egitim {len(train_rows)} | val {len(val_rows)}")
print(f"  tus {len(tus_rows)} (%{100*len(tus_rows)/len(egitim):.0f}) | "
      f"sorgu {len(sorgu_rows)} | terim {len(terim_rows)}")

test_rows = [{"question": r["soru"], "letter": r["dogru_cevap"],
              "group": (r.get("meta") or {}).get("group", "?")}
             for r in (json.loads(l) for l in open(f"{DATA_DIR}/test_cevap.jsonl", encoding="utf-8"))]
print(f"test {len(test_rows)}")

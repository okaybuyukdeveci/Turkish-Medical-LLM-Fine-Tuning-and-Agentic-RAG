import json, re, random, unicodedata
from collections import Counter

def load_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f]

THINK  = re.compile(r"<think>\n?(.*?)\n?</think>", re.S)
ANSWER = re.compile(r"Doğru cevap:\s*([A-E])")
OPTION = re.compile(r"^([A-E])\) (.+)$", re.M)

def option_text(question, letter):
    for h, t in OPTION.findall(question):
        if h == letter:
            return t.strip()
    return ""

def shorten(text, limit=MAX_EXPL_CHARS):
    if len(text) <= limit:
        return text
    cut = text[:limit]
    for sep in ("\n\n", "\n", ". "):
        i = cut.rfind(sep)
        if i > limit * 0.5:
            return cut[:i + len(sep)].strip()
    return cut.strip()

def usable(rec):
    q, a = rec["messages"][0]["content"], rec["messages"][-1]["content"]
    m = ANSWER.search(a)
    if not m:
        return False
    letters = [h for h, _ in OPTION.findall(q)]
    if letters not in (list("ABCDE"), list("ABCD")) or m.group(1) not in letters:
        return False
    th = THINK.search(a)
    return bool(th and len(th.group(1).strip()) >= 40)

def to_answer_first(rec):
    q = rec["messages"][0]["content"]
    a = rec["messages"][-1]["content"]
    letter = ANSWER.search(a).group(1)
    expl   = shorten(THINK.search(a).group(1).strip())
    opt    = option_text(q, letter)
    head   = f"Doğru cevap: {letter}) {opt}" if opt else f"Doğru cevap: {letter}"
    return {"question": q,
            "answer": f"{head}\n\nAçıklama: {expl}",
            "letter": letter,
            "group": (rec.get("meta") or {}).get("group", "?")}

raw  = load_jsonl(f"{DATA_DIR}/egitim.jsonl")
kept = [r for r in raw if usable(r)]
rows = [to_answer_first(r) for r in kept]
print(f"egitim.jsonl {len(raw)} -> kullanilabilir {len(rows)} (elenen {len(raw)-len(rows)})")

random.Random(SEED).shuffle(rows)
n_val = max(1, int(len(rows) * 0.02))
val_rows, train_rows = rows[:n_val], rows[n_val:]

test_acc = [{"question": r["soru"], "letter": r["dogru_cevap"],
             "group": (r.get("meta") or {}).get("group", "?")}
            for r in load_jsonl(f"{DATA_DIR}/test_cevap.jsonl")]

test_expl = [{"question": r["messages"][0]["content"],
              "letter": ANSWER.search(r["messages"][-1]["content"]).group(1),
              "reference": THINK.search(r["messages"][-1]["content"]).group(1).strip(),
              "group": (r.get("meta") or {}).get("group", "?")}
             for r in load_jsonl(f"{DATA_DIR}/test_aciklamali.jsonl")]

if SMOKE:
    test_acc  = random.Random(SEED).sample(test_acc, 40)
    test_expl = random.Random(SEED).sample(test_expl, 10)

def key(s):
    s = unicodedata.normalize("NFKD", s.lower()).translate(str.maketrans("çğıöşüâîû", "cgiosuaiu"))
    return frozenset(w for w in re.findall(r"[a-z]{4,}", s.split("\n")[0]))

train_keys = [key(r["question"]) for r in rows]
index = {}
for i, k in enumerate(train_keys):
    for w in k:
        index.setdefault(w, []).append(i)

def leaks(q):
    t = key(q)
    if len(t) < 4:
        return False
    hits = Counter()
    for w in t:
        for i in index.get(w, ()):
            hits[i] += 1
    return any(c / len(t | train_keys[i]) >= 0.8 for i, c in hits.most_common(3))

sizan = sum(leaks(r["question"]) for r in test_acc) + sum(leaks(r["question"]) for r in test_expl)
assert sizan == 0, f"SIZINTI: {sizan} test sorusu egitim setinde de var — koşuyu durdur"
print(f"train {len(train_rows)} | val {len(val_rows)} | test(dogruluk) {len(test_acc)} | test(aciklama) {len(test_expl)}")
print("sizinti kontrolu: temiz")

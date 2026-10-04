BASLIK = re.compile(r"^\s*Doğru cevap\s*:?\s*\(?([A-E])\)?")
HERHANGI = re.compile(r"(?:Doğru\s*)?(?:cevap|yanıt|answer)\s*[:\-]?\s*\(?([A-E])\)?", re.I)

def strip_think(t):
    return re.sub(r"<think>.*?</think>", "", t, flags=re.S).strip()

def parse_strict(t):
    m = BASLIK.match(strip_think(t))
    return m.group(1).upper() if m else None

def parse_loose(t):
    c = strip_think(t)
    m = BASLIK.match(c) or HERHANGI.search(c)
    if m:
        return m.group(1).upper()
    m = re.search(r"\b([A-E])\)", c)
    return m.group(1).upper() if m else None

@torch.no_grad()
def evaluate(rows, max_new_tokens=64, batch_size=16, etiket=""):
    FastLanguageModel.for_inference(model)
    tok.padding_side = "left"
    outs = []
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        enc = tok([build_prompt(r["question"]) for r in batch],
                  return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens,
                             do_sample=False, pad_token_id=tok.pad_token_id)
        outs += tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)

    s_pred = [parse_strict(o) for o in outs]
    l_pred = [parse_loose(o)  for o in outs]
    gold   = [r["letter"] for r in rows]
    s_acc  = sum(p == g for p, g in zip(s_pred, gold)) / len(rows)
    l_acc  = sum(p == g for p, g in zip(l_pred, gold)) / len(rows)
    uyum   = sum(p is not None for p in s_pred) / len(rows)
    yok    = sum(p is None for p in l_pred)

    print(f"{etiket}dogruluk  kati {s_acc:.1%} | gevsek {l_acc:.1%}")
    print(f"{etiket}bicime uyum {uyum:.1%} | hic cevap vermeyen {yok}/{len(rows)}")

    per = {}
    for r, p in zip(rows, l_pred):
        g = r.get("group", "?")
        d, t = per.get(g, (0, 0))
        per[g] = (d + (p == r["letter"]), t + 1)
    if len(per) > 1:
        print(f"{etiket}kaynak bazinda:")
        for g, (d, t) in sorted(per.items(), key=lambda x: -x[1][1])[:6]:
            print(f"{etiket}  {g:<34} {d}/{t}  {d/t:.0%}")
    return {"strict": s_acc, "loose": l_acc, "format": uyum, "outs": outs, "per_group": per}

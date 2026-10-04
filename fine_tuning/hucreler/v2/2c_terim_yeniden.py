import re, random

KOTU = re.compile(r"^(hiç|hepsi|yukarıdaki|aşağıdaki|diğer|tümü|bunların|yalnız|sadece|I{1,3}\b)", re.I)
def temiz_terim(s):
    t = re.sub(r"\s*\(.*?\)", "", s).strip(" .,;:")
    t = re.sub(r"\s+", " ", t)
    return t

havuz = []
for r in guclu:
    t = temiz_terim(r["sik"])
    if 4 <= len(t) <= 48 and len(t.split()) <= 4 and not KOTU.match(t) \
       and not re.search(r"[0-9]{3,}|%", t):
        havuz.append(t)
havuz = list(dict.fromkeys(havuz))
random.Random(SEED).shuffle(havuz)
havuz = havuz[:1400]
print(f"sik metninden terim havuzu: {len(havuz)}")
print("ornekler:", havuz[:8])

t_out = uret([prompt(SIS_TERIM, f"{t} nedir?") for t in havuz], 120, bs=16, etiket="[terim] ")
terim_rows = [{"gorev": "terim", "sistem": SIS_TERIM, "question": f"{t} nedir?",
               "answer": re.sub(r"<think>.*?</think>", "", o, flags=re.S).strip()}
              for t, o in zip(havuz, t_out) if 40 <= len(o.strip()) <= 700]
print(f"\nterim ornegi: {len(terim_rows)}")
for r in terim_rows[:3]:
    print(f"  {r['question']}")
    print(f"    -> {r['answer'][:110]}")

egitim = tus_rows + sorgu_rows + terim_rows
random.Random(SEED).shuffle(egitim)
nv = max(1, int(len(egitim) * 0.02))
val_rows, train_rows = egitim[:nv], egitim[nv:]
print(f"\nTOPLAM egitim {len(train_rows)} | val {len(val_rows)}")
print(f"  tus {len(tus_rows)} | sorgu {len(sorgu_rows)} | terim {len(terim_rows)} "
      f"(karma %{100*(len(sorgu_rows)+len(terim_rows))/len(egitim):.0f})")

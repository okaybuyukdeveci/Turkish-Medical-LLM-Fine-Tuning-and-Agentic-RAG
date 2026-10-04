import re

EK = re.compile(r"(?:da|de|ta|te|dan|den|tan|ten|ın|in|un|ün|nın|nin|nun|nün|"
                r"ya|ye|yi|yı|yu|yü|la|le|ile|lar|ler|sı|si|su|sü|nda|nde)$", re.I)

def saglam_terim(q):
    t = q[:-7].strip() if q.endswith(" nedir?") else q
    if not (3 <= len(t) <= 45): return False
    if " " in t and len(t.split()) > 3: return False
    son = t.split()[-1]
    if EK.search(son) and len(son) > 6: return False      # cekim eki tasiyor
    if re.search(r"[0-9,;:()\[\]]", t): return False
    return True

once = len(terim_rows)
atilan = [r for r in terim_rows if not saglam_terim(r["question"])]
terim_rows = [r for r in terim_rows if saglam_terim(r["question"])]
print(f"terim: {once} -> {len(terim_rows)} (atilan {len(atilan)})")
print("\natilan ornekler:")
for r in atilan[:6]: print("  -", r["question"])
print("\nkalan ornekler:")
for r in terim_rows[:6]: print("  +", r["question"])

import random
egitim = tus_rows + sorgu_rows + terim_rows
random.Random(SEED).shuffle(egitim)
nv = max(1, int(len(egitim) * 0.02))
val_rows, train_rows = egitim[:nv], egitim[nv:]
print(f"\nTOPLAM egitim {len(train_rows)} | val {len(val_rows)}")
print(f"  tus {len(tus_rows)} (%{100*len(tus_rows)/len(egitim):.0f}) | "
      f"sorgu {len(sorgu_rows)} | terim {len(terim_rows)} "
      f"(karma %{100*(len(sorgu_rows)+len(terim_rows))/len(egitim):.0f})")

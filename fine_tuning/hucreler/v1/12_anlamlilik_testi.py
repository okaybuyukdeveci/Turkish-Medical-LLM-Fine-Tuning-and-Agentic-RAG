from scipy.stats import binomtest

b_ok = [parse_loose(o) == r["letter"] for o, r in zip(base["outs"], test_acc)]
f_ok = [parse_loose(o) == r["letter"] for o, r in zip(ft["outs"],   test_acc)]
sadece_ft   = sum(f and not b for b, f in zip(b_ok, f_ok))
sadece_taban= sum(b and not f for b, f in zip(b_ok, f_ok))
n = sadece_ft + sadece_taban
p = binomtest(sadece_ft, n, 0.5).pvalue if n else 1.0
print(f"yalniz ince ayar dogru: {sadece_ft} | yalniz taban dogru: {sadece_taban} | ayrisma {n}")
print(f"McNemar (exact) p = {p:.3f}" + ("  -> anlamli" if p < 0.05 else "  -> anlamli degil"))

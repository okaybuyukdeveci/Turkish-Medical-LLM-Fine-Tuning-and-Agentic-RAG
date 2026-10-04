uzun = evaluate(test_expl[:5], max_new_tokens=400, etiket="[aciklama] ")
for r, o in zip(test_expl[:3], uzun["outs"]):
    print("="*78)
    print("SORU   :", r["question"].split("\n")[0][:110])
    print("GOLD   :", r["letter"])
    print("MODEL  :", o.strip()[:450])
    print("KITABIN:", r["reference"][:280])

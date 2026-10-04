import random
egitim = tus_rows + sorgu_rows
random.Random(SEED).shuffle(egitim)
nv = max(1, int(len(egitim) * 0.02))
val_rows, train_rows = egitim[:nv], egitim[nv:]
print(f"TOPLAM egitim {len(train_rows)} | val {len(val_rows)}")
print(f"  tus {len(tus_rows)} | sorgu {len(sorgu_rows)} (karma %{100*len(sorgu_rows)/len(egitim):.0f})")
print("terim gorevi cikarildi: temel modelin urettigi tanimlar dogrulanmadi")

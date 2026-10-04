kayit = f"{OUT_DIR}/final_adapter"
model.save_pretrained(kayit)
tokenizer.save_pretrained(kayit)

sonuc = {
    "model": MODEL_NAME, "smoke": SMOKE, "epochs": EPOCHS, "seed": SEED,
    "train_size": len(train_ds), "val_size": len(val_ds),
    "test_size": len(test_acc),
    "baseline": {k: base[k] for k in ("strict","loose","format")},
    "finetuned": {k: ft[k]   for k in ("strict","loose","format")},
    "mcnemar": {"only_ft": sadece_ft, "only_base": sadece_taban, "p": p},
    "per_group_finetuned": {g: list(v) for g, v in ft["per_group"].items()},
}
with open(f"{OUT_DIR}/sonuc.json", "w", encoding="utf-8") as f:
    json.dump(sonuc, f, ensure_ascii=False, indent=2)

print("adapter:", kayit)
print(json.dumps({k: v for k, v in sonuc.items() if k != "per_group_finetuned"},
                 ensure_ascii=False, indent=2))

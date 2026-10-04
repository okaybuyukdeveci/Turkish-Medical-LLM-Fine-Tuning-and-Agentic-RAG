from datasets import Dataset

if not hasattr(model, "peft_config"):
    model = FastLanguageModel.get_peft_model(
        model, r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        use_gradient_checkpointing="unsloth", random_state=SEED,
    )
model.print_trainable_parameters()

def encode(row):
    p = tok(build_prompt(row["question"]), add_special_tokens=False)["input_ids"]
    c = tok(row["answer"] + END, add_special_tokens=False)["input_ids"]
    ids    = (p + c)[:MAX_SEQ_LEN]
    labels = ([-100] * len(p) + c)[:MAX_SEQ_LEN]
    return {"input_ids": ids, "labels": labels, "uzun": len(p) >= MAX_SEQ_LEN}

def to_ds(rows):
    ds = Dataset.from_list(rows).map(encode, remove_columns=["question","answer","letter","group"])
    atilan = sum(ds["uzun"])
    ds = ds.filter(lambda x: not x["uzun"]).remove_columns(["uzun"])
    return ds.map(lambda x: {"length": len(x["input_ids"])}), atilan

train_ds, a1 = to_ds(train_rows)
val_ds,   a2 = to_ds(val_rows)

lens = sorted(train_ds["length"])
maskesiz = sum(l != -100 for l in train_ds[0]["labels"]) / len(train_ds[0]["labels"])
print(f"train {len(train_ds)} | val {len(val_ds)} | promptu sigmayan atildi: {a1+a2}")
print(f"token p50 {lens[len(lens)//2]} | p95 {lens[int(len(lens)*.95)]} | max {lens[-1]}")
print(f"loss alan token orani (ilk ornek): {maskesiz:.0%}")

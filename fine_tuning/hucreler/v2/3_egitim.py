import inspect, os, torch
from datasets import Dataset
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

SMOKE = False
OUT_DIR = "/content/drive/MyDrive/tus_sft/qwen35_4b_v2" + ("_smoke" if SMOKE else "")
EPOCHS, SEED = 2, 42
SIS_TUS = ("Sen bir tıp uzmanısın. Aşağıdaki TUS sorusunu çöz. "
           "İlk satıra 'Doğru cevap: <harf>) <şık metni>' yaz, ardından kısa bir açıklama ver.")

gpu = torch.cuda.get_device_name(0)
BS, GA = (8, 2) if ("A100" in gpu or "H100" in gpu) else (4, 4)
print(f"{gpu} | batch {BS}x{GA}")

END = "<|im_end|>"
if not hasattr(model, "peft_config"):
    model = FastLanguageModel.get_peft_model(
        model, r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        use_gradient_checkpointing="unsloth", random_state=SEED)
model.print_trainable_parameters()

def encode(row):
    sis = row.get("sistem") or SIS_TUS
    p = tok(prompt(sis, row["question"]), add_special_tokens=False)["input_ids"]
    c = tok(row["answer"] + END, add_special_tokens=False)["input_ids"]
    return {"input_ids": (p+c)[:MAX_SEQ_LEN],
            "labels": ([-100]*len(p) + c)[:MAX_SEQ_LEN],
            "uzun": len(p) >= MAX_SEQ_LEN}

def to_ds(rows):
    ds = Dataset.from_list([{k: r.get(k, "") for k in ("gorev","sistem","question","answer")} for r in rows])
    ds = ds.map(encode, remove_columns=["gorev","sistem","question","answer"])
    ds = ds.filter(lambda x: not x["uzun"]).remove_columns(["uzun"])
    return ds.map(lambda x: {"length": len(x["input_ids"])})

train_ds, val_ds = to_ds(train_rows), to_ds(val_rows)
lens = sorted(train_ds["length"])
print(f"train {len(train_ds)} | val {len(val_ds)} | token p50 {lens[len(lens)//2]} p95 {lens[int(len(lens)*.95)]}")

def collate(batch):
    n = max(len(b["input_ids"]) for b in batch)
    pad = lambda s, v: s + [v]*(n-len(s))
    return {"input_ids":      torch.tensor([pad(b["input_ids"], tok.pad_token_id) for b in batch]),
            "attention_mask": torch.tensor([pad([1]*len(b["input_ids"]), 0)       for b in batch]),
            "labels":         torch.tensor([pad(b["labels"], -100)                for b in batch])}

adim = 20 if SMOKE else 200
_k = set(inspect.signature(TrainingArguments.__init__).parameters)
_ek = {}
if "group_by_length" in _k: _ek["group_by_length"] = True
elif "train_sampling_strategy" in _k: _ek["train_sampling_strategy"] = "group_by_length"
if "warmup_ratio" in _k: _ek["warmup_ratio"] = 0.03
_ek["eval_strategy" if "eval_strategy" in _k else "evaluation_strategy"] = "steps"

args = TrainingArguments(
    output_dir=OUT_DIR, per_device_train_batch_size=BS, per_device_eval_batch_size=BS,
    gradient_accumulation_steps=GA, num_train_epochs=EPOCHS, max_steps=40 if SMOKE else -1,
    learning_rate=1e-4, lr_scheduler_type="cosine", weight_decay=0.0,
    bf16=True, optim="adamw_8bit", logging_steps=10, eval_steps=adim,
    save_strategy="steps", save_steps=adim, save_total_limit=2,
    load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
    remove_unused_columns=False, report_to="none", seed=SEED, **_ek)

FastLanguageModel.for_training(model)
tok.padding_side = "right"
trainer = Trainer(model=model, args=args, train_dataset=train_ds,
                  eval_dataset=val_ds, data_collator=collate)
son = get_last_checkpoint(OUT_DIR) if os.path.isdir(OUT_DIR) else None
print("checkpoint:", son)
trainer.train(resume_from_checkpoint=son)
model.save_pretrained(f"{OUT_DIR}/final_adapter")
tokenizer.save_pretrained(f"{OUT_DIR}/final_adapter")
print("kaydedildi:", f"{OUT_DIR}/final_adapter")

from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

def collate(batch):
    n = max(len(b["input_ids"]) for b in batch)
    pad = lambda s, v: s + [v] * (n - len(s))
    return {
        "input_ids":      torch.tensor([pad(b["input_ids"], tok.pad_token_id) for b in batch]),
        "attention_mask": torch.tensor([pad([1]*len(b["input_ids"]), 0)       for b in batch]),
        "labels":         torch.tensor([pad(b["labels"], -100)                for b in batch]),
    }

adim = 20 if SMOKE else 200

import inspect
_kabul = set(inspect.signature(TrainingArguments.__init__).parameters)
_ek = {}
if "group_by_length" in _kabul:
    _ek["group_by_length"] = True
elif "train_sampling_strategy" in _kabul:
    _ek["train_sampling_strategy"] = "group_by_length"
if "warmup_ratio" in _kabul:
    _ek["warmup_ratio"] = 0.03
elif "warmup_steps" in _kabul:
    _ek["warmup_steps"] = 30
_eval_anahtar = "eval_strategy" if "eval_strategy" in _kabul else "evaluation_strategy"
_ek[_eval_anahtar] = "steps"
print("transformers uyumu:", {k: v for k, v in _ek.items()})

args = TrainingArguments(
    output_dir=OUT_DIR,
    per_device_train_batch_size=BS,
    per_device_eval_batch_size=BS,
    gradient_accumulation_steps=GA,
    num_train_epochs=EPOCHS,
    max_steps=40 if SMOKE else -1,
    learning_rate=1e-4,
    lr_scheduler_type="cosine",
    weight_decay=0.0,
    bf16=not LOAD_4BIT, fp16=LOAD_4BIT,
    optim="adamw_8bit",
    logging_steps=10,
    eval_steps=adim,
    save_strategy="steps", save_steps=adim, save_total_limit=2,
    load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
    remove_unused_columns=False, report_to="none", seed=SEED,
    **_ek,
)

FastLanguageModel.for_training(model)
tok.padding_side = "right"
trainer = Trainer(model=model, args=args, train_dataset=train_ds,
                  eval_dataset=val_ds, data_collator=collate)

son = get_last_checkpoint(OUT_DIR) if os.path.isdir(OUT_DIR) else None
print("devam edilen checkpoint:", son)
trainer.train(resume_from_checkpoint=son)

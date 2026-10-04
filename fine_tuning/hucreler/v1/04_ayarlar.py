SMOKE = True

MODEL_NAME = "Qwen/Qwen3.5-4B"
OUT_DIR    = "/content/drive/MyDrive/tus_sft/qwen35_4b" + ("_smoke" if SMOKE else "")

MAX_SEQ_LEN    = 2048
MAX_EXPL_CHARS = 3000
EPOCHS         = 1 if SMOKE else 2
SEED           = 42

SYSTEM_PROMPT = (
    "Sen bir tıp uzmanısın. Aşağıdaki TUS sorusunu çöz. "
    "İlk satıra 'Doğru cevap: <harf>) <şık metni>' yaz, ardından kısa bir açıklama ver."
)

import torch, os
gpu = torch.cuda.get_device_name(0)
if   "A100" in gpu or "H100" in gpu: BS, GA, LOAD_4BIT = 8, 2, False
elif "L4"   in gpu or "A10"  in gpu: BS, GA, LOAD_4BIT = 4, 4, False
else:                                BS, GA, LOAD_4BIT = 2, 8, True
print(f"{gpu} | batch {BS} x grad_accum {GA} = {BS*GA} | 4bit={LOAD_4BIT}")

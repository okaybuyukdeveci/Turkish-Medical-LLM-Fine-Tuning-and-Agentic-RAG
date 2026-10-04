from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    MODEL_NAME,
    max_seq_length=MAX_SEQ_LEN,
    dtype=torch.bfloat16,
    load_in_4bit=LOAD_4BIT,
)

tok = getattr(tokenizer, "tokenizer", tokenizer)
END = "<|im_end|>"
assert tok.convert_tokens_to_ids(END) != tok.unk_token_id, "Qwen <|im_end|> tokeni bulunamadi"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

def build_prompt(question):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": question}]
    return tok.apply_chat_template(msgs, tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)

ornek = build_prompt("Soru?")
print(ornek)
assert "<think>" in ornek, "no-think sablonunda bos think blogu bekleniyordu"

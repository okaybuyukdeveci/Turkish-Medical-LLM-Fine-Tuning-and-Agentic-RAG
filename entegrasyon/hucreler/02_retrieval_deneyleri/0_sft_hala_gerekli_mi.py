@torch.no_grad()
def uret_taban(prompts, max_new_tokens=64, bs=8, etiket=""):
    tok.padding_side = "left"
    outs = []
    with model.disable_adapter():
        for i in range(0, len(prompts), bs):
            b = prompts[i:i+bs]
            enc = tok(b, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
            g = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                               pad_token_id=tok.pad_token_id)
            outs += tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            print(f"  {etiket}{min(i+bs,len(prompts))}/{len(prompts)}", end="\r")
    print()
    return outs

print("ham model + RAG (adapter kapali, ayni baglam)")
hamrag_outs = uret_taban(B_promptlar, 64, bs=8, etiket="[ham+RAG] ")
hamrag_acc, hamrag_pred = dogruluk(hamrag_outs, test_rows, "[ham+RAG] ")

print("ham model, RAG yok")
hamsade_outs = uret_taban([prompt(SYSTEM_SFT, r["question"]) for r in test_rows],
                          64, bs=16, etiket="[ham] ")
hamsade_acc, _ = dogruluk(hamsade_outs, test_rows, "[ham] ")

from math import comb
a = sum(x != r["letter"] and y == r["letter"] for x, y, r in zip(hamrag_pred, B_pred, test_rows))
b = sum(x == r["letter"] and y != r["letter"] for x, y, r in zip(hamrag_pred, B_pred, test_rows))
n = a + b
p = (sum(comb(n, k) for k in range(max(a, b), n + 1)) / 2**n * 2) if n else 1.0

print(f"""
{'='*58}
                     RAG YOK      RAG VAR     SFT KATKISI
ham model            {hamsade_acc:>6.1%}      {hamrag_acc:>6.1%}
SFT'li model         {sade_acc:>6.1%}      {B_acc:>6.1%}
                   {sade_acc-hamsade_acc:>+7.1%}    {B_acc-hamrag_acc:>+7.1%}
{'='*58}
RAG varken SFT: lehte {a} soru, aleyhte {b}, p = {min(p,1.0):.3f}
{'  -> SFT hala katki sagliyor' if p < 0.05 and B_acc > hamrag_acc else '  -> SFT katkisi RAG tarafindan ikame edilmis'}""")

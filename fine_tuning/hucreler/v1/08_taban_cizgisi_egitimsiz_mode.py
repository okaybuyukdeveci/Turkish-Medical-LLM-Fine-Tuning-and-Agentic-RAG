base = evaluate(test_acc, max_new_tokens=512, etiket="[taban] ")
print("\n--- ornek cikti ---")
print(base["outs"][0][:600])

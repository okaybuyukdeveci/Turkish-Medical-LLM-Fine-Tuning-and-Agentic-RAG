ft = evaluate(test_acc, max_new_tokens=64, etiket="[sonra] ")

print(f"""
{'='*58}
                    TABAN      SONRA     FARK
dogruluk (gevsek)   {base['loose']:>6.1%}   {ft['loose']:>6.1%}   {ft['loose']-base['loose']:+.1%}
dogruluk (kati)     {base['strict']:>6.1%}   {ft['strict']:>6.1%}   {ft['strict']-base['strict']:+.1%}
bicime uyum         {base['format']:>6.1%}   {ft['format']:>6.1%}   {ft['format']-base['format']:+.1%}
{'='*58}
{len(test_acc)} soru. Gevsek satir asil bilgi kazancini, bicim satiri
bicim ogrenmesini gosterir. Ikisi birlikte okunmali.""")

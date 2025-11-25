import sys, os, binascii
path = r'c:\Users\yueny\OneDrive\Documents\Netvlad_3001\pytorch-NetVlad\runs\Nov25_14-46-48_vgg16_netvlad\checkpoints\checkpoint.pth.tar'
if not os.path.exists(path):
    print('MISSING', path); sys.exit(1)
with open(path,'rb') as f:
    b = f.read()
keys = [b'epoch', b'best_score', b'recalls', b'optimizer', b'recalls', b'best_score']
for k in keys:
    i = b.find(k)
    print('---', k.decode('utf-8', errors='replace'), 'pos=', i)
    if i != -1:
        s = max(0, i-200)
        e = min(len(b), i+200)
        snippet = b[s:e]
        print('ASCII near match:')
        print(snippet.decode('utf-8', errors='replace'))
        print('HEX snippet (first 200 chars):')
        print(binascii.hexlify(snippet)[:200])

# Try to heuristically find an integer "epoch" by searching for the ASCII 'epoch' then scanning forward for digits
import re
m = re.search(b'epoch.*?(\d+)', b, flags=re.IGNORECASE|re.DOTALL)
if m:
    try:
        print('Heuristic epoch:', m.group(1).decode())
    except:
        print('Heuristic epoch (bytes):', m.group(1))
else:
    print('No heuristic epoch found')

# Also try to find occurrences of "best_score" with a float-looking pattern
m2 = re.search(b'best_score.*?([0-9]+\.[0-9]+)', b, flags=re.IGNORECASE|re.DOTALL)
if m2:
    print('Heuristic best_score:', m2.group(1).decode())
else:
    print('No heuristic best_score found')

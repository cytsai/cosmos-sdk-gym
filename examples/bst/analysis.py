import sys
import re
from collections import defaultdict, Counter

stats = defaultdict(Counter)
count = [0] * 31
with open(sys.argv[1]) as f:
    for l in f:        
        t = re.sub(r'\d+', 'x', l.strip())
        stats[t][l.strip()] += 1
        count[t.count('x')] += 1

for k in sorted(stats, key=len):
    print(k, stats[k])
print(count)

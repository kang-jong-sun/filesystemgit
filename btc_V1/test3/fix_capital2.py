# -*- coding: utf-8 -*-
import os, re

SPEC_DIR = 'D:/filesystem/futures/btc_V1/test3'
files = sorted([f for f in os.listdir(SPEC_DIR) if f.startswith('# BTC') and f.endswith('.md')])
count = 0

for f in files:
    fp = os.path.join(SPEC_DIR, f)
    with open(fp, 'r', encoding='utf-8') as fh:
        c = fh.read()
    orig = c

    # $3,000 -> $5,000 in capital context
    c = re.sub(r'초기 자본\s*\|\s*\$3,000', '초기 자본 | $5,000', c)
    c = re.sub(r'초기자본\s*\|\s*\$3,000', '초기자본 | $5,000', c)
    c = re.sub(r'\*\*초기 자본\*\*\s*\|\s*\$3,000', '**초기 자본** | $5,000', c)
    c = re.sub(r'초기 자본\s*\|\s*\*\*\$3,000\*\*', '초기 자본 | **$5,000**', c)
    c = re.sub(r'\$3,000 초기자본', '$5,000 초기자본', c)
    c = re.sub(r'\$3,000 USDT\*\*', '$5,000 USDT**', c)
    c = re.sub(r'초기 자본\s*\(전체\)\s*\|\s*\$3,000', '초기 자본 (전체) | $5,000', c)

    # initial_capital/initial_balance = 3000
    c = re.sub(r'"initial_capital":\s*3000', '"initial_capital": 5000', c)
    c = re.sub(r'"initial_balance":\s*3000', '"initial_balance": 5000', c)

    # **초기 자본**: $3,000
    c = re.sub(r'\*\*초기 자본\*\*:\s*\$3,000', '**초기 자본**: $5,000', c)

    if c != orig:
        with open(fp, 'w', encoding='utf-8') as fh:
            fh.write(c)
        count += 1
        print(f'  FIXED: {f[:50]}')

print(f'\n{count}/46 files modified')

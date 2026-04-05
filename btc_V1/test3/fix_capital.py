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

    # INITIAL_CAPITAL = xxxx
    c = re.sub(r'(INITIAL_CAPITAL\s*=\s*)[\d.]+', r'\g<1>5000.0', c)
    # initial_balance = xxxx
    c = re.sub(r'(initial_balance\s*=\s*)[\d.]+', r'\g<1>5000', c)
    # cap = xxxx  # 잔액
    c = re.sub(r'(cap\s*=\s*)[\d.]+(\s*#?\s*잔액)', r'\g<1>5000.0\2', c)
    # 초기 자본: $1,000 or $5,000 etc
    c = re.sub(r'초기 자[본금][^|]*?[\$]\s*[\d,]+', '초기 자본: $5,000', c)
    # **초기 자본**: $xxx
    c = re.sub(r'\*\*초기 자[본금]\*\*[^|]*?[\$]\s*[\d,]+', '**초기 자본**: $5,000', c)
    # | 초기자본 | $xxx |
    c = re.sub(r'(초기\s*자[본금]\s*\|\s*)\$?[\d,]+', r'\g<1>$5,000', c)

    if c != orig:
        with open(fp, 'w', encoding='utf-8') as fh:
            fh.write(c)
        count += 1

print(f'{count}/46 files modified')

# Verify
print('\nVerification:')
for f in files:
    fp = os.path.join(SPEC_DIR, f)
    with open(fp, 'r', encoding='utf-8') as fh:
        c = fh.read()
    # Check for non-5000 capital values
    caps = re.findall(r'INITIAL_CAPITAL\s*=\s*([\d.]+)', c)
    caps2 = re.findall(r'initial_balance\s*=\s*([\d.]+)', c)
    all_caps = caps + caps2
    bad = [x for x in all_caps if x not in ('5000.0', '5000')]
    if bad:
        print(f'  WARN {f[:45]}: {bad}')

print('Done')

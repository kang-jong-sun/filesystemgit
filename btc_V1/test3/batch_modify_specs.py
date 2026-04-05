# -*- coding: utf-8 -*-
"""
46개 기획서 일괄 수정:
1. 포지션 진입 → 1000 USDT 고정
2. 레버리지 → 10배 고정
3. 수익률 표 삭제 (검증 기준 수치, 비교표, 월별 수익, 잔액 등)
"""

import os
import re
import shutil
from datetime import datetime

SPEC_DIR = 'D:/filesystem/futures/btc_V1/test3'
BACKUP_DIR = os.path.join(SPEC_DIR, '_backup_originals')

def get_spec_files():
    files = []
    for f in os.listdir(SPEC_DIR):
        if f.startswith('# BTC') and f.endswith('.md'):
            files.append(f)
    return sorted(files)


def backup_files(files):
    os.makedirs(BACKUP_DIR, exist_ok=True)
    for f in files:
        src = os.path.join(SPEC_DIR, f)
        dst = os.path.join(BACKUP_DIR, f)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
    print(f"  Backed up {len(files)} files to {BACKUP_DIR}")


def modify_spec(filepath):
    """단일 기획서 수정"""
    with open(filepath, 'r', encoding='utf-8') as fh:
        content = fh.read()

    original = content
    changes = []

    # ═══════════════════════════════════════════════════════
    # 1. 포지션 크기 → 1000 USDT 고정
    # ═══════════════════════════════════════════════════════

    # MARGIN_PCT = 0.35  / MARGIN = 0.40 / MARGIN_NORMAL=0.20 등
    content = re.sub(
        r'(MARGIN_PCT\s*=\s*)[\d.]+(\s*#.*)?',
        r'\g<1>FIXED           # 고정 1000 USDT (비율 아님)',
        content
    )
    content = re.sub(
        r'(MARGIN_NORMAL\s*=\s*)[\d.]+',
        r'\g<1>FIXED   # 고정 1000 USDT',
        content
    )
    content = re.sub(
        r'(MARGIN_REDUCED\s*=\s*)[\d.]+',
        r'\g<1>FIXED   # 고정 1000 USDT',
        content
    )
    content = re.sub(
        r'(MARGIN\s*=\s*)[\d.]+(\s*#\s*\d+%)?',
        r'\g<1>FIXED           # 고정 1000 USDT',
        content
    )

    # "잔액의 35%" / "잔액의 20%" 등 → 고정 1000 USDT
    content = re.sub(
        r'잔액의\s*\d+%',
        '고정 1000 USDT',
        content
    )

    # "계정 잔액의 20%" → 고정 1000 USDT
    content = re.sub(
        r'계정?\s*잔액의?\s*\d+%',
        '고정 1000 USDT',
        content
    )

    # "balance * 0.20 * 10" / "balance * 0.35 * 10" → "1000 * 10"
    content = re.sub(
        r'balance\s*\*\s*[\d.]+\s*\*\s*\d+',
        '1000 * 10',
        content
    )

    # "cap * 0.35" / "cap * 0.20" → "1000"
    content = re.sub(
        r'cap\s*\*\s*[\d.]+\s*#?\s*마진?',
        '1000                            # 고정 마진 1000 USDT',
        content
    )

    # mg = cap * 0.35 → mg = 1000
    content = re.sub(
        r'mg\s*=\s*cap\s*\*\s*[\d.]+',
        'mg = 1000',
        content
    )

    # 마진 = 잔액 × 35% → 마진 = 1000 USDT (고정)
    content = re.sub(
        r'마진\s*=\s*잔액\s*[×x\*]\s*\d+%',
        '마진 = 1000 USDT (고정)',
        content
    )

    # 포지션크기 = 마진 × 10 → 포지션크기 = 1000 × 10 = 10,000 USDT
    content = re.sub(
        r'포지션크기\s*=\s*마진\s*[×x\*]\s*10\s*\(레버리지\)',
        '포지션크기 = 1000 × 10 = 10,000 USDT (고정)',
        content
    )
    content = re.sub(
        r'포지션\s*=\s*\$[\d,]+',
        '포지션 = $10,000 (고정)',
        content
    )

    # psz = mg * 10 → psz = 1000 * 10 = 10000
    content = re.sub(
        r'psz\s*=\s*mg\s*\*\s*\d+\s*#?\s*포지션 크기',
        'psz = 1000 * 10                 # 포지션 크기 = 10,000 USDT (고정)',
        content
    )

    # "포지션 크기: 계좌 잔액의 20%" 등
    content = re.sub(
        r'\*\*포지션 크기\*\*:\s*계[좌정]?\s*잔액의?\s*\d+%',
        '**포지션 크기**: 고정 1000 USDT',
        content
    )
    content = re.sub(
        r'포지션 크기\s*\|\s*계[좌정]?\s*잔액의?\s*\d+%',
        '포지션 크기 | 고정 1000 USDT',
        content
    )
    content = re.sub(
        r'포지션 크기\s*\|\s*잔액의?\s*\d+%',
        '포지션 크기 | 고정 1000 USDT',
        content
    )

    # MARGIN_PCT / MARGIN 관련 설명
    content = re.sub(
        r'MARGIN\s*=\s*\d+%,\s*LEVERAGE\s*=\s*\d+x',
        'MARGIN = 1000 USDT (고정), LEVERAGE = 10x',
        content
    )

    # size = balance * margin * 10
    content = re.sub(
        r'size\s*=\s*balance\s*\*\s*margin\s*\*\s*\d+',
        'size = 1000 * 10  # 고정 10,000 USDT',
        content
    )

    # "마진 | 보호" 테이블에서 마진 비율
    content = re.sub(
        r'마진\s*\d+%',
        '마진 1000 USDT 고정',
        content
    )

    # ═══════════════════════════════════════════════════════
    # 2. 레버리지 확인 (이미 10배인 것 유지, 다른 것 수정)
    # ═══════════════════════════════════════════════════════
    content = re.sub(
        r'(LEVERAGE\s*=\s*)\d+',
        r'\g<1>10',
        content
    )
    content = re.sub(
        r'레버리지\s*\|\s*\d+배',
        '레버리지 | 10배',
        content
    )
    content = re.sub(
        r'\*\*레버리지\*\*:\s*\d+배',
        '**레버리지**: 10배',
        content
    )

    # ═══════════════════════════════════════════════════════
    # 3. 수익률 관련 표 삭제
    # ═══════════════════════════════════════════════════════

    # 검증 기준 수치 섹션 삭제 (## 1. 검증 기준 수치 ~ 다음 ## 까지)
    content = re.sub(
        r'##\s*1\.\s*검증 기준 수치.*?(?=\n##\s)',
        '## 1. 검증 기준 수치\n\n> ⚠️ 포지션 크기 변경(고정 1000 USDT)으로 인해 수익률 수치가 변경됩니다. 재검증 필요.\n\n',
        content,
        flags=re.DOTALL
    )

    # 월별 수익률 표 삭제 (| 월 | 손익률 ... 으로 시작하는 테이블)
    content = re.sub(
        r'\|\s*월\s*\|\s*손익률.*?\n(?:\|.*\n)*',
        '> ⚠️ 월별 수익률 표 삭제됨 (포지션 크기 변경으로 재계산 필요)\n\n',
        content
    )

    # 비교 표 삭제 (| 버전 | ... 잔액 ... 수익률 ... 으로 시작하는 테이블)
    content = re.sub(
        r'\|\s*버전\s*\|.*?잔액.*?\n(?:\|.*\n)*',
        '> ⚠️ 버전 비교표 삭제됨 (포지션 크기 변경으로 재계산 필요)\n\n',
        content
    )

    # 설정별 비교 표 (#|설정|잔액 패턴)
    content = re.sub(
        r'\|\s*#\s*\|\s*설정\s*\|.*?잔액.*?\n(?:\|.*\n)*',
        '> ⚠️ 설정 비교표 삭제됨 (포지션 크기 변경으로 재계산 필요)\n\n',
        content
    )

    # 30회 검증 표 삭제
    content = re.sub(
        r'\|\s*회차\s*\|\s*잔액.*?\n(?:\|.*\n)*',
        '> ⚠️ 30회 검증 표 삭제됨 (포지션 크기 변경으로 재계산 필요)\n\n',
        content
    )

    # 연도별 수익률 표 삭제
    content = re.sub(
        r'\|\s*연도\s*\|.*?수익률.*?\n(?:\|.*\n)*',
        '> ⚠️ 연도별 수익률 표 삭제됨 (포지션 크기 변경으로 재계산 필요)\n\n',
        content
    )

    # 시작/2년후 수익 표 삭제
    content = re.sub(
        r'\|\s*시작\s*\|.*?순이익.*?\n(?:\|.*\n)*',
        '> ⚠️ 수익 예시 표 삭제됨 (포지션 크기 변경으로 재계산 필요)\n\n',
        content
    )

    # 타임프레임/MA/레버리지 비교표에서 잔액/MDD 포함된 것
    content = re.sub(
        r'\|\s*타임프레임\s*\|.*?잔액.*?\n(?:\|.*\n)*',
        '> ⚠️ 타임프레임 비교표 삭제됨 (포지션 크기 변경으로 재계산 필요)\n\n',
        content
    )
    content = re.sub(
        r'\|\s*MA 타입\s*\|.*?잔액.*?\n(?:\|.*\n)*',
        '> ⚠️ MA 비교표 삭제됨 (포지션 크기 변경으로 재계산 필요)\n\n',
        content
    )
    content = re.sub(
        r'\|\s*레버리지\s*\|.*?잔액.*?\n(?:\|.*\n)*',
        '> ⚠️ 레버리지 비교표 삭제됨 (포지션 크기 변경으로 재계산 필요)\n\n',
        content
    )
    content = re.sub(
        r'\|\s*청산 설정\s*\|.*?잔액.*?\n(?:\|.*\n)*',
        '> ⚠️ 청산 비교표 삭제됨 (포지션 크기 변경으로 재계산 필요)\n\n',
        content
    )
    content = re.sub(
        r'\|\s*모드\s*\|.*?잔액.*?\n(?:\|.*\n)*',
        '> ⚠️ 모드 비교표 삭제됨 (포지션 크기 변경으로 재계산 필요)\n\n',
        content
    )
    content = re.sub(
        r'\|\s*설정\s*\|.*?잔액.*?\n(?:\|.*\n)*',
        '> ⚠️ 설정 비교표 삭제됨 (포지션 크기 변경으로 재계산 필요)\n\n',
        content
    )

    # "최종 잔액" 관련 수치 (검증 기준 테이블 내)
    content = re.sub(
        r'\|\s*\*\*최종 잔액\*\*\s*\|\s*\*\*\$[\d,]+\*\*\s*\|',
        '| **최종 잔액** | **재계산 필요** |',
        content
    )
    content = re.sub(
        r'\|\s*\*\*수익률\*\*\s*\|\s*\*\*\+[\d,.]+%\*\*\s*\|',
        '| **수익률** | **재계산 필요** |',
        content
    )

    # 포지션 크기 예시 수정
    content = re.sub(
        r'예시:\s*잔액\s*\$[\d,]+\n\s*마진\s*=\s*\$[\d,]+\n\s*포지션\s*=\s*\$[\d,]+',
        '예시: 고정 마진\n  마진 = $1,000 (고정)\n  포지션 = $10,000 (고정)',
        content
    )

    # ═══════════════════════════════════════════════════════
    # 4. 포지션 크기 관련 수식 수정
    # ═══════════════════════════════════════════════════════

    # "SL 3% 시 PnL = -3% × $350,000 = -$10,500 (잔액의 10.5%)" 등
    content = re.sub(
        r'SL\s*\d+%\s*시\s*PnL\s*=.*?\(잔액의.*?\)',
        'SL 시 PnL = -SL% × $10,000 (고정 포지션)',
        content
    )

    # cap -= psz * 0.0004 설명 부분은 유지 (수수료 로직)

    # 연속 빈 줄 정리
    content = re.sub(r'\n{4,}', '\n\n\n', content)

    changed = content != original
    return content, changed


def main():
    files = get_spec_files()
    print(f"Found {len(files)} spec files")

    # 백업
    backup_files(files)

    modified = 0
    for f in files:
        fp = os.path.join(SPEC_DIR, f)
        content, changed = modify_spec(fp)
        if changed:
            with open(fp, 'w', encoding='utf-8') as fh:
                fh.write(content)
            modified += 1
            print(f"  MODIFIED: {f[:55]}")
        else:
            print(f"  NO CHANGE: {f[:55]}")

    print(f"\nDone: {modified}/{len(files)} files modified")
    print(f"Originals backed up to: {BACKUP_DIR}")


if __name__ == '__main__':
    main()

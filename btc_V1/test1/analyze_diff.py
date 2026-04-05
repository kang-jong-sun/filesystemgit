import sys; sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd

df = pd.read_csv('ai_comparison_trades.csv')
mdf = pd.read_csv('ai_comparison_monthly.csv')

print("=" * 80)
print("  AI 백테스트 상세 분석 리포트")
print("=" * 80)

# ── 1. 전체 Action 분포 ──
print("\n[1] Action 분포")
print(f"  {'봇':>8} | {'LONG':>5} | {'SHORT':>5} | {'HOLD':>5} | {'합계':>5}")
print("  " + "-" * 45)
for name, col in [('코드봇', 'code_action'), ('GPT-4o', 'gpt_action'), ('Claude', 'claude_action')]:
    vc = df[col].value_counts()
    l = vc.get('LONG', 0); s = vc.get('SHORT', 0); h = vc.get('HOLD', 0)
    print(f"  {name:>8} | {l:>5} | {s:>5} | {h:>5} | {l+s+h:>5}")

# ── 2. 불일치 건 분류 ──
hold_all = df[(df['code_action']=='HOLD') & (df['gpt_action']=='HOLD') & (df['claude_action']=='HOLD')]
gpt_only = df[(df['gpt_action'] != df['code_action']) & (df['claude_action'] == df['code_action'])]
cla_only = df[(df['claude_action'] != df['code_action']) & (df['gpt_action'] == df['code_action'])]
both_diff = df[(df['gpt_action'] != df['code_action']) & (df['claude_action'] != df['code_action']) & ~((df['code_action']=='HOLD') & (df['gpt_action']=='HOLD') & (df['claude_action']=='HOLD'))]

print(f"\n[2] 불일치 분류 (총 {len(df)}회 크로스)")
print(f"  3봇 일치:        {len(df) - len(gpt_only) - len(cla_only) - len(both_diff):>3}건")
print(f"  3봇 모두 HOLD:   {len(hold_all):>3}건 (월간손실한도)")
print(f"  GPT만 다름:      {len(gpt_only):>3}건")
print(f"  Claude만 다름:   {len(cla_only):>3}건")
print(f"  둘 다 다름:      {len(both_diff):>3}건")

# ── 3. GPT만 HOLD한 경우 상세 ──
print(f"\n[3] GPT만 HOLD한 경우 ({len(gpt_only)}건)")
print("  " + "-" * 75)
for _, r in gpt_only.iterrows():
    print(f"  {r['timestamp'][:16]} | {r['cross']} | ADX={r['adx']:.1f} RSI={r['rsi']:.1f}")
    print(f"    코드: {r['code_action']} | GPT: {r['gpt_action']} | Claude: {r['claude_action']}")
    print(f"    GPT 이유: {r['gpt_reason']}")
    print(f"    잔액 → 코드:${r['code_bal']:,.0f}  GPT:${r['gpt_bal']:,.0f}  Claude:${r['claude_bal']:,.0f}")
    print()

# ── 4. Claude만 HOLD한 경우 상세 ──
print(f"\n[4] Claude만 HOLD한 경우 ({len(cla_only)}건)")
print("  " + "-" * 75)
for _, r in cla_only.iterrows():
    print(f"  {r['timestamp'][:16]} | {r['cross']} | ADX={r['adx']:.1f} RSI={r['rsi']:.1f}")
    print(f"    코드: {r['code_action']} | GPT: {r['gpt_action']} | Claude: {r['claude_action']}")
    print(f"    Claude 이유: {r['claude_reason']}")
    print(f"    잔액 → 코드:${r['code_bal']:,.0f}  GPT:${r['gpt_bal']:,.0f}  Claude:${r['claude_bal']:,.0f}")
    print()

# ── 5. 둘 다 다른 경우 상세 ──
print(f"\n[5] AI 둘 다 코드와 다른 경우 ({len(both_diff)}건)")
print("  " + "-" * 75)
for _, r in both_diff.iterrows():
    print(f"  {r['timestamp'][:16]} | {r['cross']} | ADX={r['adx']:.1f} RSI={r['rsi']:.1f}")
    print(f"    코드: {r['code_action']} | GPT: {r['gpt_action']} | Claude: {r['claude_action']}")
    print(f"    GPT 이유: {r['gpt_reason']}")
    print(f"    Claude 이유: {r['claude_reason']}")
    print(f"    잔액 → 코드:${r['code_bal']:,.0f}  GPT:${r['gpt_bal']:,.0f}  Claude:${r['claude_bal']:,.0f}")
    print()

# ── 6. 잔액 분기점 분석 ──
print("\n[6] 잔액 분기점 분석 (불일치 이후 잔액 변화)")
print("  " + "-" * 75)
diff_rows = df[(df['code_action'] != df['gpt_action']) | (df['code_action'] != df['claude_action'])]
for idx, (_, r) in enumerate(diff_rows.iterrows()):
    c_b = r['code_bal']; g_b = r['gpt_bal']; cl_b = r['claude_bal']
    g_diff = g_b - c_b; cl_diff = cl_b - c_b
    print(f"  {r['timestamp'][:16]} | 코드:{r['code_action']:>5} GPT:{r['gpt_action']:>5} Claude:{r['claude_action']:>6}")
    print(f"    잔액차이 → GPT-코드: ${g_diff:+,.0f}  Claude-코드: ${cl_diff:+,.0f}")
    # 다음 크로스까지의 잔액 변화 보기
    cur_idx = df[df['timestamp'] == r['timestamp']].index[0]
    if cur_idx + 1 < len(df):
        nr = df.iloc[cur_idx + 1]
        print(f"    다음크로스 {nr['timestamp'][:16]} → 코드:${nr['code_bal']:,.0f}  GPT:${nr['gpt_bal']:,.0f}  Claude:${nr['claude_bal']:,.0f}")
        ng = nr['gpt_bal'] - nr['code_bal']; nc = nr['claude_bal'] - nr['code_bal']
        print(f"    잔액차이변화 → GPT-코드: ${ng:+,.0f}  Claude-코드: ${nc:+,.0f}")
    print()

# ── 7. GPT vs Claude 직접 비교 ──
print("\n[7] GPT vs Claude 판단 차이")
gpt_cla_diff = df[df['gpt_action'] != df['claude_action']]
print(f"  GPT와 Claude가 서로 다른 판단: {len(gpt_cla_diff)}건")
print("  " + "-" * 75)
for _, r in gpt_cla_diff.iterrows():
    print(f"  {r['timestamp'][:16]} | {r['cross']} | ADX={r['adx']:.1f} RSI={r['rsi']:.1f}")
    print(f"    GPT: {r['gpt_action']:>5} | Claude: {r['claude_action']:>6} | 코드: {r['code_action']:>5}")
    print(f"    GPT이유: {r['gpt_reason'][:60]}")
    print(f"    Claude이유: {r['claude_reason'][:60]}")
    winner = ""
    # 다음 크로스에서 잔액 비교
    cur_idx = df[df['timestamp'] == r['timestamp']].index[0]
    if cur_idx + 1 < len(df):
        nr = df.iloc[cur_idx + 1]
        gd = nr['gpt_bal'] - r['gpt_bal']; cd = nr['claude_bal'] - r['claude_bal']
        if gd > cd: winner = "GPT 승"
        elif cd > gd: winner = "Claude 승"
        else: winner = "무승부"
        print(f"    결과 → GPT PnL:${gd:+,.0f}  Claude PnL:${cd:+,.0f}  → {winner}")
    print()

# ── 8. 월별 잔액 차이 추세 ──
print("\n[8] 월별 잔액 차이 추세 (Claude-코드, GPT-코드)")
print(f"  {'월':>7} | {'코드':>10} | {'GPT':>10} | {'Claude':>10} | {'GPT-코드':>10} | {'Claude-코드':>12} | {'Claude-GPT':>11}")
print("  " + "-" * 85)
for _, r in mdf.iterrows():
    cb = r['code_bal']; gb = r['gpt_bal']; clb = r['claude_bal']
    if cb > 0 or gb > 0:
        gd = gb - cb; cd = clb - cb; cg = clb - gb
        print(f"  {r['month']:>7} | ${cb:>8,.0f} | ${gb:>8,.0f} | ${clb:>8,.0f} | ${gd:>+9,.0f} | ${cd:>+10,.0f} | ${cg:>+9,.0f}")

# ── 9. 거래수가 비슷한 이유 분석 ──
print("\n\n[9] 거래수 비슷한 이유 분석")
print("=" * 75)
c_trades = len(df[df['code_action'].isin(['LONG','SHORT'])])
g_trades = len(df[df['gpt_action'].isin(['LONG','SHORT'])])
cl_trades = len(df[df['claude_action'].isin(['LONG','SHORT'])])
print(f"  크로스에서의 진입 시도: 코드={c_trades}  GPT={g_trades}  Claude={cl_trades}")
print(f"  코드 HOLD: {len(df[df['code_action']=='HOLD'])}건 (월간손실한도)")
print(f"  GPT HOLD:  {len(df[df['gpt_action']=='HOLD'])}건")
print(f"  Claude HOLD: {len(df[df['claude_action']=='HOLD'])}건")

# AI가 HOLD한 이유 분류
gpt_holds = df[df['gpt_action'] == 'HOLD']
cla_holds = df[df['claude_action'] == 'HOLD']
print(f"\n  GPT HOLD 이유 분류 ({len(gpt_holds)}건):")
for _, r in gpt_holds.iterrows():
    reason = r['gpt_reason'][:50]
    print(f"    {r['timestamp'][:16]}: {reason}")

print(f"\n  Claude HOLD 이유 분류 ({len(cla_holds)}건):")
for _, r in cla_holds.iterrows():
    reason = r['claude_reason'][:50]
    print(f"    {r['timestamp'][:16]}: {reason}")

# ── 10. 최종 요약 ──
print("\n\n" + "=" * 75)
print("  최종 요약")
print("=" * 75)
final_c = df.iloc[-1]['code_bal']
final_g = df.iloc[-1]['gpt_bal']
final_cl = df.iloc[-1]['claude_bal']
print(f"\n  최종잔액: 코드=${final_c:,.0f}  GPT=${final_g:,.0f}  Claude=${final_cl:,.0f}")
print(f"  Claude vs 코드: +${final_cl-final_c:,.0f} ({(final_cl-final_c)/final_c*100:+.1f}%)")
print(f"  GPT vs 코드:    +${final_g-final_c:,.0f} ({(final_g-final_c)/final_c*100:+.1f}%)")
print(f"  Claude vs GPT:  +${final_cl-final_g:,.0f} ({(final_cl-final_g)/final_g*100:+.1f}%)")

"""
Cross Margin vs Isolated Margin Comparison
- Same strategy, different margin modes
- Impact on liquidation, SL, position sizing, compounding
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def main():
    print("=" * 100)
    print("  CROSS MARGIN vs ISOLATED MARGIN ANALYSIS")
    print("  BTC/USDT Futures | Binance")
    print("=" * 100)

    # ============================================================
    # 1. CORE MECHANICS COMPARISON
    # ============================================================
    print(f"""
  1. CORE MECHANICS
  {'='*80}

  [ISOLATED MARGIN]
  - Position margin = allocated amount only
  - Liquidation: only position margin is lost
  - Account balance is PROTECTED from liquidation
  - Each position has independent liquidation price

  [CROSS MARGIN]
  - Position margin = ENTIRE account balance
  - Liquidation: ENTIRE account balance is lost
  - All positions share the same margin pool
  - Liquidation price is much further away (more buffer)
  """)

    # ============================================================
    # 2. LIQUIDATION PRICE COMPARISON
    # ============================================================
    print(f"  2. LIQUIDATION PRICE COMPARISON")
    print(f"  {'='*80}")
    print(f"  Account: $3,000 | BTC: $80,000 | Leverage: 10x\n")

    scenarios = [
        ("ISOLATED M20%", 0.20, True),
        ("ISOLATED M40%", 0.40, True),
        ("ISOLATED M60%", 0.60, True),
        ("CROSS (100%)", 1.00, False),
    ]

    print(f"  {'Mode':<20} {'Margin':>8} {'Notional':>10} {'Liq Price(L)':>14} {'Liq Move%':>10} {'SL -8% Loss':>12} {'Survives?':>10}")
    print(f"  {'-'*90}")

    balance = 3000
    btc = 80000
    lev = 10

    for name, ratio, isolated in scenarios:
        margin = balance * ratio
        notional = margin * lev
        qty = notional / btc

        if isolated:
            # Isolated: liq when margin is consumed
            # Approx: liq_price = entry * (1 - 1/lev) for long
            liq_move = 1.0 / lev  # ~10% for 10x
            liq_price_long = btc * (1 - liq_move)
            sl_loss = margin * 0.08 * lev  # SL at -8%
            survives = "YES (protected)"
        else:
            # Cross: liq when ENTIRE balance is consumed
            # liq_price = entry - (balance / qty) approximately
            liq_price_long = btc - (balance / qty)
            liq_move = (btc - liq_price_long) / btc
            sl_loss = notional * 0.08  # SL at -8%
            survives = "NO (all lost)" if sl_loss > balance else "YES"

        sl_loss_pct = sl_loss / balance * 100
        print(f"  {name:<20} ${margin:>6,.0f} ${notional:>8,.0f}  ${liq_price_long:>11,.0f}  {liq_move*100:>8.1f}%  ${sl_loss:>10,.0f}({sl_loss_pct:.0f}%) {survives}")

    # ============================================================
    # 3. REAL SCENARIO SIMULATION
    # ============================================================
    print(f"\n  3. CONSECUTIVE SL HITS SIMULATION")
    print(f"  {'='*80}")
    print(f"  Strategy: SL -8%, Leverage 10x, Starting $3,000\n")

    for mode, ratio in [("ISOLATED M20%", 0.20), ("ISOLATED M40%", 0.40), ("ISOLATED M60%", 0.60), ("CROSS", 1.00)]:
        bal = 3000.0
        print(f"  [{mode}]")
        for hit in range(1, 8):
            margin = bal * ratio
            notional = margin * lev
            loss = notional * 0.08 + notional * 0.0008  # SL + fees
            if mode == "CROSS":
                loss = min(loss, bal)  # cross can't lose more than balance
            bal -= loss
            if bal <= 0:
                print(f"    SL#{hit}: LIQUIDATED! Balance = $0")
                break
            print(f"    SL#{hit}: -{loss:>7,.0f} -> ${bal:>8,.1f} ({(bal/3000-1)*100:>+.1f}%)")
        print()

    # ============================================================
    # 4. CROSS MARGIN ADVANTAGES
    # ============================================================
    print(f"  4. CROSS MARGIN ADVANTAGES")
    print(f"  {'='*80}")
    print(f"""
  [1] Wider liquidation buffer
      - ISOLATED M40% Lev10x: liquidation at -10% (very tight)
      - CROSS Lev10x: liquidation at ~-{100/lev:.0f}% equivalent
        (but uses FULL balance as collateral)

  [2] No margin selection needed
      - Don't need to decide 20%/40%/60% allocation
      - System automatically uses optimal margin

  [3] Multiple positions share margin
      - Engine A loss can be offset by Engine B's unrealized profit
      - More efficient capital utilization

  [4] Lower liquidation risk per trade
      - Same SL -8% with 10x: loss = 80% of ALLOCATED margin
      - ISOLATED M40%: 80% of 40% = 32% of balance
      - CROSS: 80% of effective margin = 8% of balance (if position is small)
  """)

    # ============================================================
    # 5. CROSS MARGIN DANGERS
    # ============================================================
    print(f"  5. CROSS MARGIN DANGERS")
    print(f"  {'='*80}")
    print(f"""
  [1] CATASTROPHIC RISK: Flash crash without SL
      - ISOLATED: lose allocated margin only ($600-$1,800)
      - CROSS: lose ENTIRE balance ($3,000)
      - If SL fails (exchange lag, extreme volatility): total wipeout

  [2] Multiple positions amplify risk
      - 2 LONG positions in CROSS mode:
        Combined exposure = 2x, shared margin pool
        Single crash destroys both + entire account

  [3] No isolation between strategies
      - Engine A's bad trade drains margin from Engine B
      - Can trigger cascading liquidation

  [4] Psychological trap: false sense of security
      - Wider liquidation buffer encourages larger positions
      - "SL will protect me" -> but SL can fail (slippage, gap)
  """)

    # ============================================================
    # 6. OPTIMAL CROSS MARGIN STRATEGY
    # ============================================================
    print(f"  6. IF USING CROSS MARGIN - OPTIMAL APPROACH")
    print(f"  {'='*80}")
    print(f"""
  [Rule 1] SMALL POSITION SIZE (key!)
      - Don't use 40-60% of balance as margin
      - Use 10-15% equivalent position size
      - Cross margin benefit: no liquidation even on -10% move
      - SL -8% loss = only 8-12% of balance

  [Rule 2] ALWAYS USE SL (server-side)
      - Cross margin without SL = gambling
      - SL must be on Binance server (Stop Market order)
      - Never rely on bot for SL (bot crash = total loss)

  [Rule 3] ONE POSITION AT A TIME
      - Multiple cross positions = multiplied risk
      - If running dual engine: use ISOLATED for each

  [Rule 4] MAX LEVERAGE = 5-7x in CROSS mode
      - Higher leverage = smaller move to liquidation
      - Even in cross mode, 20x leverage is dangerous
  """)

    # ============================================================
    # 7. CROSS MARGIN SCALPING ANALYSIS
    # ============================================================
    print(f"  7. CROSS MARGIN FOR SCALPING")
    print(f"  {'='*80}")
    print(f"""
  Q: Does cross margin fix the 1-minute scalping problem?
  A: NO. The core problem is FEE DRAG, not margin mode.

  Fee calculation is IDENTICAL in both modes:
    Taker fee = 0.04% x notional (regardless of margin mode)

  Cross margin changes:
    - Liquidation distance (wider buffer) -> GOOD
    - Risk per trade (same SL = same loss) -> NO CHANGE
    - Fee per trade -> NO CHANGE
    - Required win rate -> NO CHANGE

  The only benefit for scalping:
    - Can use TIGHTER SL (e.g., -0.5%) without liquidation
    - This reduces loss per trade
    - But fee still eats 27-53% of micro-profits

  VERDICT: Cross margin is NOT the solution for 1m scalping.
           Fee reduction (maker orders, BNB discount) is the real fix.
  """)

    # ============================================================
    # 8. RECOMMENDED MARGIN MODE BY STRATEGY
    # ============================================================
    print(f"  8. RECOMMENDATION BY STRATEGY TYPE")
    print(f"  {'='*80}")

    recs = [
        ("1m Scalping (TP<1%)", "ISOLATED M10-15%", "Loss must be capped per trade, fee is real problem"),
        ("5m Short Swing", "ISOLATED M20-30%", "Moderate risk, need isolation"),
        ("30m Trend (v22.1)", "ISOLATED M40-50%", "Proven strategy, isolation protects capital"),
        ("Multi-Engine Dual", "ISOLATED per engine", "Each engine needs independent margin"),
        ("Single Engine Conservative", "CROSS + small size", "OK if position size <15% and strict SL"),
        ("High Leverage (15-20x)", "ISOLATED only", "NEVER use cross with high leverage"),
    ]

    print(f"  {'Strategy':<30} {'Margin Mode':<25} {'Reason'}")
    print(f"  {'-'*95}")
    for strat, mode, reason in recs:
        print(f"  {strat:<30} {mode:<25} {reason}")

    print(f"\n{'='*100}")
    print(f"  FINAL VERDICT")
    print(f"{'='*100}")
    print(f"""
  CROSS MARGIN is acceptable ONLY when:
    1. Position size <= 15% of balance
    2. Leverage <= 7x
    3. Server-side SL is ALWAYS set
    4. Single position at a time
    5. Strategy has proven SL hit rate < 5%

  For the v22.1 strategy (best verified):
    - ISOLATED M50% Lev10x is the tested configuration
    - CROSS with 15% position would give same exposure with wider liq buffer
    - But changes the risk profile from verified backtest

  RECOMMENDATION: Stay with ISOLATED margin.
    - Backtested and verified in isolated mode
    - Cross margin adds unquantified tail risk
    - The marginal benefit (wider liq buffer) is not worth the catastrophic risk
  """)

    print("=" * 100)

if __name__ == "__main__":
    main()

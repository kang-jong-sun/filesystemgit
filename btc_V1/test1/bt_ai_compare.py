"""
AI 비교 백테스트 v14.4
- 크로스 발생 시점마다 Claude API에 지표/캔들 데이터를 보내 진입/스킵 판단
- 코드봇 vs AI 비교 리포트 생성
- 기획서 Section 19 방식 재현

사용법:
  python bt_ai_compare.py                    # 코드봇만 (AI 호출 없이 시뮬레이션)
  python bt_ai_compare.py --ai               # AI 비교 실행 (ANTHROPIC_API_KEY 필요)
  python bt_ai_compare.py --ai --dry-run     # AI 호출 시뮬레이션 (비용 없음)
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

from bt_engine_v144 import (
    CONFIG, prepare_data, BacktestEngine, Position, print_report
)


# ============================================================
# AI 판단 모듈
# ============================================================
class AIJudge:
    """크로스 시점에서 AI 판단을 받는 모듈"""

    SYSTEM_PROMPT = """You are a BTC/USDT futures trading bot assistant.
You will receive market data at an EMA crossover point. Your job is to decide whether to ENTER or HOLD.

Rules:
- Strategy: 30m EMA(3/200) cross + ADX(14) >= 35 + RSI(14) 30~65
- If conditions are met, decide ENTER or HOLD
- If you already have a position in the SAME direction as the signal, you may HOLD to keep the existing position (avoid unnecessary re-entry)
- If you have a position in the OPPOSITE direction, always ENTER (reverse)
- If no position, and conditions are met, ENTER

Respond in JSON format only:
{"decision": "ENTER" or "HOLD", "reason": "brief reason in English"}"""

    def __init__(self, api_key: str = None, model: str = "claude-sonnet-4-6-20250514", dry_run: bool = False):
        self.api_key = api_key or os.environ.get('ANTHROPIC_API_KEY', '')
        self.model = model
        self.dry_run = dry_run
        self.call_count = 0
        self.decisions: List[Dict] = []

        if not dry_run and not self.api_key:
            print("[WARNING] ANTHROPIC_API_KEY not set. Use --dry-run for simulation.")

    def judge(self, cross_event: Dict, recent_candles: pd.DataFrame) -> Dict:
        """
        크로스 시점에서 AI 판단

        Args:
            cross_event: 크로스 이벤트 정보
            recent_candles: 최근 20봉 캔들 데이터

        Returns:
            {'decision': 'ENTER'|'HOLD', 'reason': str}
        """
        self.call_count += 1

        if self.dry_run:
            return self._simulate_judge(cross_event)

        return self._call_api(cross_event, recent_candles)

    def _simulate_judge(self, event: Dict) -> Dict:
        """AI 호출 시뮬레이션 (Claude 로직 재현: 동일방향이면 HOLD)"""
        current_pos = event.get('current_position')
        signal = event['signal']

        if current_pos == signal:
            decision = {
                'decision': 'HOLD',
                'reason': f'Already holding {signal}, skip re-entry'
            }
        else:
            decision = {
                'decision': 'ENTER',
                'reason': f'Conditions met for {signal}'
            }

        self.decisions.append({**event, **decision})
        return decision

    def _call_api(self, event: Dict, recent_candles: pd.DataFrame) -> Dict:
        """실제 Claude API 호출"""
        try:
            import anthropic

            client = anthropic.Anthropic(api_key=self.api_key)

            # 최근 캔들 데이터 포맷팅
            candle_text = recent_candles[['timestamp', 'open', 'high', 'low', 'close', 'volume',
                                          'ema_fast', 'ema_slow', 'adx', 'rsi']].tail(20).to_string(index=False)

            user_msg = f"""Cross Event at {event['timestamp']}:
- Cross Type: {event['cross_type']} ({"Golden" if event['cross_type'] == 'golden' else "Dead"} Cross)
- Signal: {event['signal']}
- Close: ${event['close']:,.2f}
- EMA(3): ${event['ema_fast']:,.2f}
- EMA(200): ${event['ema_slow']:,.2f}
- ADX(14): {event['adx']:.1f}
- RSI(14): {event['rsi']:.1f}
- Current Position: {event['current_position'] or 'None'}

Recent 20 candles (30m):
{candle_text}

Decision (JSON only):"""

            response = client.messages.create(
                model=self.model,
                max_tokens=200,
                temperature=0,
                system=self.SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_msg}]
            )

            # JSON 파싱
            text = response.content[0].text.strip()
            # JSON 블록 추출
            if '```' in text:
                text = text.split('```')[1].replace('json', '').strip()

            result = json.loads(text)
            decision = {
                'decision': result.get('decision', 'ENTER'),
                'reason': result.get('reason', 'No reason provided')
            }

        except Exception as e:
            print(f"  [AI ERROR] {e} -> defaulting to ENTER")
            decision = {
                'decision': 'ENTER',
                'reason': f'API error: {str(e)[:50]}'
            }

        self.decisions.append({**event, **decision})
        return decision


# ============================================================
# AI 비교 백테스트 엔진
# ============================================================
class AICompareEngine(BacktestEngine):
    """AI 판단을 적용하는 백테스트 엔진"""

    def __init__(self, ai_judge: AIJudge, config: dict = None):
        super().__init__(config)
        self.ai_judge = ai_judge
        self.ai_decisions: List[Dict] = []

    def run_with_ai(self, df: pd.DataFrame) -> Dict:
        """AI 판단을 적용한 백테스트"""
        self.balance = self.cfg['INITIAL_BALANCE']
        self.position = None
        self.trades = []
        self.equity_curve = []
        self.peak_balance = self.cfg['INITIAL_BALANCE']
        self.max_drawdown = 0.0
        self.current_month = None
        self.monthly_locked = False
        self.ai_decisions = []

        cross_events = []
        total_crosses = 0

        for i in range(len(df)):
            row = df.iloc[i]
            ts = row['timestamp']
            close = row['close']
            high = row['high']
            low = row['low']

            self._check_monthly_reset(ts)

            # ---- 포지션 보유 중 처리 ----
            if self.position is not None:
                sl_result = self._check_sl(high, low)
                if sl_result:
                    self._close_position(sl_result[0], sl_result[1], ts, i)
                    self._update_equity(ts)
                else:
                    self._update_trailing(high, low, close)
                    tsl_result = self._check_trail_sl_close(close)
                    if tsl_result:
                        self._close_position(tsl_result[0], tsl_result[1], ts, i)
                        self._update_equity(ts)

            # ---- 신호 감지 ----
            cross_up = row.get('cross_up', False)
            cross_dn = row.get('cross_dn', False)
            adx = row.get('adx', 0)
            rsi = row.get('rsi', 50)

            if cross_up or cross_dn:
                signal = None
                if cross_up and adx >= self.cfg['ADX_MIN'] and self.cfg['RSI_MIN'] <= rsi <= self.cfg['RSI_MAX']:
                    signal = 'LONG'
                elif cross_dn and adx >= self.cfg['ADX_MIN'] and self.cfg['RSI_MIN'] <= rsi <= self.cfg['RSI_MAX']:
                    signal = 'SHORT'

                if signal:
                    total_crosses += 1
                    cross_event = {
                        'idx': i,
                        'timestamp': ts,
                        'cross_type': 'golden' if cross_up else 'dead',
                        'signal': signal,
                        'close': close,
                        'adx': adx,
                        'rsi': rsi,
                        'ema_fast': row['ema_fast'],
                        'ema_slow': row['ema_slow'],
                        'current_position': self.position.direction if self.position else None,
                    }
                    cross_events.append(cross_event)

                    # AI 판단 요청
                    recent = df.iloc[max(0, i-20):i+1]
                    ai_result = self.ai_judge.judge(cross_event, recent)
                    ai_decision = ai_result['decision']

                    self.ai_decisions.append({
                        **cross_event,
                        'ai_decision': ai_decision,
                        'ai_reason': ai_result['reason'],
                    })

                    print(f"  [{total_crosses:>3}] {ts} | {cross_event['cross_type']:>6} | "
                          f"Signal={signal:>5} | Pos={cross_event['current_position'] or 'None':>5} | "
                          f"AI={ai_decision:>5} | ADX={adx:.1f} RSI={rsi:.1f}")

                    if ai_decision == 'HOLD':
                        # AI가 HOLD → 기존 포지션 유지, 신규 진입 스킵
                        self._update_equity(ts)
                        continue

                    # AI가 ENTER → 코드봇 로직 실행
                    if self.position is not None:
                        if self.position.direction != signal:
                            self._close_position(close, 'REV', ts, i)
                        else:
                            # 동일방향인데 AI가 ENTER → 재진입
                            self._close_position(close, 'REV', ts, i)

                    if self.position is None and not self._is_monthly_locked():
                        self._open_position(signal, close, ts, i)

            self._update_equity(ts)

        # 미청산 포지션
        if self.position is not None:
            last_close = df.iloc[-1]['close']
            last_ts = df.iloc[-1]['timestamp']
            self._close_position(last_close, 'END', last_ts, len(df)-1)

        result = self._compile_results(cross_events)
        result['ai_decisions'] = self.ai_decisions
        result['total_crosses'] = total_crosses
        return result


# ============================================================
# 비교 리포트
# ============================================================
def print_comparison(code_result: Dict, ai_result: Dict):
    """코드봇 vs AI 비교 리포트"""
    print(f"\n{'='*70}")
    print(f"  CODE BOT vs AI COMPARE")
    print(f"{'='*70}")
    print(f"  {'Item':<18} {'Code Bot':>15} {'AI Bot':>15} {'Diff':>12}")
    print(f"  {'-'*63}")
    print(f"  {'Balance':<18} ${code_result['balance']:>13,.0f} ${ai_result['balance']:>13,.0f} ${ai_result['balance']-code_result['balance']:>+10,.0f}")
    print(f"  {'Return':<18} {code_result['return_pct']:>+14.1f}% {ai_result['return_pct']:>+14.1f}% {ai_result['return_pct']-code_result['return_pct']:>+11.1f}%p")
    print(f"  {'PF':<18} {code_result['pf']:>15.2f} {ai_result['pf']:>15.2f} {ai_result['pf']-code_result['pf']:>+12.2f}")
    print(f"  {'MDD':<18} {code_result['mdd']:>14.1f}% {ai_result['mdd']:>14.1f}% {ai_result['mdd']-code_result['mdd']:>+11.1f}%p")
    print(f"  {'Trades':<18} {code_result['trades']:>15} {ai_result['trades']:>15} {ai_result['trades']-code_result['trades']:>+12}")
    print(f"  {'Win Rate':<18} {code_result['win_rate']:>14.1f}% {ai_result['win_rate']:>14.1f}%")
    print(f"  {'Payoff Ratio':<18} {code_result.get('payoff_ratio',0):>14.2f}:1 {ai_result.get('payoff_ratio',0):>14.2f}:1")
    print(f"  {'SL':<18} {code_result.get('sl_count',0):>15} {ai_result.get('sl_count',0):>15}")
    print(f"  {'TSL':<18} {code_result.get('tsl_count',0):>15} {ai_result.get('tsl_count',0):>15}")
    print(f"  {'REV':<18} {code_result.get('rev_count',0):>15} {ai_result.get('rev_count',0):>15}")

    # AI 독자 판단 상세
    if ai_result.get('ai_decisions'):
        holds = [d for d in ai_result['ai_decisions'] if d['ai_decision'] == 'HOLD']
        if holds:
            print(f"\n  [AI HOLD Decisions: {len(holds)}]")
            print(f"  {'#':<3} {'Date':<20} {'Cross':<8} {'Signal':<6} {'Position':<8} {'Reason'}")
            print(f"  {'-'*80}")
            for j, h in enumerate(holds, 1):
                print(f"  {j:<3} {str(h['timestamp']):<20} {h['cross_type']:<8} {h['signal']:<6} "
                      f"{h.get('current_position','None'):<8} {h['ai_reason']}")

    # 연도별 비교
    if code_result.get('yearly_stats') and ai_result.get('yearly_stats'):
        all_years = sorted(set(list(code_result['yearly_stats'].keys()) + list(ai_result['yearly_stats'].keys())))
        print(f"\n  [Yearly Comparison]")
        print(f"  {'Year':<6} {'Code Bot':>12} {'AI Bot':>12} {'Diff':>12}")
        print(f"  {'-'*45}")
        for y in all_years:
            cb = code_result['yearly_stats'].get(y, {}).get('end_balance', 0)
            ab = ai_result['yearly_stats'].get(y, {}).get('end_balance', 0)
            print(f"  {y:<6} ${cb:>10,.0f} ${ab:>10,.0f} ${ab-cb:>+10,.0f}")


# ============================================================
# 메인
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='BTC v14.4 AI Compare Backtest')
    parser.add_argument('--ai', action='store_true', help='Enable AI comparison')
    parser.add_argument('--dry-run', action='store_true', help='Simulate AI calls (no API cost)')
    parser.add_argument('--model', default='claude-sonnet-4-6-20250514', help='Claude model')
    parser.add_argument('--data', default=r'D:\filesystem\futures\btc_V1\test1\btc_usdt_5m_merged.csv')
    args = parser.parse_args()

    print("=" * 70)
    print("  BTC/USDT v14.4 AI Compare Backtest")
    print("=" * 70)

    start = time.time()

    # 데이터 준비
    df = prepare_data(args.data)

    # 1. 코드봇 (기본)
    print("\n[1/3] Code Bot (basic)...")
    engine_code = BacktestEngine()
    result_code = engine_code.run(df, skip_same_direction=False)
    print_report(result_code, "Code Bot v14.4")

    # 2. 코드봇 (동일방향 스킵 = Claude 로직 시뮬레이션)
    print("[2/3] Code Bot (skip same direction = Claude logic)...")
    engine_skip = BacktestEngine()
    result_skip = engine_skip.run(df, skip_same_direction=True)
    print_report(result_skip, "Code Bot v14.4 (Same Dir Skip)")

    # 3. AI 비교 (선택적)
    if args.ai:
        print(f"[3/3] AI Bot ({args.model}, dry_run={args.dry_run})...")
        ai_judge = AIJudge(
            model=args.model,
            dry_run=args.dry_run,
        )
        engine_ai = AICompareEngine(ai_judge)
        result_ai = engine_ai.run_with_ai(df)
        print_report(result_ai, f"AI Bot ({args.model})")

        # 비교
        print_comparison(result_code, result_ai)

        # AI 판단 저장
        if ai_judge.decisions:
            output_file = 'ai_decisions.json'
            with open(output_file, 'w', encoding='utf-8') as f:
                # timestamp를 문자열로 변환
                serializable = []
                for d in ai_judge.decisions:
                    sd = {k: (str(v) if isinstance(v, pd.Timestamp) else v) for k, v in d.items()}
                    serializable.append(sd)
                json.dump(serializable, f, indent=2, ensure_ascii=False)
            print(f"\n  AI decisions saved to {output_file}")

    else:
        # AI 없이 코드봇 기본 vs 동일방향 스킵 비교
        print_comparison(result_code, result_skip)

    elapsed = time.time() - start
    print(f"\nTotal time: {elapsed:.1f}s")
    print(f"API calls: {0 if not args.ai else ai_judge.call_count}")


if __name__ == '__main__':
    main()

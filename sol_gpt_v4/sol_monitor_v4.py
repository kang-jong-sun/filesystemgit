#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📺 SOL Futures 자동매매 시스템 v4.0 - Monitor
10초 간격 실시간 모니터링
"""

import asyncio
import time
from datetime import datetime
from typing import Dict, Optional
import logging
from pathlib import Path
import sys
from io import StringIO

class Monitor:
    """실시간 모니터링"""
    
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        
        # 컴포넌트 참조
        self.bot = None
        self.data_collector = None
        self.order_executor = None
        self.telegram = None
        
        # 모니터링 데이터
        self.monitoring_data = {
            'system_status': 'INITIALIZING',
            'current_price': 0,
            'ema_status': {},
            'position': None,
            'stats': {},
            'last_update': None
        }
        
        # 다중 타임프레임 정렬 추적
        self.last_alignment_state = {
            'bullish_dominant': False,
            'bearish_dominant': False,
            'alert_sent_at': None
        }
        
        # 태스크
        self.monitoring_task = None
        
        # 모니터링 로그 파일 설정
        self.setup_monitoring_log()
        
        self.logger.info("📺 Monitor 초기화")
    
    def set_components(self, bot, data_collector, order_executor, telegram=None):
        """컴포넌트 설정"""
        self.bot = bot
        self.data_collector = data_collector
        self.order_executor = order_executor
        self.telegram = telegram
    
    def setup_monitoring_log(self):
        """모니터링 로그 파일 설정 (12시간 간격)"""
        # 로그 디렉토리 생성
        log_dir = Path("monitoring_logs")
        log_dir.mkdir(exist_ok=True)
        
        # 12시간 간격 파일명 (00-11, 12-23)
        now = datetime.now()
        hour_block = "00-11" if now.hour < 12 else "12-23"
        log_file = log_dir / f"monitoring_{now.strftime('%Y%m%d')}_{hour_block}.log"
        
        # 모니터링 로거 설정
        self.monitoring_logger = logging.getLogger("MonitoringLog")
        self.monitoring_logger.setLevel(logging.INFO)
        
        # 기존 핸들러 제거
        self.monitoring_logger.handlers.clear()
        
        # 파일 핸들러
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # 포맷 설정 (타임스탬프 없이)
        formatter = logging.Formatter('%(message)s')
        file_handler.setFormatter(formatter)
        
        self.monitoring_logger.addHandler(file_handler)
        self.monitoring_logger.propagate = False  # 상위 로거로 전파 방지
        
        self.current_log_hour = now.hour
        self.logger.info(f"📝 모니터링 로그 파일: {log_file}")
    
    async def start(self):
        """모니터링 시작"""
        try:
            self.monitoring_data['system_status'] = 'RUNNING'
            
            # 모니터링 태스크 시작
            self.monitoring_task = asyncio.create_task(self._run_monitoring())
            
            self.logger.info("✅ 모니터링 시작")
            
        except Exception as e:
            self.logger.error(f"모니터링 시작 오류: {str(e)}")
            self.monitoring_data['system_status'] = 'ERROR'
    
    async def _run_monitoring(self):
        """모니터링 실행"""
        while True:
            try:
                # 12시간 경과 체크
                current_hour = datetime.now().hour
                if (self.current_log_hour < 12 and current_hour >= 12) or \
                   (self.current_log_hour >= 12 and current_hour < 12):
                    self.setup_monitoring_log()  # 새 로그 파일 생성
                
                # 데이터 업데이트
                await self._update_data()
                
                # 상태 출력
                self._print_status()
                
                # 30초 대기
                await asyncio.sleep(self.config.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"모니터링 오류: {str(e)}")
                await asyncio.sleep(5)
    
    async def check_multi_tf_alignment(self):
        """다중 타임프레임 정렬 체크 및 알림"""
        try:
            multi_tf_ema = self.monitoring_data.get('multi_tf_ema', {})
            ema = self.monitoring_data.get('ema_status', {})
            
            if not multi_tf_ema:
                return
            
            # 각 타임프레임 정렬 상태 수집
            alignment_details = {}
            bullish_count = 0
            bearish_count = 0
            
            for tf in ['1m', '3m', '5m', '15m', '30m']:
                # 30분봉은 메인 EMA 사용
                if tf == '30m':
                    tf_data = ema if ema else {}
                else:
                    tf_data = multi_tf_ema.get(tf, {})
                
                if tf_data and tf_data.get('ema10'):
                    ema10 = tf_data.get('ema10', 0)
                    ema21 = tf_data.get('ema21', 0)
                    ema21_high = tf_data.get('ema21_high', 0)
                    ema21_low = tf_data.get('ema21_low', 0)
                    
                    is_bullish = ema10 > ema21
                    
                    # 터치 조건 체크
                    diff_high = abs(ema10 - ema21_high) / ema21_high * 100 if ema21_high > 0 else 100
                    diff_low = abs(ema10 - ema21_low) / ema21_low * 100 if ema21_low > 0 else 100
                    touch_high = diff_high <= 0.2 and is_bullish
                    touch_low = diff_low <= 0.2 and not is_bullish
                    
                    alignment_details[tf] = {
                        'is_bullish': is_bullish,
                        'touch_high': touch_high,
                        'touch_low': touch_low,
                        'ema10': ema10,
                        'ema21': ema21
                    }
                    
                    if is_bullish:
                        bullish_count += 1
                    else:
                        bearish_count += 1
            
            # 정렬 알림 조건 체크 (5개 모두 동일)
            current_time = datetime.now()
            should_alert = False
            
            if bullish_count == 5:
                # 이전 상태가 정배열 우세가 아니었거나, 마지막 알림으로부터 30분 경과
                if not self.last_alignment_state['bullish_dominant'] or \
                   (self.last_alignment_state['alert_sent_at'] and 
                    (current_time - self.last_alignment_state['alert_sent_at']).total_seconds() > 1800):
                    should_alert = True
                    self.last_alignment_state['bullish_dominant'] = True
                    self.last_alignment_state['bearish_dominant'] = False
                    
            elif bearish_count == 5:
                # 이전 상태가 역배열 우세가 아니었거나, 마지막 알림으로부터 30분 경과
                if not self.last_alignment_state['bearish_dominant'] or \
                   (self.last_alignment_state['alert_sent_at'] and 
                    (current_time - self.last_alignment_state['alert_sent_at']).total_seconds() > 1800):
                    should_alert = True
                    self.last_alignment_state['bearish_dominant'] = True
                    self.last_alignment_state['bullish_dominant'] = False
            else:
                # 정렬 해제
                self.last_alignment_state['bullish_dominant'] = False
                self.last_alignment_state['bearish_dominant'] = False
            
            # 알림 전송
            if should_alert and self.telegram:
                alignment_data = {
                    'bullish_count': bullish_count,
                    'bearish_count': bearish_count,
                    'details': alignment_details
                }
                await self.telegram.send_multi_tf_alignment_alert(alignment_data)
                self.last_alignment_state['alert_sent_at'] = current_time
                
        except Exception as e:
            self.logger.error(f"다중 타임프레임 정렬 체크 오류: {e}")
    
    async def _update_data(self):
        """모니터링 데이터 업데이트"""
        try:
            # 현재가 및 EMA 상태
            if self.data_collector:
                try:
                    ticker = await self.data_collector.fetch_ticker()
                    self.monitoring_data['current_price'] = ticker.get('last', 0) if ticker else 0
                    self.monitoring_data['ema_status'] = self.data_collector.get_current_ema() or {}
                    
                    # 다중 타임프레임 EMA 데이터 가져오기
                    self.monitoring_data['multi_tf_ema'] = self.data_collector.get_multi_tf_ema() or {}
                    
                    # 다중 타임프레임 데이터 업데이트 (10초마다)
                    await self.data_collector.update_multi_tf_candles()
                    
                    # 다중 타임프레임 정렬 체크
                    await self.check_multi_tf_alignment()
                    
                except Exception as e:
                    self.logger.error(f"데이터 수집 오류: {e}")
                    self.monitoring_data['current_price'] = 0
                    self.monitoring_data['ema_status'] = {}
                    self.monitoring_data['multi_tf_ema'] = {}
            
            # 포지션 정보
            if self.order_executor:
                if self.order_executor.has_position():
                    # 최신 포지션 정보로 업데이트
                    current_pos = self.order_executor.current_position
                    
                    # API에서 최신 정보 가져오기 (10초마다)
                    try:
                        positions = await self.order_executor.get_open_positions()
                        for pos in positions:
                            # 심볼 매칭 개선 (SOL/USDT:USDT 형식 처리)
                            normalized_api_symbol = pos['symbol'].replace('/', '').replace(':USDT', '')
                            normalized_current_symbol = current_pos['symbol'].replace('/', '').replace(':USDT', '')
                            
                            if normalized_api_symbol == normalized_current_symbol:
                                # 최신 정보로 업데이트 (partial_closed는 유지)
                                partial_closed = current_pos.get('partial_closed', False)
                                current_pos.update({
                                    'mark_price': pos.get('mark_price', self.monitoring_data['current_price']),
                                    'unrealized_pnl': pos.get('unrealized_pnl', 0),
                                    'percentage': pos.get('percentage'),  # Binance ROI 값
                                    'margin': pos.get('margin', 0),
                                    'liquidation_price': pos.get('liquidation_price', 0),
                                    'position_size_usdt': pos.get('position_size_usdt', current_pos['quantity'] * current_pos['entry_price']),
                                    'partial_closed': partial_closed  # partial_closed 상태 유지
                                })
                                break
                    except Exception as e:
                        self.logger.error(f"포지션 업데이트 오류: {str(e)}")
                    
                    self.monitoring_data['position'] = current_pos
                    # 디버깅: position 데이터 확인
                    self.logger.info(f"[모니터링 업데이트] position partial_closed: {current_pos.get('partial_closed')}")
                else:
                    self.monitoring_data['position'] = None
                
                # 통계
                self.monitoring_data['stats'] = self.order_executor.stats
            
            self.monitoring_data['last_update'] = datetime.now()
            
        except Exception as e:
            self.logger.error(f"데이터 업데이트 오류: {str(e)}")
    
    def _print_status(self):
        """상태 출력"""
        try:
            # StringIO를 사용하여 모든 출력을 캡처
            buffer = StringIO()
            
            # 원래 stdout 저장
            old_stdout = sys.stdout
            sys.stdout = buffer
            
            # 이제 모든 print는 buffer로 들어감
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            price = self.monitoring_data.get('current_price', 0.0)
            ema = self.monitoring_data.get('ema_status', {})
            
            # 헤더
            print(f"\n{'='*80}")
            print(f" SOL Futures 자동매매 v4.0 - Quadruple EMA Touch Strategy | {now}")
            print(f"{'='*80}")
            
            # 시장 상태
            print(f"\n 시장 정보:")
            if price and price > 0:
                print(f"  • 현재가: ${price:,.2f}")
            else:
                print(f"  • 현재가: 데이터 없음")
            # 타임프레임 매핑
            timeframe_map = {
                '1m': '1분봉',
                '3m': '3분봉', 
                '5m': '5분봉',
                '15m': '15분봉',
                '30m': '30분봉',
                '1h': '1시간봉',
                '4h': '4시간봉',
                '1d': '일봉'
            }
            tf_display = timeframe_map.get(self.config.timeframe, self.config.timeframe)
            print(f"  • 타임프레임: {tf_display}")
            
            # 다중 타임프레임 EMA 표시
            multi_tf_ema = self.monitoring_data.get('multi_tf_ema', {})
            if multi_tf_ema:
                print(f"\n 📊 다중 타임프레임 EMA 상태:")
                print(f" {'─'*76}")
                
                for tf in ['1m', '3m', '5m', '15m', '30m']:
                    tf_data = multi_tf_ema.get(tf, {})
                    
                    # 30분봉은 메인 EMA 사용
                    if tf == '30m':
                        tf_data = ema if ema else {}
                    
                    if tf_data and tf_data.get('ema10'):
                        ema10_tf = tf_data.get('ema10', 0)
                        ema21_tf = tf_data.get('ema21', 0)
                        ema21_high_tf = tf_data.get('ema21_high', 0)
                        ema21_low_tf = tf_data.get('ema21_low', 0)
                        
                        # 배열 상태 확인
                        is_bullish_tf = ema10_tf > ema21_tf
                        state_emoji = "🟢" if is_bullish_tf else "🔴"
                        state_text = "정배열" if is_bullish_tf else "역배열"
                        
                        # 타임프레임 표시
                        timeframe_map = {'1m': '1분봉', '3m': '3분봉', '5m': '5분봉', '15m': '15분봉', '30m': '30분봉'}
                        tf_name = timeframe_map.get(tf, tf)
                        
                        # 차이 계산
                        if ema10_tf > 0 and ema21_high_tf > 0 and ema21_low_tf > 0:
                            diff_high = abs(ema10_tf - ema21_high_tf) / ema21_high_tf * 100
                            diff_low = abs(ema10_tf - ema21_low_tf) / ema21_low_tf * 100
                            
                            # 터치 판단
                            touch_high = diff_high <= 0.2 and is_bullish_tf
                            touch_low = diff_low <= 0.2 and not is_bullish_tf
                            
                            # 시각적 표시
                            def get_mini_bars(diff_pct):
                                if diff_pct < 0.5:
                                    return "━━━━━"
                                elif diff_pct < 1.0:
                                    return "━━━"
                                else:
                                    return "━"
                            
                            # 한 줄로 표시
                            print(f" {state_emoji} {tf_name:6} {state_text:4}:", end=" ")
                            
                            # EMA 값들과 차이 표시
                            if is_bullish_tf:
                                # 정배열: EMA10이 위에 있음
                                touch_mark = "⭐" if touch_high else ""
                                print(f"EMA21(H): ${ema21_high_tf:.2f} {get_mini_bars(diff_high)} ", end="")
                                print(f"↑{diff_high:.2f}% {touch_mark}", end=" ")
                                print(f"| EMA10: ${ema10_tf:.2f}", end=" ")
                                print(f"| EMA21(L): ${ema21_low_tf:.2f} ↓{diff_low:.2f}%")
                            else:
                                # 역배열: EMA10이 아래에 있음
                                touch_mark = "⭐" if touch_low else ""
                                print(f"EMA21(H): ${ema21_high_tf:.2f} ↑{diff_high:.2f}%", end=" ")
                                print(f"| EMA10: ${ema10_tf:.2f}", end=" ")
                                print(f"| EMA21(L): ${ema21_low_tf:.2f} {get_mini_bars(diff_low)} ", end="")
                                print(f"↓{diff_low:.2f}% {touch_mark}")
                        else:
                            print(f" {state_emoji} {tf_name:6}: 데이터 계산 중...")
                    else:
                        tf_name = {'1m': '1분봉', '3m': '3분봉', '5m': '5분봉', '15m': '15분봉', '30m': '30분봉'}.get(tf, tf)
                        print(f" ⚪ {tf_name:6}: 데이터 수집 중...")
                
                print(f" {'─'*76}")
                
                # 정렬 상태 요약
                alignment_details = {}
                for tf in ['1m', '3m', '5m', '15m', '30m']:
                    if tf == '30m':
                        tf_data = ema if ema else {}
                    else:
                        tf_data = multi_tf_ema.get(tf, {})
                    
                    if tf_data and tf_data.get('ema10'):
                        is_bullish = tf_data.get('ema10', 0) > tf_data.get('ema21', 0)
                        alignment_details[tf] = {'is_bullish': is_bullish}
                
                if alignment_details:
                    bullish_count = sum(1 for d in alignment_details.values() if d.get('is_bullish', False))
                    bearish_count = 5 - bullish_count
                    
                    if bullish_count == 5:
                        print(f" ⚡ 강한 정배열 신호: {bullish_count}/5 타임프레임 정배열 🟢📈")
                    elif bearish_count == 5:
                        print(f" ⚡ 강한 역배열 신호: {bearish_count}/5 타임프레임 역배열 🔴📉")
                    else:
                        print(f" 📊 혼재 신호: 정배열 {bullish_count}/5, 역배열 {bearish_count}/5")
            
            # 메인 EMA 상태 (30분봉 상세)
            if ema and ema.get('ema10') is not None:
                print(f"\n 📈 메인 EMA 지표 (30분봉 상세):")
                ema10 = ema.get('ema10', 0)
                ema21 = ema.get('ema21', 0)
                ema21_high = ema.get('ema21_high', 0)
                ema21_low = ema.get('ema21_low', 0)
                
                print(f"\n EMA 배열 상태 (30분봉 메인):")
                
                # 시각적 EMA 표시
                if ema10 > 0 and ema21 > 0:
                    # 각 EMA와 EMA10의 차이 계산 (백분율)
                    diff_21 = abs(ema10 - ema21) / ema21 * 100
                    diff_21_high = abs(ema10 - ema21_high) / ema21_high * 100
                    diff_21_low = abs(ema10 - ema21_low) / ema21_low * 100
                    
                    # 터치 판단
                    touch_high = diff_21_high <= 0.2
                    touch_low = diff_21_low <= 0.2
                    
                    # EMA 배열 상태 확인
                    is_bullish = ema10 > ema21
                    
                    # 정배열/역배열에 따른 터치 가능 여부
                    if is_bullish:  # 정배열
                        touch_low = False  # 정배열에서는 저가 터치 불가능
                    else:  # 역배열
                        touch_high = False  # 역배열에서는 고가 터치 불가능
                    
                    # 시각적 표시를 위한 막대 계산
                    def get_bars(diff_pct):
                        """차이 비율에 따른 막대 개수 계산"""
                        if diff_pct < 0.5:
                            return "━━━━━━━━━━"  # 10개
                        elif diff_pct < 1.0:
                            return "━━━━━━━━"
                        elif diff_pct < 1.5:
                            return "━━━━━━"
                        elif diff_pct < 2.0:
                            return "━━━━"
                        else:
                            return "━━"
                    
                    # 위치 관계 표시 (EMA10 기준)
                    if ema10 > max(ema21, ema21_high, ema21_low):
                        # EMA10이 가장 위
                        print(f"  • EMA10: ${ema10:.2f} {get_bars(0)} (현재 위치)")
                        
                        # 나머지를 높은 순서대로 정렬
                        ema_list = [
                            ('EMA21(고가)', ema21_high, diff_21_high, touch_high),
                            ('EMA21(시가)', ema21, diff_21, False),
                            ('EMA21(저가)', ema21_low, diff_21_low, touch_low)
                        ]
                        ema_list.sort(key=lambda x: x[1], reverse=True)
                        
                        for name, value, diff, touch in ema_list:
                            star = " ⭐" if touch else ""
                            bars = get_bars(diff)
                            print(f"  • {name}: ${value:.2f} {bars} ↓ {diff:.2f}%{star}")
                    
                    elif ema10 < min(ema21, ema21_high, ema21_low):
                        # EMA10이 가장 아래
                        ema_list = [
                            ('EMA21(고가)', ema21_high, diff_21_high, touch_high),
                            ('EMA21(시가)', ema21, diff_21, False),
                            ('EMA21(저가)', ema21_low, diff_21_low, touch_low)
                        ]
                        ema_list.sort(key=lambda x: x[1], reverse=True)
                        
                        for name, value, diff, touch in ema_list:
                            star = " ⭐" if touch else ""
                            bars = get_bars(diff)
                            print(f"  • {name}: ${value:.2f} {bars} ↑ {diff:.2f}%{star}")
                        
                        print(f"  • EMA10: ${ema10:.2f} {get_bars(0)} (현재 위치)")
                    
                    else:
                        # EMA10이 중간 어딘가
                        all_emas = [
                            ('EMA10', ema10, 0, False, True),  # is_ema10 = True
                            ('EMA21(고가)', ema21_high, diff_21_high, touch_high, False),
                            ('EMA21(시가)', ema21, diff_21, False, False),
                            ('EMA21(저가)', ema21_low, diff_21_low, touch_low, False)
                        ]
                        all_emas.sort(key=lambda x: x[1], reverse=True)
                        
                        for name, value, diff, touch, is_ema10 in all_emas:
                            if is_ema10:
                                print(f"  • {name}: ${value:.2f} {get_bars(0)} (현재 위치)")
                            else:
                                star = " ⭐" if touch else ""
                                if value > ema10:
                                    bars = get_bars(diff)
                                    print(f"  • {name}: ${value:.2f} {bars} ↑ {diff:.2f}%{star}")
                                else:
                                    bars = get_bars(diff)
                                    print(f"  • {name}: ${value:.2f} {bars} ↓ {diff:.2f}%{star}")
                else:
                    # 기본 표시 (데이터 부족 시)
                    print(f"  • EMA10: ${ema10:.2f} (시가)")
                    print(f"  • EMA21: ${ema21:.2f} (시가)")
                    print(f"  • EMA21(고가): ${ema21_high:.2f}")
                    print(f"  • EMA21(저가): ${ema21_low:.2f}")
                
                # EMA 상태 판단
                if ema10 > ema21:
                    ema_state = " 정배열 (BULLISH)"
                    signal = "롱 대기 (EMA21 고가 터치시)"
                else:
                    ema_state = " 역배열 (BEARISH)"
                    signal = "숏 대기 (EMA21 저가 터치시)"
                print(f"  • 상태: {ema_state}")
                print(f"  • 신호: {signal}")
                
                # 터치 상태 섹션
                if ema10 > 0 and (touch_high or touch_low):
                    print(f"\n 터치 상태:")
                    if is_bullish and touch_high:
                        print(f"  • ⭐ EMA21(고가) 터치 조건 충족 ({diff_21_high:.3f}% < 0.2%)")
                        # 터치 유지 시간 체크 (bot의 touch_state 참조)
                        if self.bot and hasattr(self.bot, 'touch_state'):
                            if self.bot.touch_state['long']['is_touching']:
                                duration = (datetime.now() - self.bot.touch_state['long']['start_time']).total_seconds()
                                if duration >= 180:
                                    duration_min = duration / 60
                                    print(f"     → ✅ 롱 진입 가능 (3분 이상 유지: {duration_min:.1f}분)")
                                else:
                                    print(f"     → ⏱️ 대기 중 ({duration:.0f}초/180초)")
                            else:
                                print(f"     → 터치 시작 대기")
                        else:
                            print(f"     → 3분 이상 유지시 롱 진입")
                    elif not is_bullish and touch_low:
                        print(f"  • ⭐ EMA21(저가) 터치 조건 충족 ({diff_21_low:.3f}% < 0.2%)")
                        # 터치 유지 시간 체크
                        if self.bot and hasattr(self.bot, 'touch_state'):
                            if self.bot.touch_state['short']['is_touching']:
                                duration = (datetime.now() - self.bot.touch_state['short']['start_time']).total_seconds()
                                if duration >= 180:
                                    duration_min = duration / 60
                                    print(f"     → ✅ 숏 진입 가능 (3분 이상 유지: {duration_min:.1f}분)")
                                else:
                                    print(f"     → ⏱️ 대기 중 ({duration:.0f}초/180초)")
                            else:
                                print(f"     → 터치 시작 대기")
                        else:
                            print(f"     → 3분 이상 유지시 숏 진입")
            
            # 1m/3m/5m 동시 정렬 상태 표시
            if self.bot and hasattr(self.bot, 'tf_alignment_state'):
                tf_state = self.bot.tf_alignment_state
                current_time = datetime.now()
                
                # 동시 정렬 상태 표시
                alignment_info = []
                
                if tf_state['bullish']['is_aligned']:
                    duration = (current_time - tf_state['bullish']['start_time']).total_seconds() / 60
                    alignment_info.append(f"🟢 1m/3m/5m 모두 정배열: {duration:.1f}분")
                
                if tf_state['bearish']['is_aligned']:
                    duration = (current_time - tf_state['bearish']['start_time']).total_seconds() / 60
                    alignment_info.append(f"🔴 1m/3m/5m 모두 역배열: {duration:.1f}분")
                
                if alignment_info:
                    print(f"\n ⏱️ 하위 타임프레임 동시 정렬:")
                    for info in alignment_info:
                        print(f"  • {info}")
                        
                        # 포지션이 있고 15분 이상 반대 배열 시 경고
                        if self.monitoring_data.get('position'):
                            pos = self.monitoring_data['position']
                            side = pos['side'].lower()
                            if '🔴' in info and side in ['buy', 'long']:  # 역배열이고 롱 포지션
                                if duration >= 120.0:
                                    print(f"     ⚠️ 긴급청산 발동!")
                                elif duration >= 110.0:
                                    print(f"     ⚠️ 긴급청산 임박 (남은시간: {120-duration:.1f}분)")
                            elif '🟢' in info and side in ['sell', 'short']:  # 정배열이고 숏 포지션
                                if duration >= 120.0:
                                    print(f"     ⚠️ 긴급청산 발동!")
                                elif duration >= 110.0:
                                    print(f"     ⚠️ 긴급청산 임박 (남은시간: {120-duration:.1f}분)")
            
            # 포지션 정보
            position = self.monitoring_data.get('position')
            if position:
                # side가 잘못된 경우 체크 (buy/long, sell/short)
                side = position['side'].lower()
                if side in ['buy', 'long']:
                    side_kr = "매수"
                    side_en = "LONG"
                elif side in ['sell', 'short']:
                    side_kr = "매도"
                    side_en = "SHORT"
                else:
                    side_kr = side
                    side_en = side.upper()
                
                print(f"\n 활성 포지션:")
                print(f"  • 방향: {side_kr} ({side_en})")
                print(f"  • 수량: {position['quantity']:.3f} SOL")
                
                # USDT 가치 표시
                if 'position_size_usdt' in position:
                    print(f"  • 포지션 크기: ${position['position_size_usdt']:.4f} USDT")
                else:
                    position_value = position['quantity'] * position['entry_price']
                    print(f"  • 포지션 크기: ${position_value:.4f} USDT")
                
                print(f"  • 진입가: ${position.get('entry_price', 0):.4f}")
                if price and price > 0:
                    print(f"  • 현재가: ${price:.4f}")
                else:
                    print(f"  • 현재가: 데이터 없음")
                print(f"  • 레버리지: {position.get('leverage', 10)}x")
                print(f"  • 격리마진")
                
                # 추가 정보 표시
                if 'margin' in position and position['margin'] > 0:
                    print(f"  • 마진: ${position['margin']:.2f} USDT")
                if 'liquidation_price' in position and position['liquidation_price'] > 0:
                    print(f"  • 청산가: ${position['liquidation_price']:.4f}")
                
                # ROI 계산 (레버리지 적용)
                leverage = position.get('leverage', 10)
                side = position['side'].lower()
                
                # API에서 제공하는 percentage가 있으면 우선 사용
                if 'percentage' in position and position['percentage'] is not None:
                    roi = position['percentage']
                else:
                    # 직접 계산 - None 체크 추가
                    if price is None or position.get('entry_price') is None:
                        roi = 0.0
                    elif side in ['buy', 'long']:
                        roi = ((price - position['entry_price']) / position['entry_price']) * 100 * leverage
                    else:
                        roi = ((position['entry_price'] - price) / position['entry_price']) * 100 * leverage
                
                roi_display = f"{roi:+.2f}%"
                if roi >= 0:
                    roi_display = f" {roi_display}"
                else:
                    roi_display = f" {roi_display}"
                
                print(f"  • ROI: {roi_display}")
                
                # 최고 ROI 및 가격 표시 (DB에서 가져오거나 트레일링 정보 활용)
                if self.bot and hasattr(self.bot, 'last_trailing_update'):
                    trailing_info = self.bot.last_trailing_update
                    
                    if position['side'].lower() in ['buy', 'long'] and trailing_info.get('highest_price'):
                        highest_price = trailing_info['highest_price']
                        highest_roi = ((highest_price - position['entry_price']) / position['entry_price']) * 100 * leverage
                        print(f"  • 최고가: ${highest_price:.4f} (최고 ROI: +{highest_roi:.2f}%)")
                    elif position['side'].lower() in ['sell', 'short'] and trailing_info.get('lowest_price'):
                        lowest_price = trailing_info['lowest_price']
                        highest_roi = ((position['entry_price'] - lowest_price) / position['entry_price']) * 100 * leverage
                        print(f"  • 최저가: ${lowest_price:.4f} (최고 ROI: +{highest_roi:.2f}%)")
                    else:
                        # 현재가가 최고/최저가인 경우
                        if roi > 0:
                            if position['side'].lower() in ['buy', 'long']:
                                print(f"  • 최고가: ${price:.4f} (최고 ROI: {roi_display})")
                            else:
                                print(f"  • 최저가: ${price:.4f} (최고 ROI: {roi_display})")
                
                # 보유 시간
                holding_time = (datetime.now() - position['opened_at']).total_seconds() / 60
                holding_hours = holding_time / 60
                time_display = f"{holding_time:.0f}분" if holding_time < 60 else f"{holding_hours:.1f}시간"
                print(f"  • 보유시간: {time_display}")
                
                # SL 및 트레일링 정보
                if 'sl_price' in position and position['sl_price'] is not None and position['entry_price'] is not None:
                    sl_roi = ((position['sl_price'] - position['entry_price']) / position['entry_price']) * 100
                    if position['side'] == 'sell':
                        sl_roi = -sl_roi
                    print(f"  • SL: ${position['sl_price']:.2f} ({sl_roi:+.1f}%)")
                    
                    # 트레일링 스톱 상태 표시
                    if self.bot and hasattr(self.bot, 'last_trailing_update'):
                        trailing_info = self.bot.last_trailing_update
                        if trailing_info.get('breakeven_moved'):
                            print(f"  • 🛡️ 손익분기점 보호 활성화")
                            
                            if position['side'].lower() in ['buy', 'long'] and trailing_info.get('highest_price'):
                                highest_roi = ((trailing_info['highest_price'] - position['entry_price']) / position['entry_price']) * 100
                                print(f"  • 📈 최고가: ${trailing_info['highest_price']:.2f} (+{highest_roi:.1f}%)")
                            elif position['side'].lower() in ['sell', 'short'] and trailing_info.get('lowest_price'):
                                lowest_roi = ((position['entry_price'] - trailing_info['lowest_price']) / position['entry_price']) * 100
                                print(f"  • 📉 최저가: ${trailing_info['lowest_price']:.2f} (+{lowest_roi:.1f}%)")
                
                
                # 수동 포지션 표시
                if position.get('is_manual'):
                    print(f"  •  수동 포지션 (추적 중)")
                
                # 청산 조건 표시
                print(f"\n 청산 조건:")
                if position['side'] == 'buy':
                    print(f"  • 역배열 전환 후 EMA21(저가) 터치시 청산 → 숏 진입")
                else:
                    print(f"  • 정배열 전환 후 EMA21(고가) 터치시 청산 → 롱 진입")
            else:
                print(f"\n 포지션 없음")
                print(f"\n 진입 대기 중:")
                print(f"  • 정배열시: EMA10이 EMA21(고가)에 터치 (0.2% 이내) → 롱 진입")
                print(f"  • 역배열시: EMA10이 EMA21(저가)에 터치 (0.2% 이내) → 숏 진입")
            
            # 통계
            stats = self.monitoring_data['stats']
            if stats:
                print(f"\n 거래 통계:")
                total = stats.get('total_trades', 0)
                if total > 0:
                    win_rate = (stats.get('winning_trades', 0) / total) * 100
                    print(f"  • 총 거래: {total}")
                    print(f"  • 승률: {win_rate:.1f}%")
                    print(f"  • 총 손익: ${stats.get('total_profit', 0):.2f}")
                else:
                    print(f"  • 거래 없음")
            
            # 리스크 관리 설정
            print(f"\n 리스크 관리:")
            print(f"  • 기본 손절: {self.config.base_stop_loss}%")
            print(f"  • 트레일링 스톱: ROI 30%에서 손익분기점 이동, 최고점에서 30% 트레일링")
            
            # 시스템 상태
            print(f"\n 시스템 상태: {self.monitoring_data['system_status']}")
            print(f"  • 모니터링 주기: 30초")
            print(f"  • 수동 포지션: 추적 지원")
            print(f"{'='*80}")
            
            # stdout 복원
            sys.stdout = old_stdout
            
            # 버퍼에서 내용 가져오기
            output = buffer.getvalue()
            
            # 콘솔에 출력
            print(output, end='')
            
            # 파일에 저장
            self.monitoring_logger.info(output)
            
        except Exception as e:
            # stdout 복원 (에러 발생 시에도)
            sys.stdout = old_stdout
            self.logger.error(f"상태 출력 오류: {str(e)}")
    
    async def stop(self):
        """모니터링 중지"""
        try:
            self.monitoring_data['system_status'] = 'STOPPING'
            
            # 태스크 취소
            if self.monitoring_task and not self.monitoring_task.done():
                self.monitoring_task.cancel()
                try:
                    await self.monitoring_task
                except asyncio.CancelledError:
                    pass
            
            self.logger.info("✅ 모니터링 중지")
            
        except Exception as e:
            self.logger.error(f"모니터링 중지 오류: {str(e)}")
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
💰 SOL Futures 자동매매 시스템 v4.0 - Order Executor
주문 실행 및 포지션 관리
"""

import asyncio
import ccxt.async_support as ccxt
from datetime import datetime
from typing import Dict, Optional, List
import logging
import time

class OrderExecutor:
    """주문 실행기"""
    
    def __init__(self, config, logger, db, telegram=None):
        self.config = config
        self.logger = logger
        self.db = db
        self.telegram = telegram
        
        # 거래소
        self.exchange = None
        
        # 현재 포지션
        self.current_position = None
        
        # 통계
        self.stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit': 0.0
        }
        
        self.logger.info("✅ Order Executor 초기화 완료")
    
    async def initialize(self):
        """거래소 연결 초기화"""
        try:
            # Binance futures 설정
            self.exchange = ccxt.binance({
                'apiKey': self.config.api_key,
                'secret': self.config.api_secret,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'future',
                    'adjustForTimeDifference': True,
                    'recvWindow': 60000,  # 60초 허용 범위
                    'timeDifference': 0  # 시간 차이 자동 조정
                }
            })
            
            # 시간 동기화
            await self.exchange.load_time_difference()
            
            # 시장 정보 로드
            await self.exchange.load_markets()
            
            # 포지션 모드 설정 (One-way mode)
            try:
                await self.exchange.set_position_mode(hedged=False)
                self.logger.info("✅ 포지션 모드: One-way")
            except Exception as e:
                if 'position side' in str(e).lower():
                    self.logger.info("✅ 포지션 모드 이미 설정됨: One-way")
            
            # 마진 모드 설정
            await self.set_margin_mode()
            
            # 레버리지 설정
            await self.set_leverage()
            
            # 잔고 확인
            balance = await self.get_balance()
            self.logger.info(f"✅ 거래소 연결 성공 - USDT 잔고: ${balance:.2f}")
            
        except Exception as e:
            self.logger.error(f"❌ 거래소 초기화 실패: {str(e)}")
            raise
    
    async def set_margin_mode(self):
        """마진 모드 설정 (격리)"""
        try:
            await self.exchange.set_margin_mode(
                marginMode=self.config.margin_mode.lower(),
                symbol=self.config.symbol
            )
            self.logger.info(f"✅ 마진 모드 설정: {self.config.margin_mode}")
        except Exception as e:
            if 'No need to change' in str(e) or 'margin mode is not changed' in str(e).lower():
                self.logger.info(f"✅ 마진 모드 이미 설정됨: {self.config.margin_mode}")
            else:
                self.logger.error(f"마진 모드 설정 오류: {str(e)}")
    
    async def set_leverage(self):
        """레버리지 설정"""
        try:
            # 현재 포지션 확인
            positions = await self.get_open_positions()
            if positions:
                # 열린 포지션이 있을 때
                current_leverage = positions[0].get('leverage', 0)
                if current_leverage > self.config.leverage:
                    self.logger.warning(f"⚠️ 현재 포지션 레버리지({current_leverage}x)가 설정값({self.config.leverage}x)보다 높음")
                    self.logger.warning(f"   격리마진 모드에서는 포지션이 있을 때 레버리지 감소 불가")
                    self.logger.info(f"   현재 레버리지 유지: {current_leverage}x")
                    return
            
            await self.exchange.set_leverage(
                leverage=self.config.leverage,
                symbol=self.config.symbol
            )
            self.logger.info(f"✅ 레버리지 설정: {self.config.leverage}x")
        except Exception as e:
            error_msg = str(e).lower()
            if 'leverage not changed' in error_msg:
                self.logger.info(f"✅ 레버리지 이미 설정됨: {self.config.leverage}x")
            elif 'leverage reduction is not supported' in error_msg:
                self.logger.warning(f"⚠️ 격리마진 모드에서 포지션이 있을 때 레버리지 감소 불가")
                self.logger.info(f"   현재 레버리지를 유지합니다")
            else:
                self.logger.error(f"레버리지 설정 오류: {str(e)}")
    
    async def get_balance(self) -> float:
        """USDT 잔고 조회"""
        try:
            balance = await self.exchange.fetch_balance()
            usdt_balance = balance['USDT']['free']
            return float(usdt_balance)
        except Exception as e:
            self.logger.error(f"잔고 조회 오류: {str(e)}")
            return 0.0
    
    async def get_open_positions(self) -> List[Dict]:
        """열린 포지션 조회"""
        try:
            # 시간 동기화 재시도
            retries = 3
            for attempt in range(retries):
                try:
                    positions = await self.exchange.fetch_positions([self.config.symbol])
                    break
                except Exception as e:
                    if 'Timestamp' in str(e) and attempt < retries - 1:
                        self.logger.warning(f"시간 동기화 오류, 재시도 중... ({attempt + 1}/{retries})")
                        await self.exchange.load_time_difference()
                        await asyncio.sleep(0.5)
                    else:
                        raise e
            
            open_positions = []
            
            for pos in positions:
                if pos['contracts'] > 0:
                    # 실제 포지션 크기 계산 (USDT)
                    position_size = abs(pos.get('notional', 0))  # notional이 포지션 크기
                    if position_size == 0:
                        position_size = pos['contracts'] * pos.get('markPrice', pos.get('entryPrice', 0))
                    
                    # ccxt에서 side는 'long' 또는 'short'로 제공됨
                    side = pos['side']
                    if side == 'long':
                        side = 'buy'
                    elif side == 'short':
                        side = 'sell'
                    
                    open_positions.append({
                        'symbol': pos['symbol'],
                        'side': side,  # buy/sell로 변환
                        'quantity': pos['contracts'],
                        'position_size_usdt': position_size,  # 실제 포지션 크기 (USDT)
                        'entry_price': pos['entryPrice'] or 0,
                        'mark_price': pos['markPrice'] or 0,
                        'unrealized_pnl': pos['unrealizedPnl'] or 0,
                        'percentage': pos['percentage'] or 0,  # Binance가 제공하는 ROI
                        'leverage': int(pos.get('leverage', 10)),  # 레버리지를 정수로
                        'is_manual': False,  # API로 조회한 포지션
                        'opened_at': datetime.now(),  # 추적 시작 시간
                        'margin': pos.get('initialMargin', 0),  # 초기 마진
                        'margin_ratio': pos.get('marginRatio', 0),  # 마진 비율
                        'liquidation_price': pos.get('liquidationPrice', 0)  # 청산가
                    })
            
            return open_positions
            
        except Exception as e:
            self.logger.error(f"포지션 조회 오류: {str(e)}")
            return []
    
    def has_position(self) -> bool:
        """포지션 보유 여부"""
        has_pos = self.current_position is not None
        if not has_pos:
            self.logger.debug(f"has_position: False (current_position is None)")
        else:
            self.logger.debug(f"has_position: True - {self.current_position['side']} {self.current_position['quantity']:.3f} SOL")
        return has_pos
    
    async def open_position(self, side: str, quantity: float, leverage: int) -> bool:
        """포지션 진입"""
        try:
            # 기존 포지션 확인
            if self.has_position():
                self.logger.warning("이미 포지션이 있습니다")
                return False
            
            # 주문 실행
            order = await self.exchange.create_market_order(
                symbol=self.config.symbol,
                side=side,  # 'buy' or 'sell'
                amount=quantity,
                params={'leverage': leverage}
            )
            
            if order and order['status'] == 'closed':
                # 포지션 정보 저장
                self.current_position = {
                    'symbol': self.config.symbol,
                    'side': side,
                    'quantity': quantity,
                    'entry_price': order['average'],
                    'opened_at': datetime.now(),
                    'order_id': order['id'],
                    'leverage': leverage,
                    'partial_closed': False
                }
                
                # SL 설정
                await self.set_tp_sl()
                
                # 통계 업데이트
                self.stats['total_trades'] += 1
                
                # 거래 기록
                self.record_trade_open(order)
                
                # 텔레그램 알림
                if self.telegram:
                    await self.telegram.send_position_open(
                        side=side,
                        quantity=quantity,
                        price=order['average'],
                        leverage=leverage
                    )
                
                self.logger.info(f"✅ {side.upper()} 포지션 진입 성공 | {quantity:.3f} SOL @ ${order['average']:.2f} | {leverage}x")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"포지션 진입 오류: {str(e)}")
            return False
    
    async def close_position(self, reason: str = "MANUAL") -> bool:
        """전체 포지션 청산"""
        try:
            if not self.has_position():
                self.logger.warning("청산할 포지션이 없습니다")
                
                # 실제 포지션 다시 확인
                actual_positions = await self.get_open_positions()
                if actual_positions:
                    self.logger.warning(f"⚠️ current_position은 None이지만 실제 포지션 {len(actual_positions)}개 발견")
                    # 첫 번째 포지션으로 설정
                    for pos in actual_positions:
                        normalized_api_symbol = pos['symbol'].replace('/', '').replace(':USDT', '')
                        normalized_config_symbol = self.config.symbol.replace('/', '')
                        if normalized_api_symbol == normalized_config_symbol or 'SOL' in pos['symbol']:
                            self.current_position = pos
                            self.logger.info(f"포지션 복구: {pos['side'].upper()} {pos['quantity']:.3f} SOL @ ${pos.get('entry_price', 0):.2f}")
                            break
                
                if not self.current_position:
                    return False
            
            position = self.current_position
            self.logger.info(f"📌 청산할 포지션: {position['side']} {position['quantity']:.3f} SOL @ ${position['entry_price']:.2f} (사유: {reason})")
            
            # 반대 방향으로 시장가 주문
            close_side = 'sell' if position['side'] == 'buy' else 'buy'
            
            order = await self.exchange.create_market_order(
                symbol=self.config.symbol,
                side=close_side,
                amount=position['quantity']
            )
            
            if order and order['status'] == 'closed':
                # SL 주문 취소
                self.logger.info("포지션 청산 - SL 주문 취소 중...")
                await self.cancel_all_orders()
                
                # 손익 계산
                exit_price = order['average']
                pnl = self.calculate_pnl(position, exit_price)
                
                # 통계 업데이트
                if pnl['profit'] > 0:
                    self.stats['winning_trades'] += 1
                else:
                    self.stats['losing_trades'] += 1
                self.stats['total_profit'] += pnl['profit']
                
                # 거래 기록
                self.record_trade_close(position, exit_price, pnl, reason)
                
                # 텔레그램 알림
                if self.telegram:
                    await self.telegram.send_position_close(
                        side=position['side'],
                        entry_price=position['entry_price'],
                        exit_price=exit_price,
                        profit=pnl['profit'],
                        roi=pnl['percent'],
                        holding_time=pnl['holding_time'],
                        reason=reason
                    )
                
                if pnl['profit'] >= 0:
                    self.logger.info(
                        f"✅ 포지션 청산 성공 | "
                        f"수익: +${pnl['profit']:.2f} (+{pnl['percent']:.2f}%) 📈 | "
                        f"사유: {reason}"
                    )
                else:
                    self.logger.info(
                        f"✅ 포지션 청산 성공 | "
                        f"손실: ${pnl['profit']:.2f} ({pnl['percent']:.2f}%) 📉 | "
                        f"사유: {reason}"
                    )
                
                # 포지션 초기화
                self.current_position = None
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"포지션 청산 오류: {str(e)}")
            return False
    
    
    async def set_tp_sl(self):
        """SL 설정"""
        try:
            if not self.has_position():
                return
            
            position = self.current_position
            entry_price = position['entry_price']
            
            # 기존 스톱 주문 모두 취소
            self.logger.info("기존 스톱 주문 취소 중...")
            await self.cancel_all_orders()
            await asyncio.sleep(0.5)  # 취소 처리 대기
            
            # SL 가격 계산 (레버리지 적용 ROI)
            leverage = position.get('leverage', self.config.leverage)
            if position['side'] == 'buy':
                sl_price = entry_price * (1 - self.config.base_stop_loss / 100 / leverage)
            else:
                sl_price = entry_price * (1 + self.config.base_stop_loss / 100 / leverage)
            
            # SL 주문만 설정 (TP 제거)
            await self.place_stop_order(sl_price, 'STOP_LOSS')
            
            # 포지션 정보 업데이트
            position['sl_price'] = sl_price
            
            self.logger.info(f"✅ SL 설정 - SL: ${sl_price:.2f}")
            
        except Exception as e:
            self.logger.error(f"SL 설정 오류: {str(e)}")
    
    async def place_stop_order(self, price: float, order_type: str):
        """스톱 주문 생성"""
        try:
            position = self.current_position
            side = 'sell' if position['side'] == 'buy' else 'buy'
            
            # STOP_LOSS만 처리
            if order_type == 'STOP_LOSS':
                order = await self.exchange.create_order(
                    symbol=self.config.symbol,
                    type='stop_market',
                    side=side,
                    amount=position['quantity'],
                    params={
                        'stopPrice': price,
                        'workingType': 'MARK_PRICE',
                        'positionSide': 'BOTH'
                    }
                )
                return order
            
            return None
            
        except Exception as e:
            self.logger.error(f"스톱 주문 오류: {str(e)}")
            return None
    
    async def modify_position(self, sl_price: Optional[float] = None):
        """포지션 수정 (SL 업데이트)"""
        try:
            if not self.has_position():
                return False
            
            # 기존 스톱 주문 취소
            await self.cancel_all_orders()
            
            # 새로운 SL 설정
            if sl_price:
                await self.place_stop_order(sl_price, 'STOP_LOSS')
                self.current_position['sl_price'] = sl_price
            
            return True
            
        except Exception as e:
            self.logger.error(f"포지션 수정 오류: {str(e)}")
            return False
    
    async def cancel_all_orders(self):
        """모든 주문 취소"""
        try:
            # 열린 주문 조회
            open_orders = await self.exchange.fetch_open_orders(self.config.symbol)
            if open_orders:
                self.logger.info(f"취소할 주문 {len(open_orders)}개 발견")
                for order in open_orders:
                    try:
                        await self.exchange.cancel_order(order['id'], self.config.symbol)
                        self.logger.info(f"주문 취소: {order['id']} ({order.get('type', 'unknown')})")
                    except Exception as e:
                        self.logger.error(f"개별 주문 취소 실패: {e}")
            else:
                # 전체 취소 시도
                await self.exchange.cancel_all_orders(self.config.symbol)
        except Exception as e:
            self.logger.error(f"주문 취소 오류: {str(e)}")
    
    def calculate_pnl(self, position: Dict, exit_price: float) -> Dict:
        """손익 계산"""
        entry_price = position['entry_price']
        quantity = position['quantity']
        
        if position['side'] == 'buy':
            profit = (exit_price - entry_price) * quantity
            percent = ((exit_price - entry_price) / entry_price) * 100
        else:
            profit = (entry_price - exit_price) * quantity
            percent = ((entry_price - exit_price) / entry_price) * 100
        
        # 레버리지는 이미 포지션 크기에 반영되어 있으므로 추가 적용하지 않음
        # profit *= position['leverage']  # 제거
        # percent *= position['leverage']  # 제거
        
        # 보유 시간
        holding_time = (datetime.now() - position['opened_at']).total_seconds() / 60
        
        return {
            'profit': profit,
            'percent': percent,
            'holding_time': holding_time
        }
    
    def record_trade_open(self, order: Dict):
        """거래 시작 기록"""
        try:
            cursor = self.db.cursor()
            cursor.execute('''
                INSERT INTO trades 
                (symbol, side, quantity, entry_price, order_id, partial_closed, breakeven_moved)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                self.config.symbol,
                self.current_position['side'],
                self.current_position['quantity'],
                order['average'],
                order['id'],
                0,  # 처음엔 부분청산 안됨
                0   # 처음엔 손익분기점 이동 안됨
            ))
            self.db.commit()
        except Exception as e:
            self.logger.error(f"거래 기록 오류: {str(e)}")
    
    def record_trade_close(self, position: Dict, exit_price: float, 
                          pnl: Dict, reason: str):
        """거래 종료 기록"""
        try:
            # order_id가 없는 경우 처리 (수동 포지션 등)
            order_id = position.get('order_id')
            if not order_id:
                self.logger.warning("order_id 없음 - 거래 기록 스킵")
                return
                
            cursor = self.db.cursor()
            cursor.execute('''
                UPDATE trades 
                SET exit_price = ?, profit_loss = ?, profit_percent = ?, 
                    holding_time_minutes = ?, close_reason = ?
                WHERE order_id = ?
            ''', (
                exit_price,
                pnl['profit'],
                pnl['percent'],
                pnl['holding_time'],
                reason,
                order_id
            ))
            self.db.commit()
        except Exception as e:
            self.logger.error(f"거래 기록 오류: {str(e)}")
    
    async def send_telegram_notification(self, message: str):
        """텔레그램 알림 전송"""
        try:
            if self.telegram and hasattr(self.telegram, 'send_message'):
                await self.telegram.send_message(message)
        except Exception as e:
            self.logger.error(f"텔레그램 알림 오류: {str(e)}")
    
    async def track_manual_position(self, position_info: Dict) -> bool:
        """수동 포지션 추적"""
        try:
            # 필수 정보 확인
            required_fields = ['symbol', 'side', 'quantity', 'entry_price']
            for field in required_fields:
                if field not in position_info:
                    self.logger.error(f"수동 포지션 추적 실패: {field} 정보 누락")
                    return False
            
            # 이미 추적 중인지 확인 - 강제 업데이트 허용
            if self.current_position:
                self.logger.info("기존 포지션을 새로운 수동 포지션으로 교체")
                # 기존 포지션을 무시하고 새로운 포지션으로 교체
            
            # 현재가 조회
            ticker = await self.exchange.fetch_ticker(position_info['symbol'])
            current_price = ticker['last']
            
            
            # 포지션 정보 저장
            self.current_position = {
                'symbol': position_info['symbol'],
                'side': position_info['side'].lower(),
                'quantity': float(position_info['quantity']),
                'entry_price': float(position_info['entry_price']),
                'opened_at': position_info.get('opened_at', datetime.now()),
                'order_id': position_info.get('order_id', f"manual_{int(time.time() * 1000)}"),
                'leverage': position_info.get('leverage', self.config.leverage),
                'is_manual': True,
                'partial_closed': False
            }
            
            
            # 기존 SL 주문 확인
            existing_sl = await self.get_existing_stop_loss()
            if existing_sl:
                self.current_position['sl_price'] = existing_sl
                self.logger.info(f"기존 SL 발견: ${existing_sl:.2f}")
            else:
                # SL 설정
                await self.set_tp_sl()
            
            self.logger.info(f"🔄 수동 포지션 추적 시작")
            self.logger.info(f"  • {self.current_position['side'].upper()} {self.current_position['quantity']:.3f} SOL")
            self.logger.info(f"  • 진입가: ${self.current_position['entry_price']:.2f}")
            self.logger.info(f"  • 레버리지: {self.current_position['leverage']}x")
            
            # 텔레그램 알림
            if self.telegram:
                await self.telegram.send_manual_position_tracking(
                    position=self.current_position
                )
            
            # DB에 기록 (기존 order_id가 없을 때만)
            if not position_info.get('order_id'):
                self.record_trade_open({
                    'id': self.current_position['order_id'],
                    'average': self.current_position['entry_price']
                })
            else:
                self.logger.info(f"📌 기존 order_id 사용: {self.current_position['order_id']}")
                
            
            return True
            
        except Exception as e:
            self.logger.error(f"수동 포지션 추적 오류: {str(e)}")
            return False
    
    async def cleanup(self):
        """리소스 정리"""
        try:
            if self.exchange:
                await self.exchange.close()
            
            self.logger.info("✅ Order Executor 정리 완료")
            
        except Exception as e:
            self.logger.error(f"정리 중 오류: {str(e)}")
    
    
    async def get_existing_stop_loss(self) -> Optional[float]:
        """기존 스톱로스 주문 확인"""
        try:
            # 열린 주문 조회
            open_orders = await self.exchange.fetch_open_orders(self.config.symbol)
            
            for order in open_orders:
                # 스톱 마켓 주문 찾기
                if order.get('type') == 'stop_market' and order.get('status') == 'open':
                    stop_price = order.get('stopPrice', 0)
                    if stop_price > 0:
                        self.logger.info(f"기존 스톱로스 주문 발견: ${stop_price:.2f}")
                        return stop_price
            
            return None
            
        except Exception as e:
            self.logger.error(f"기존 SL 확인 오류: {str(e)}")
            return None
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ALT/USDT 주문 실행 모듈
바이낸스 선물 주문 실행 및 관리
"""

import ccxt
import asyncio
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List
import time

class OrderExecutor:
    """주문 실행 클래스"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("ALTTrading.OrderExecutor")
        self.exchange = None
        self.open_orders = {}
        
    async def initialize(self):
        """Exchange 초기화"""
        try:
            # Binance futures 설정
            self.exchange = ccxt.binance({
                'apiKey': self.config.binance_api_key,
                'secret': self.config.binance_api_secret,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'future',
                    'adjustForTimeDifference': True,
                    'recvWindow': 60000  # 60초 타임윈도우
                }
            })
            
            # 시간 동기화
            await self.sync_time()
            
            # 마켓 로드
            self.exchange.load_markets()
            
            self.logger.info("✅ OrderExecutor 초기화 완료")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ OrderExecutor 초기화 실패: {e}")
            return False
    
    async def sync_time(self):
        """바이낸스 서버와 시간 동기화"""
        try:
            # 서버 시간 가져오기
            server_time = self.exchange.fetch_time()
            local_time = self.exchange.milliseconds()
            time_diff = server_time - local_time
            
            if abs(time_diff) > 1000:  # 1초 이상 차이
                self.logger.warning(f"⚠️ 시간 차이 감지: {time_diff}ms, 동기화 중...")
                # 시간 오프셋 설정
                self.exchange.options['timestamp'] = server_time
                self.exchange.options['serverTime'] = server_time
                self.logger.info(f"✅ 시간 동기화 완료: 오프셋 {time_diff}ms")
            
        except Exception as e:
            self.logger.error(f"시간 동기화 실패: {e}")
    
    async def place_order(
        self,
        symbol: str,
        side: str,  # LONG/SHORT or BUY/SELL
        quantity: float,
        price: Optional[float] = None,
        order_type: str = "MARKET"
    ) -> Optional[Dict]:
        """주문 실행"""
        try:
            # LONG/SHORT를 BUY/SELL로 변환
            if side == "LONG":
                order_side = "buy"
            elif side == "SHORT":
                order_side = "sell"
            else:
                order_side = side.lower()
            
            # 수량 정밀도 조정
            market = self.exchange.market(symbol)
            quantity = self.exchange.amount_to_precision(symbol, quantity)
            
            # 시장가 주문
            if order_type == "MARKET":
                order = self.exchange.create_order(
                    symbol=symbol,
                    type='market',
                    side=order_side,
                    amount=quantity
                )
                
                self.logger.info(f"✅ 시장가 주문 실행: {order_side} {quantity} {symbol}")
                
            # 지정가 주문
            else:
                if price is None:
                    ticker = self.exchange.fetch_ticker(symbol)
                    price = ticker['last']
                
                price = self.exchange.price_to_precision(symbol, price)
                
                order = self.exchange.create_order(
                    symbol=symbol,
                    type='limit',
                    side=order_side,
                    amount=quantity,
                    price=price
                )
                
                self.logger.info(f"✅ 지정가 주문 실행: {order_side} {quantity} @ {price} {symbol}")
            
            # 주문 정보 저장
            self.open_orders[order['id']] = order
            
            return order
            
        except Exception as e:
            self.logger.error(f"❌ 주문 실행 실패: {e}")
            return None
    
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """주문 취소"""
        try:
            self.exchange.cancel_order(order_id, symbol)
            
            if order_id in self.open_orders:
                del self.open_orders[order_id]
            
            self.logger.info(f"✅ 주문 취소: {order_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"주문 취소 실패: {e}")
            return False
    
    async def cancel_all_orders(self, symbol: str) -> bool:
        """모든 주문 취소"""
        try:
            self.exchange.cancel_all_orders(symbol)
            self.open_orders.clear()
            
            self.logger.info("✅ 모든 주문 취소 완료")
            return True
            
        except Exception as e:
            self.logger.error(f"모든 주문 취소 실패: {e}")
            return False
    
    async def get_open_orders(self, symbol: str) -> List[Dict]:
        """미체결 주문 조회"""
        try:
            orders = self.exchange.fetch_open_orders(symbol)
            return orders
            
        except Exception as e:
            self.logger.error(f"미체결 주문 조회 실패: {e}")
            return []
    
    async def get_order_status(self, order_id: str, symbol: str) -> Optional[Dict]:
        """주문 상태 조회"""
        try:
            order = self.exchange.fetch_order(order_id, symbol)
            return order
            
        except Exception as e:
            self.logger.error(f"주문 상태 조회 실패: {e}")
            return None
    
    async def get_positions(self) -> List[Dict]:
        """현재 포지션 조회"""
        try:
            positions = self.exchange.fetch_positions([self.config.symbol])
            
            # 활성 포지션만 필터링
            active_positions = [
                pos for pos in positions
                if pos['contracts'] > 0
            ]
            
            return active_positions
            
        except Exception as e:
            self.logger.error(f"포지션 조회 실패: {e}")
            return []
    
    async def get_account_balance(self) -> float:
        """계정 잔액 조회"""
        try:
            balance = self.exchange.fetch_balance()
            
            # USDT 잔액
            usdt_free = balance['USDT']['free']
            usdt_used = balance['USDT']['used']
            total_balance = usdt_free + usdt_used
            
            return total_balance
            
        except Exception as e:
            self.logger.error(f"잔액 조회 실패: {e}")
            return 0.0
    
    async def set_stop_loss(
        self,
        symbol: str,
        side: str,
        quantity: float,
        stop_price: float
    ) -> Optional[Dict]:
        """스톱로스 설정"""
        try:
            # 포지션과 반대 방향
            if side == "LONG":
                stop_side = "sell"
            else:
                stop_side = "buy"
            
            # 스톱 마켓 주문
            order = self.exchange.create_order(
                symbol=symbol,
                type='stop_market',
                side=stop_side,
                amount=quantity,
                stopPrice=stop_price,
                params={'reduceOnly': True}
            )
            
            self.logger.info(f"✅ 스톱로스 설정: {stop_price}")
            return order
            
        except Exception as e:
            self.logger.error(f"스톱로스 설정 실패: {e}")
            return None
    
    async def set_take_profit(
        self,
        symbol: str,
        side: str,
        quantity: float,
        take_profit_price: float
    ) -> Optional[Dict]:
        """익절 설정"""
        try:
            # 포지션과 반대 방향
            if side == "LONG":
                tp_side = "sell"
            else:
                tp_side = "buy"
            
            # 익절 마켓 주문
            order = self.exchange.create_order(
                symbol=symbol,
                type='take_profit_market',
                side=tp_side,
                amount=quantity,
                stopPrice=take_profit_price,
                params={'reduceOnly': True}
            )
            
            self.logger.info(f"✅ 익절 설정: {take_profit_price}")
            return order
            
        except Exception as e:
            self.logger.error(f"익절 설정 실패: {e}")
            return None
    
    async def close_position(self, symbol: str, side: str, quantity: float) -> Optional[Dict]:
        """포지션 청산"""
        try:
            # 포지션과 반대 방향으로 시장가 주문
            if side == "LONG":
                close_side = "sell"
            else:
                close_side = "buy"
            
            order = self.exchange.create_order(
                symbol=symbol,
                type='market',
                side=close_side,
                amount=quantity,
                params={'reduceOnly': True}
            )
            
            self.logger.info(f"✅ 포지션 청산: {side} {quantity}")
            return order
            
        except Exception as e:
            self.logger.error(f"포지션 청산 실패: {e}")
            return None
    
    async def get_trading_fees(self) -> Dict[str, float]:
        """거래 수수료 조회"""
        try:
            trading_fee = self.exchange.fetch_trading_fee(self.config.symbol)
            
            return {
                'maker': trading_fee['maker'],
                'taker': trading_fee['taker']
            }
            
        except Exception as e:
            self.logger.error(f"수수료 조회 실패: {e}")
            return {
                'maker': 0.0002,  # 기본값 0.02%
                'taker': 0.0004   # 기본값 0.04%
            }
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📱 SOL Futures 자동매매 시스템 v4.0 - Telegram Helper
텔레그램 알림 시스템
"""

import asyncio
import aiohttp
from datetime import datetime
from typing import Optional, Dict
import logging

class TelegramBot:
    """텔레그램 알림 봇"""
    
    def __init__(self, token: str, chat_id: str, logger: logging.Logger):
        self.token = token
        self.chat_id = chat_id
        self.logger = logger
        self.base_url = f"https://api.telegram.org/bot{token}"
        self.session = None
        
    async def initialize(self):
        """초기화"""
        try:
            self.session = aiohttp.ClientSession()
            
            # 봇 정보 확인
            async with self.session.get(f"{self.base_url}/getMe") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data['ok']:
                        bot_info = data['result']
                        self.logger.info(f"✅ 텔레그램 봇 연결: @{bot_info['username']}")
                        return True
                    
            self.logger.error("❌ 텔레그램 봇 연결 실패")
            return False
            
        except Exception as e:
            self.logger.error(f"텔레그램 초기화 오류: {str(e)}")
            return False
    
    async def send_message(self, text: str, parse_mode: str = "HTML") -> bool:
        """메시지 전송"""
        try:
            if not self.session:
                return False
                
            data = {
                'chat_id': self.chat_id,
                'text': text,
                'parse_mode': parse_mode
            }
            
            async with self.session.post(
                f"{self.base_url}/sendMessage",
                data=data
            ) as resp:
                if resp.status == 200:
                    return True
                else:
                    self.logger.error(f"텔레그램 전송 실패: {resp.status}")
                    return False
                    
        except Exception as e:
            self.logger.error(f"텔레그램 메시지 전송 오류: {str(e)}")
            return False
    
    async def send_startup_message(self, config):
        """프로그램 시작 알림"""
        message = f"""
🚀 <b>솔라나(SOL) Futures Bot v4.0 시작</b>

📊 <b>시스템 설정</b>
• 거래쌍: {config.symbol}
• 레버리지: {config.leverage}x
• 포지션 크기: {config.position_size}%
• 타임프레임: {config.timeframe}

📈 <b>전략</b>
• Quadruple EMA Touch Strategy
• 정배열시 EMA21(고가) 터치 → 매수
• 역배열시 EMA21(저가) 터치 → 매도

🛡️ <b>리스크 관리</b>
• 기본 손절: {config.base_stop_loss}%
• 보유 시간: 1분 ~ {config.max_holding_hours}시간

⏰ 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        await self.send_message(message)
    
    async def send_multi_tf_alignment_alert(self, alignment_data: Dict):
        """다중 타임프레임 정렬 알림"""
        try:
            bullish_count = alignment_data.get('bullish_count', 0)
            bearish_count = alignment_data.get('bearish_count', 0)
            details = alignment_data.get('details', {})
            
            # 5개 모두 동일한 정렬시에만 알림
            if bullish_count == 5:
                trend = "정배열"
                emoji = "🟢📈"
                signal = "상승 추세 강화"
                color = "초록색"
            elif bearish_count == 5:
                trend = "역배열"
                emoji = "🔴📉"
                signal = "하락 추세 강화"
                color = "빨간색"
            else:
                return  # 5개 모두 동일하지 않으면 알림 안함
            
            message = f"""
{emoji} <b>솔라나(SOL) 다중 타임프레임 {trend} 신호</b>

⚡ <b>{signal}</b>
• {color} 신호: {bullish_count if bullish_count == 5 else bearish_count}/5 타임프레임

📊 <b>타임프레임별 상태</b>"""
            
            # 각 타임프레임 상태 추가
            for tf, data in details.items():
                tf_map = {'1m': '1분', '3m': '3분', '5m': '5분', '15m': '15분', '30m': '30분'}
                tf_name = tf_map.get(tf, tf)
                
                if data.get('is_bullish'):
                    status_emoji = "🟢"
                    status = "정배열"
                else:
                    status_emoji = "🔴"
                    status = "역배열"
                
                # 터치 조건 체크
                touch_mark = ""
                if data.get('touch_high') and data.get('is_bullish'):
                    touch_mark = " ⭐"
                elif data.get('touch_low') and not data.get('is_bullish'):
                    touch_mark = " ⭐"
                
                message += f"\n• {status_emoji} {tf_name:4}: {status}{touch_mark}"
            
            message += f"""

💡 <b>신호 해석</b>
• 5개 모두 {trend} = 매우 강한 추세
• 터치 조건 ⭐ = 진입 신호
• 추세 전환 가능성 {'매우 낮음' if bullish_count == 5 or bearish_count == 5 else '높음'}

⏰ {datetime.now().strftime('%H:%M:%S')}
"""
            
            await self.send_message(message)
            self.logger.info(f"📱 다중 타임프레임 {trend} 알림 전송: {bullish_count if bullish_count == 5 else bearish_count}/5")
            
        except Exception as e:
            self.logger.error(f"다중 타임프레임 알림 오류: {e}")
    
    async def send_shutdown_message(self, stats: Optional[Dict] = None):
        """프로그램 종료 알림"""
        message = f"""
🛑 <b>솔라나(SOL) Futures 자동매매 v4.0 종료</b>

⏰ 종료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        if stats:
            message += f"""
📊 <b>거래 통계</b>
• 총 거래: {stats.get('total_trades', 0)}
• 승률: {stats.get('win_rate', 0):.1f}%
• 총 손익: ${stats.get('total_profit', 0):.2f}
"""
        
        await self.send_message(message)
    
    async def send_position_open(self, side: str, quantity: float, price: float, 
                                leverage: int):
        """포지션 진입 알림"""
        emoji = "🟢" if side == "buy" else "🔴"
        side_kr = "매수" if side == "buy" else "매도"
        
        message = f"""
{emoji} <b>포지션 진입</b>

• 방향: {side_kr}
• 수량: {quantity:.3f} SOL (솔라나)
• 진입가: ${price:.2f}
• 레버리지: {leverage}x

⏰ {datetime.now().strftime('%H:%M:%S')}
"""
        await self.send_message(message)
    
    async def send_position_close(self, side: str, entry_price: float, 
                                 exit_price: float, profit: float, 
                                 roi: float, holding_time: float, reason: str):
        """포지션 청산 알림"""
        emoji = "✅" if profit > 0 else "❌"
        side_kr = "매수" if side == "buy" else "매도"
        
        message = f"""
{emoji} <b>포지션 청산</b>

• 방향: {side_kr}
• 진입가: ${entry_price:.2f}
• 청산가: ${exit_price:.2f}
• 손익: ${profit:.2f} ({roi:+.2f}%)
• 보유시간: {holding_time:.1f}분
• 청산사유: {reason}

⏰ {datetime.now().strftime('%H:%M:%S')}
"""
        await self.send_message(message)
    
    
    async def send_trailing_stop_update(self, trigger_roi: float, new_sl: float, 
                                       entry_price: float):
        """트레일링 스톱 업데이트 알림"""
        sl_roi = ((new_sl - entry_price) / entry_price) * 100
        
        message = f"""
🛡️ <b>트레일링 스톱 업데이트</b>

• 트리거 ROI: {trigger_roi:.0f}%
• 새로운 SL: ${new_sl:.2f} ({sl_roi:+.1f}%)
• 진입가: ${entry_price:.2f}

⏰ {datetime.now().strftime('%H:%M:%S')}
"""
        await self.send_message(message)
    
    
    async def send_stop_loss_alert(self, roi: float, entry_price: float, current_price: float):
        """손절 도달 알림"""
        message = f"""
🔴 <b>손절 도달!</b>

• ROI: {roi:.2f}%
• 진입가: ${entry_price:.2f}
• 현재가: ${current_price:.2f}
• 자동 청산 실행

⏰ {datetime.now().strftime('%H:%M:%S')}
"""
        await self.send_message(message)
    
    async def send_max_holding_alert(self, holding_hours: float):
        """최대 보유시간 도달 알림"""
        message = f"""
⏰ <b>최대 보유시간 도달</b>

• 보유시간: {holding_hours:.1f}시간
• 자동 청산 실행

⏰ {datetime.now().strftime('%H:%M:%S')}
"""
        await self.send_message(message)
    
    async def send_error_alert(self, error_type: str, error_msg: str):
        """오류 알림"""
        message = f"""
⚠️ <b>시스템 오류</b>

• 유형: {error_type}
• 내용: {error_msg}

⏰ {datetime.now().strftime('%H:%M:%S')}
"""
        await self.send_message(message)
    
    async def cleanup(self):
        """리소스 정리"""
        try:
            if self.session:
                await self.session.close()
        except Exception as e:
            self.logger.error(f"텔레그램 정리 오류: {str(e)}")
    
    async def send_manual_position_tracking(self, position: Dict):
        """수동 포지션 추적 시작 알림"""
        side_kr = "매수" if position['side'] == 'buy' else "매도"
        
        message = f"""
🔄 <b>수동 포지션 추적 시작</b>

• 방향: {side_kr}
• 수량: {position['quantity']:.3f} SOL
• 진입가: ${position['entry_price']:.2f}
• 레버리지: {position['leverage']}x

⚠️ 수동으로 진입한 포지션을 자동 추적합니다

⏰ {datetime.now().strftime('%H:%M:%S')}
"""
        await self.send_message(message)
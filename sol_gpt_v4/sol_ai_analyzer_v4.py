#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🤖 SOL Futures 자동매매 시스템 v4.0 - AI Analyzer
OpenAI GPT를 활용한 시장 분석 모듈

전략 개요: Quadruple EMA Touch Strategy
- 이것은 EMA 크로스오버 전략이 아니라 터치 기반 전략입니다!
- 4개의 EMA 사용: EMA10(종가), EMA21(시가/고가/저가)

진입 규칙:
1. 정배열 상태 (EMA10 > EMA21 open):
   - 조건1: EMA10이 EMA21(high)에 0.2% 이내로 터치하면 LONG 진입
   - 조건2: EMA10이 EMA21(high)를 상향 돌파시 LONG 진입 (거리 제한 없음)
   
2. 역배열 상태 (EMA10 < EMA21 open):
   - 조건1: EMA10이 EMA21(low)에 0.2% 이내로 터치하면 SHORT 진입
   - 조건2: EMA10이 EMA21(low)를 하향 돌파시 SHORT 진입 (거리 제한 없음)

중요: EMA가 서로 교차하는 것을 기다리지 않습니다.
단지 특정 레벨에 근접(터치)할 때 진입합니다.
"""

import os
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Tuple
import json
from openai import AsyncOpenAI
import pandas as pd
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

class AIAnalyzer:
    """AI 기반 시장 분석기"""
    
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        
        # OpenAI 클라이언트 초기화
        self.client = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # 캐싱 설정 (비용 절감)
        self.analysis_cache = {}
        self.cache_duration = 60  # 60초 캐시
        
        # AI 모델 설정
        self.model = "gpt-4o"  # GPT-4O 모델 (최신 고성능)
        self.temperature = 0.3  # 일관성 있는 응답
        
        self.logger.info("🤖 AI Analyzer 초기화 완료")
    
    async def analyze_market_condition(self, 
                                     ema_data: Dict, 
                                     price_data: Dict,
                                     position_info: Optional[Dict] = None) -> Dict:
        """시장 상황 종합 분석"""
        try:
            # 캐시 확인
            cache_key = f"{ema_data['ema10']:.2f}_{ema_data['ema21']:.2f}"
            if cache_key in self.analysis_cache:
                cached_time, cached_result = self.analysis_cache[cache_key]
                if (datetime.now() - cached_time).seconds < self.cache_duration:
                    self.logger.info("📦 캐시된 AI 분석 사용")
                    return cached_result
            
            # 분석 프롬프트 생성
            prompt = self._create_analysis_prompt(ema_data, price_data, position_info)
            
            # AI 호출
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=500
            )
            
            # 응답 파싱
            result = self._parse_ai_response(response.choices[0].message.content)
            
            # 캐싱
            self.analysis_cache[cache_key] = (datetime.now(), result)
            
            # 오래된 캐시 정리
            self._clean_old_cache()
            
            return result
            
        except Exception as e:
            self.logger.error(f"AI 분석 오류: {str(e)}")
            return self._get_default_analysis()
    
    async def validate_trade_signal(self, 
                                   signal_type: str,
                                   ema_data: Dict,
                                   market_data: Dict) -> Tuple[bool, str]:
        """Quadruple EMA Touch 신호 검증"""
        try:
            # 터치 거리 계산
            touch_distance_high = abs((ema_data['ema10'] - ema_data.get('ema21_high', 0)) / ema_data.get('ema21_high', 1) * 100)
            touch_distance_low = abs((ema_data['ema10'] - ema_data.get('ema21_low', 0)) / ema_data.get('ema21_low', 1) * 100)
            
            # EMA10이 EMA21_high 또는 EMA21_low를 돌파했는지 확인
            is_above_high = ema_data['ema10'] > ema_data.get('ema21_high', 0)
            is_below_low = ema_data['ema10'] < ema_data.get('ema21_low', 0)
            
            prompt = f"""
            Quadruple EMA Touch 전략 신호 검증:
            
            전략 설명: 4개의 EMA를 사용하는 터치 기반 진입 전략 (크로스오버가 아님!)
            - 정배열 상태(EMA10 > EMA21 open): 
              * 조건1: EMA10이 EMA21(high)에 0.2% 이내로 터치하면 LONG 진입
              * 조건2: EMA10이 EMA21(high)를 상향 돌파시 LONG 진입 (거리 제한 없음)
            - 역배열 상태(EMA10 < EMA21 open): 
              * 조건1: EMA10이 EMA21(low)에 0.2% 이내로 터치하면 SHORT 진입
              * 조건2: EMA10이 EMA21(low)를 하향 돌파시 SHORT 진입 (거리 제한 없음)
            
            ⚠️ 중요: 이것은 EMA 크로스오버 전략이 아닙니다! 
            EMA10이 특정 EMA21 레벨에 터치(근접)할 때 또는 돌파할 때 진입하는 전략입니다
            
            신호 타입: {signal_type}
            EMA10 (종가): ${ema_data['ema10']:.2f}
            EMA21 (시가): ${ema_data['ema21']:.2f}
            EMA21 (고가): ${ema_data.get('ema21_high', 0):.2f}
            EMA21 (저가): ${ema_data.get('ema21_low', 0):.2f}
            현재가: ${market_data['ticker']['last']:.2f}
            
            터치 거리:
            - EMA21(고가)까지: {touch_distance_high:.3f}%
            - EMA21(저가)까지: {touch_distance_low:.3f}%
            - 터치 기준: 0.2% 이내
            
            돌파 상태:
            - EMA10이 EMA21(high) 위에 있음: {'예' if is_above_high else '아니오'}
            - EMA10이 EMA21(low) 아래에 있음: {'예' if is_below_low else '아니오'}
            
            이 신호가 신뢰할 만한지 판단해주세요.
            고려사항:
            1. EMA 배열 상태 (정배열/역배열)
            2. 터치 거리가 0.2% 이내인지 또는 돌파 상태인지
            3. 시장 모멘텀과 신호 방향의 일치성
            4. EMA21의 고가/저가 스프레드
            
            응답 형식:
            {{
                "valid": true/false,
                "confidence": 0-100,
                "reason": "판단 이유"
            }}
            """
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "당신은 Quadruple EMA Touch 전략을 전문으로 하는 기술적 분석 전문가입니다. 이것은 EMA 크로스오버가 아닌 터치 기반 전략입니다. 정배열시 EMA10이 EMA21(high)에 터치하면 LONG, 역배열시 EMA10이 EMA21(low)에 터치하면 SHORT 진입합니다. JSON 형식으로만 응답하세요."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=200
            )
            
            # 응답 내용 확인
            content = response.choices[0].message.content
            if not content:
                self.logger.warning("AI 응답이 비어있음")
                return True, "AI 응답 없음, 기본 신호 사용"
            
            # JSON 파싱 시도 with error handling
            try:
                # JSON 블록이 포함된 경우 추출
                if '```json' in content:
                    content = content.split('```json')[1].split('```')[0].strip()
                elif '```' in content:
                    content = content.split('```')[1].split('```')[0].strip()
                
                # JSON 파싱
                result = json.loads(content)
                
                # 필수 필드 확인
                if 'valid' not in result:
                    self.logger.warning(f"AI 응답에 'valid' 필드 없음: {content}")
                    return True, "AI 응답 형식 오류, 기본 신호 사용"
                
                return result.get('valid', True), result.get('reason', '응답 파싱 성공')
                
            except json.JSONDecodeError as e:
                self.logger.error(f"JSON 파싱 오류: {str(e)}")
                self.logger.error(f"원본 응답: {content}")
                return True, "AI JSON 파싱 실패, 기본 신호 사용"
            
        except Exception as e:
            self.logger.error(f"신호 검증 오류: {str(e)}")
            return True, "AI 검증 실패, 기본 신호 사용"
    
    async def suggest_position_size(self, 
                                  market_volatility: float,
                                  account_balance: float,
                                  current_drawdown: float) -> float:
        """AI 기반 포지션 크기 제안"""
        try:
            prompt = f"""
            포지션 크기 제안:
            
            계정 잔액: ${account_balance:.2f}
            현재 변동성: {market_volatility:.2f}%
            현재 손실: {current_drawdown:.2f}%
            기본 설정: 잔액의 20%
            손절 설정: -20% (백테스트 최적값)
            
            리스크를 고려한 적절한 포지션 크기 비율을 제안해주세요.
            
            응답 형식:
            {{
                "position_size_percent": 5-15,
                "risk_level": "LOW/MEDIUM/HIGH",
                "reason": "이유"
            }}
            """
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "당신은 리스크 관리 전문가입니다. JSON 형식으로만 응답하세요."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=200
            )
            
            content = response.choices[0].message.content
            if not content:
                self.logger.warning("AI 응답이 비어있음")
                return 10.0  # 기본값
            
            # JSON 파싱
            try:
                if '```json' in content:
                    content = content.split('```json')[1].split('```')[0].strip()
                elif '```' in content:
                    content = content.split('```')[1].split('```')[0].strip()
                
                result = json.loads(content)
                return result.get('position_size_percent', 10.0)
            except json.JSONDecodeError as e:
                self.logger.error(f"JSON 파싱 오류: {str(e)}")
                return 10.0  # 기본값
            
        except Exception as e:
            self.logger.error(f"포지션 크기 제안 오류: {str(e)}")
            return 10.0  # 기본값
    
    def _create_analysis_prompt(self, ema_data: Dict, price_data: Dict, position_info: Optional[Dict]) -> str:
        """분석 프롬프트 생성"""
        prompt = f"""
        SOL/USDT 1시간봉 시장 분석 (Quadruple EMA Touch 전략):
        
        현재 가격: ${price_data['ticker']['last']:.2f}
        24시간 변동: {price_data['ticker']['change_24h']:.2f}%
        
        Quadruple EMA 지표:
        - EMA10 (종가): ${ema_data['ema10']:.2f}
        - EMA21 (시가): ${ema_data['ema21']:.2f}
        - EMA21 (고가): ${ema_data.get('ema21_high', 0):.2f}
        - EMA21 (저가): ${ema_data.get('ema21_low', 0):.2f}
        
        EMA 배열: {'정배열' if ema_data['ema10'] > ema_data['ema21'] else '역배열'}
        
        전략 규칙:
        - 정배열 상태: 
          * 조건1: EMA10이 EMA21(high)에 0.2% 이내 터치시 LONG 진입
          * 조건2: EMA10이 EMA21(high)를 상향 돌파시 LONG 진입
        - 역배열 상태: 
          * 조건1: EMA10이 EMA21(low)에 0.2% 이내 터치시 SHORT 진입
          * 조건2: EMA10이 EMA21(low)를 하향 돌파시 SHORT 진입
        - 터치 기준: 0.2% 이내 근접 (돌파시 거리 제한 없음)
        - 손절: -20% (백테스트 최적값)
        - 부분청산: 비활성화 (수익 극대화)
        """
        
        if position_info:
            prompt += f"""
        
        현재 포지션:
        - 방향: {position_info['side']}
        - 진입가: ${position_info['entry_price']:.2f}
        - 수익률: {position_info.get('roi', 0):.2f}%
        """
        
        prompt += """
        
        다음을 분석해주세요:
        1. 현재 시장 추세 (상승/하락/횡보)
        2. 진입 신호 강도 (0-100)
        3. 리스크 수준 (LOW/MEDIUM/HIGH)
        4. 권장 행동 (BUY/SELL/HOLD/CLOSE)
        
        응답 형식:
        {
            "trend": "BULLISH/BEARISH/NEUTRAL",
            "signal_strength": 0-100,
            "risk_level": "LOW/MEDIUM/HIGH",
            "recommendation": "BUY/SELL/HOLD/CLOSE",
            "confidence": 0-100,
            "reason": "분석 근거"
        }
        """
        
        return prompt
    
    def _get_system_prompt(self) -> str:
        """시스템 프롬프트"""
        return """
        당신은 암호화폐 선물 거래 전문가입니다.
        Quadruple EMA Touch 전략을 기반으로 기술적 분석을 수행합니다.
        항상 JSON 형식으로 응답하며, 리스크 관리를 최우선으로 고려합니다.
        
        리스크 관리 설정:
        - 손절: -20% (백테스트 최적값)
        - 부분청산: 비활성화 (백테스트에서 수익 48.8% 제한 확인)
        - 레버리지: 10배
        
        Quadruple EMA Touch 전략 (크로스오버가 아닌 터치 기반!):
        - 4개의 EMA 사용: EMA10(종가), EMA21(시가/고가/저가)
        - 정배열 상태(EMA10 > EMA21 open): 
          * 조건1: EMA10이 EMA21(high)에 0.2% 이내로 터치하면 LONG 진입
          * 조건2: EMA10이 EMA21(high)를 상향 돌파시 LONG 진입 (거리 제한 없음)
        - 역배열 상태(EMA10 < EMA21 open): 
          * 조건1: EMA10이 EMA21(low)에 0.2% 이내로 터치하면 SHORT 진입
          * 조건2: EMA10이 EMA21(low)를 하향 돌파시 SHORT 진입 (거리 제한 없음)
        - 터치 기준: 0.2% 이내 근접 (크로스오버 필요 없음)
        - 이것은 EMA가 교차하는 것을 기다리지 않고, 특정 레벨에 터치할 때 진입하는 전략입니다
        
        중요: EMA10이 EMA21을 넘어서는 크로스오버를 기다리지 마세요. 
        단지 특정 EMA21 레벨(high 또는 low)에 근접하는 것만 확인하세요.
        
        응답은 반드시 요청된 JSON 형식을 따라야 합니다.
        """
    
    def _parse_ai_response(self, response: str) -> Dict:
        """AI 응답 파싱"""
        try:
            # 빈 응답 체크
            if not response:
                self.logger.warning("AI 응답이 비어있음")
                return self._get_default_analysis()
            
            # JSON 블록이 포함된 경우 추출
            if '```json' in response:
                response = response.split('```json')[1].split('```')[0].strip()
            elif '```' in response:
                response = response.split('```')[1].split('```')[0].strip()
            
            # JSON 파싱 시도
            return json.loads(response)
        except json.JSONDecodeError as e:
            # 파싱 실패 시 텍스트에서 추출 시도
            self.logger.warning(f"JSON 파싱 실패: {str(e)}, 기본값 사용")
            self.logger.debug(f"원본 응답: {response[:200]}...")
            return self._get_default_analysis()
    
    def _get_default_analysis(self) -> Dict:
        """기본 분석 결과"""
        return {
            "trend": "NEUTRAL",
            "signal_strength": 50,
            "risk_level": "MEDIUM",
            "recommendation": "HOLD",
            "confidence": 50,
            "reason": "AI 분석 실패, 기본값 사용"
        }
    
    def _clean_old_cache(self):
        """오래된 캐시 정리"""
        current_time = datetime.now()
        keys_to_remove = []
        
        for key, (cache_time, _) in self.analysis_cache.items():
            if (current_time - cache_time).seconds > self.cache_duration * 2:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.analysis_cache[key]
    
    async def analyze_touch_condition(self, ema_data: Dict) -> Dict:
        """EMA 터치 조건 상세 분석"""
        try:
            ema10 = ema_data.get('ema10', 0)
            ema21 = ema_data.get('ema21', 0)
            ema21_high = ema_data.get('ema21_high', 0)
            ema21_low = ema_data.get('ema21_low', 0)
            
            # 터치 거리 계산
            touch_distance_high = abs((ema10 - ema21_high) / ema21_high * 100) if ema21_high else 999
            touch_distance_low = abs((ema10 - ema21_low) / ema21_low * 100) if ema21_low else 999
            
            prompt = f"""
            Quadruple EMA Touch 전략 상태 분석:
            
            현재 EMA 값:
            - EMA10: ${ema10:.2f}
            - EMA21 (open): ${ema21:.2f}
            - EMA21 (high): ${ema21_high:.2f}
            - EMA21 (low): ${ema21_low:.2f}
            
            터치 거리:
            - EMA10과 EMA21(high) 거리: {touch_distance_high:.3f}%
            - EMA10과 EMA21(low) 거리: {touch_distance_low:.3f}%
            
            현재 배열: {'정배열' if ema10 > ema21 else '역배열'}
            
            터치 기준: 0.2% 이내 (돌파시 거리 제한 없음)
            
            질문:
            1. 현재 터치 조건에 얼마나 가까운가?
            2. 정배열/역배열 상태에서 올바른 터치 포인트를 향해 움직이고 있는가?
            3. 터치가 임박했는가?
            
            중요: 이것은 크로스오버가 아니라 터치 전략입니다!
            - 정배열: EMA10이 EMA21(high)에 터치할 때 LONG
            - 역배열: EMA10이 EMA21(low)에 터치할 때 SHORT
            
            응답 형식:
            {{
                "touch_status": "NEAR/FAR/TOUCHED",
                "target_level": "EMA21_HIGH 또는 EMA21_LOW",
                "distance_percent": 숫자,
                "analysis": "상세 분석"
            }}
            """
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "당신은 Quadruple EMA Touch 전략 전문가입니다. 터치 기반 진입 신호를 정확히 이해하고 있습니다."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=200
            )
            
            content = response.choices[0].message.content
            if not content:
                return {
                    "touch_status": "UNKNOWN",
                    "target_level": "UNKNOWN",
                    "distance_percent": 0,
                    "analysis": "응답 없음"
                }
            
            try:
                if '```json' in content:
                    content = content.split('```json')[1].split('```')[0].strip()
                elif '```' in content:
                    content = content.split('```')[1].split('```')[0].strip()
                
                return json.loads(content)
            except json.JSONDecodeError as e:
                self.logger.error(f"JSON 파싱 오류: {str(e)}")
                return {
                    "touch_status": "UNKNOWN",
                    "target_level": "UNKNOWN",
                    "distance_percent": 0,
                    "analysis": "JSON 파싱 실패"
                }
            
        except Exception as e:
            self.logger.error(f"터치 조건 분석 오류: {str(e)}")
            return {
                "touch_status": "UNKNOWN",
                "target_level": "UNKNOWN",
                "distance_percent": 0,
                "analysis": "분석 실패"
            }
    
    async def get_market_sentiment(self, timeframe: str = "1h") -> Dict:
        """시장 심리 분석"""
        try:
            prompt = f"""
            SOL/USDT {timeframe} 시장 심리 분석:
            
            최근 가격 움직임과 거래량을 고려하여 시장 심리를 평가해주세요.
            
            응답 형식:
            {{
                "sentiment": "EXTREME_FEAR/FEAR/NEUTRAL/GREED/EXTREME_GREED",
                "score": 0-100,
                "description": "설명"
            }}
            """
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "당신은 시장 심리 분석 전문가입니다."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=200
            )
            
            content = response.choices[0].message.content
            if not content:
                return {
                    "sentiment": "NEUTRAL",
                    "score": 50,
                    "description": "응답 없음"
                }
            
            try:
                if '```json' in content:
                    content = content.split('```json')[1].split('```')[0].strip()
                elif '```' in content:
                    content = content.split('```')[1].split('```')[0].strip()
                
                return json.loads(content)
            except json.JSONDecodeError as e:
                self.logger.error(f"JSON 파싱 오류: {str(e)}")
                return {
                    "sentiment": "NEUTRAL",
                    "score": 50,
                    "description": "JSON 파싱 실패"
                }
            
        except Exception as e:
            self.logger.error(f"시장 심리 분석 오류: {str(e)}")
            return {
                "sentiment": "NEUTRAL",
                "score": 50,
                "description": "분석 실패"
            }
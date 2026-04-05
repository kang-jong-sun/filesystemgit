#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 SOL Futures 자동매매 시스템 v4.0
Quadruple EMA Touch 기반 자동매매 시스템
"""

import os
import sys
import asyncio
import signal
import traceback
import time
from datetime import datetime
import logging
from pathlib import Path

# .env 파일 자동 로드
def load_env_file(env_file='.env'):
    """
    .env 파일에서 환경 변수를 로드
    """
    env_path = Path(env_file)
    
    if env_path.exists():
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # 따옴표 제거
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    elif value.startswith("'") and value.endswith("'"):
                        value = value[1:-1]
                    
                    # 환경 변수 설정
                    os.environ[key] = value

# .env 파일 로드 시도
load_env_file()

# 환경 변수에서 API 키 확인 (AI 제외)
REQUIRED_KEYS = ['BINANCE_API_KEY', 'BINANCE_API_SECRET']
missing_keys = [key for key in REQUIRED_KEYS if not os.getenv(key) or os.getenv(key).startswith('your_')]

if missing_keys:
    print(" 필수 환경 변수가 설정되지 않았습니다:")
    for key in missing_keys:
        print(f"   - {key}")
    
    if not Path('.env').exists():
        print("\n📝 .env 파일을 생성해주세요:")
        print("1. .env.example 파일을 복사:")
        print("   copy .env.example .env")
        print("\n2. .env 파일을 열어 실제 API 키를 입력하세요.")
    else:
        print("\n📝 .env 파일을 열어 실제 API 키를 입력하세요.")
        print("   파일 위치: " + str(Path('.env').absolute()))
    
    sys.exit(1)

# 컴포넌트 import
try:
    from sol_core_v4 import TradingBot, TradingConfig, setup_logging
    from sol_data_v4 import DataCollector
    from sol_executor_v4 import OrderExecutor
    from sol_monitor_v4 import Monitor
    from sol_telegram_v4 import TelegramBot
    
    print(" 모든 필수 모듈 import 성공")
    
except ImportError as e:
    print(f" 모듈 import 오류: {e}")
    print("필요한 파일들이 같은 디렉토리에 있는지 확인하세요.")
    sys.exit(1)

def print_banner(config):
    """시작 배너 출력"""
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
    tf_display = timeframe_map.get(config.timeframe, config.timeframe)
    
    print("=" * 100)
    print(" SOL Futures 자동매매 시스템 v4.0")
    print("=" * 100)
    print(" 전략: Quadruple EMA Touch")
    print(" - EMA10: 시가(Open) / EMA21: 시가(Open) / EMA21(고가/저가): High/Low")
    print(" - 진입: EMA10이 EMA21(고가/저가)에 터치시")
    print(f" 타임프레임: {tf_display}")
    print(f" 포지션: 계정 {config.position_size}%, 레버리지 {config.leverage}x, 격리마진")
    print(f" 보유시간: {'무제한' if config.max_holding_hours == 0 else f'{config.max_holding_hours}시간'}")
    print("=" * 100)
    print("  실거래 모드 - 실제 자금이 거래됩니다")
    print("=" * 100)

async def initialize_components(config, logger, db):
    """컴포넌트 초기화"""
    components = {}
    
    try:
        # 1. 데이터 수집기
        logger.info(" 데이터 수집기 초기화 중...")
        components['data_collector'] = DataCollector(config, logger)
        await components['data_collector'].initialize()
        
        # 2. 텔레그램 봇 (옵션)
        telegram = None
        if config.telegram_bot_token and config.telegram_chat_id:
            logger.info(" 텔레그램 봇 초기화 중...")
            telegram = TelegramBot(config.telegram_bot_token, config.telegram_chat_id, logger)
            if await telegram.initialize():
                await telegram.send_startup_message(config)
                components['telegram'] = telegram
            else:
                logger.warning("텔레그램 초기화 실패 - 알림 없이 진행")
                telegram = None
        
        # 3. 주문 실행기
        logger.info(" 주문 실행기 초기화 중...")
        components['order_executor'] = OrderExecutor(config, logger, db, telegram)
        await components['order_executor'].initialize()
        
        # 4. 실시간 모니터
        logger.info(" 실시간 모니터 초기화 중...")
        components['monitor'] = Monitor(config, logger)
        
        logger.info(" 모든 컴포넌트 초기화 완료")
        return components
        
    except Exception as e:
        logger.error(f" 컴포넌트 초기화 실패: {str(e)}")
        raise

def setup_component_connections(bot, components):
    """컴포넌트 간 연결 설정"""
    # Bot에 컴포넌트 설정
    bot.data_collector = components['data_collector']
    bot.order_executor = components['order_executor']
    
    # Monitor에 컴포넌트 설정
    components['monitor'].set_components(
        bot,
        components['data_collector'],
        components['order_executor'],
        components.get('telegram')  # 텔레그램 봇 연결
    )

async def safe_shutdown(bot, components, logger):
    """안전한 종료"""
    logger.info(" 시스템 종료 시작...")
    
    try:
        # 봇 정지
        if bot:
            try:
                await bot.stop()
            except Exception as e:
                logger.error(f"봇 정지 오류: {str(e)}")
        
        # 모니터 정지
        if 'monitor' in components:
            try:
                await components['monitor'].stop()
            except Exception as e:
                logger.error(f"모니터 정지 오류: {str(e)}")
        
        # 데이터 수집기 정리
        if 'data_collector' in components:
            try:
                await components['data_collector'].cleanup()
            except Exception as e:
                logger.error(f"데이터 수집기 정리 오류: {str(e)}")
        
        # 주문 실행기 정리
        if 'order_executor' in components and hasattr(components['order_executor'], 'cleanup'):
            try:
                await components['order_executor'].cleanup()
            except Exception as e:
                logger.error(f"주문 실행기 정리 오류: {str(e)}")
        
        # 텔레그램 종료 알림
        if 'telegram' in components:
            try:
                stats = None
                if 'order_executor' in components:
                    stats = components['order_executor'].stats
                await components['telegram'].send_shutdown_message(stats)
                await components['telegram'].cleanup()
            except Exception as e:
                logger.error(f"텔레그램 정리 오류: {str(e)}")
        
        # 잠시 대기하여 모든 비동기 작업이 완료되도록 함
        await asyncio.sleep(0.5)
        
        logger.info(" 모든 컴포넌트 안전하게 종료됨")
        
    except Exception as e:
        logger.error(f"종료 중 오류: {str(e)}")

async def recover_existing_positions(bot, config, logger):
    """기존 포지션 복구"""
    try:
        positions = await bot.order_executor.get_open_positions()
        
        if positions:
            logger.warning(f" 기존 포지션 {len(positions)}개 발견")
            for pos in positions:
                logger.info(f"   - {pos['symbol']} {pos['side']} {pos['quantity']} @ ${pos['entry_price']}")
            
            # 포지션 모니터링 시작
            for pos in positions:
                # DB에서 기존 거래 정보 조회
                partial_closed = False
                breakeven_moved = False
                highest_price = None
                lowest_price = None
                trailing_sl_price = None
                order_id = None
                
                # 진입가와 수량으로 DB에서 최신 거래 찾기 (수량은 근사값 허용)
                cursor = bot.db.cursor()
                cursor.execute('''
                    SELECT order_id, partial_closed, breakeven_moved, highest_price, lowest_price, trailing_sl_price
                    FROM trades 
                    WHERE symbol = ? AND ABS(entry_price - ?) < 0.01 AND ABS(quantity - ?) < 0.001 AND exit_price IS NULL
                    ORDER BY timestamp DESC LIMIT 1
                ''', (config.symbol, pos['entry_price'], pos['quantity']))
                result = cursor.fetchone()
                
                if result:
                    order_id = result[0]
                    partial_closed = result[1] == 1
                    breakeven_moved = result[2] == 1
                    highest_price = result[3]
                    lowest_price = result[4]
                    trailing_sl_price = result[5]
                    
                    logger.info(f"📌 DB에서 기존 거래 정보 복구:")
                    logger.info(f"   - order_id: {order_id}")
                    logger.info(f"   - partial_closed: {partial_closed} (원본값: {result[1]})")
                    if partial_closed:
                        logger.info(f"   - 부분익절 완료")
                    if breakeven_moved:
                        logger.info(f"   - 손익분기점 이동 완료")
                        # 트레일링 업데이트 정보 복구
                        bot.last_trailing_update['breakeven_moved'] = True
                        bot.last_trailing_update['sl_price'] = trailing_sl_price
                    if highest_price:
                        logger.info(f"   - 최고가: ${highest_price:.2f}")
                        bot.last_trailing_update['highest_price'] = highest_price
                    if lowest_price:
                        logger.info(f"   - 최저가: ${lowest_price:.2f}")
                        bot.last_trailing_update['lowest_price'] = lowest_price
                
                # 포지션 정보 보강
                position_info = {
                    **pos,
                    'opened_at': datetime.now(),
                    'is_manual': True,  # 프로그램 시작 전 진입한 포지션은 수동으로 간주
                    'partial_closed': partial_closed,
                    'order_id': order_id if order_id else f"manual_{int(time.time() * 1000)}"
                }
                bot.active_positions[pos['symbol']] = position_info
                
                logger.info(f"📌 position_info 생성 - partial_closed: {position_info.get('partial_closed')}")
                
                # order_executor의 current_position 설정
                # symbol 형식 확인 (SOL/USDT:USDT vs SOL/USDT)
                symbol_match = (
                    pos['symbol'] == config.symbol or 
                    pos['symbol'].startswith(config.symbol.replace('/', '')) or
                    config.symbol in pos['symbol']
                )
                
                if symbol_match:
                    logger.info(f" 포지션을 order_executor에 설정: {pos['symbol']}")
                    logger.info(f"   - position_info partial_closed: {position_info.get('partial_closed')}")
                    bot.order_executor.current_position = position_info
                    # 수동 포지션 추적 시작
                    track_result = await bot.order_executor.track_manual_position(position_info)
                    if not track_result:
                        logger.error("수동 포지션 추적 실패")
                    
                    # 프로그램 재시작 텔레그램 알림
                    if bot.order_executor.telegram:
                        side_kr = '매수' if pos['side'] == 'buy' else '매도'
                        await bot.order_executor.telegram.send_message(
                            f" <b>프로그램 재시작 - 기존 포지션 복구</b>\n\n"
                            f"• 방향: {side_kr}\n"
                            f"• 수량: {pos['quantity']:.3f} SOL\n"
                            f"• 진입가: ${pos['entry_price']:.2f}\n"
                            f"• 현재가: ${pos.get('mark_price', pos['entry_price']):.2f}\n"
                            f"• ROI: {pos.get('percentage', 0):+.2f}%\n"
                            f"• 레버리지: {pos.get('leverage', 10)}x\n\n"
                            f" 기존 포지션을 계속 추적합니다\n\n"
                            f" {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                        )
                else:
                    logger.warning(f" 심볼 불일치: {pos['symbol']} != {config.symbol}")
            
            logger.info(" 기존 포지션 복구 완료")
        else:
            logger.info(" 열린 포지션 없음")
            # 포지션이 없을 때도 텔레그램 알림
            if bot.order_executor.telegram:
                await bot.order_executor.telegram.send_message(
                    f" <b>프로그램 재시작</b>\n\n"
                    f" 현재 열린 포지션이 없습니다\n\n"
                    f" 시스템이 정상적으로 시작되었습니다\n\n"
                    f" {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                )
            
    except Exception as e:
        logger.error(f"포지션 복구 오류: {str(e)}")

async def main():
    """메인 실행 함수"""
    # 설정 및 로깅 초기화
    config = TradingConfig()
    logger, db = setup_logging(config)
    
    # 배너 출력
    print_banner(config)
    
    # 시스템 체크
    logger.info(" 시스템 체크 시작...")
    logger.info(f" 거래 쌍: {config.symbol}")
    logger.info(f" 포지션 크기: 계정의 {config.position_size}%")
    logger.info(f" 레버리지: {config.leverage}x")
    logger.info(f" 손절: {config.base_stop_loss}%")
    logger.info(f" 최대 보유시간: {config.max_holding_hours}시간")
    
    bot = None
    components = {}
    
    try:
        # 컴포넌트 초기화
        components = await initialize_components(config, logger, db)
        
        # 봇 생성
        logger.info(" 트레이딩 봇 생성 중...")
        bot = TradingBot(
            config=config,
            logger=logger,
            db=db
        )
        
        # 컴포넌트 연결
        setup_component_connections(bot, components)
        
        # 기존 포지션 복구
        await recover_existing_positions(bot, config, logger)
        
        # 모니터링 시작
        await components['monitor'].start()
        
        # 봇 실행
        logger.info(" 트레이딩 봇 시작")
        await bot.run()
        
    except KeyboardInterrupt:
        logger.info("  사용자에 의한 중단")
    except Exception as e:
        logger.error(f" 치명적 오류: {str(e)}")
        logger.error(traceback.format_exc())
    finally:
        # 안전한 종료
        await safe_shutdown(bot, components, logger)
        
        # DB 연결 종료
        if db:
            db.close()

if __name__ == "__main__":
    # 종료 플래그
    shutdown_event = None
    
    # 시그널 핸들러 설정
    def signal_handler(sig, frame):
        print("\n 종료 신호 수신...")
        if shutdown_event and not shutdown_event.is_set():
            shutdown_event.set()
    
    # 메인 실행 함수
    async def run_with_shutdown():
        global shutdown_event
        shutdown_event = asyncio.Event()
        
        # 메인 태스크 실행
        main_task = asyncio.create_task(main())
        shutdown_task = asyncio.create_task(shutdown_event.wait())
        
        # 둘 중 하나가 완료되면 종료
        done, pending = await asyncio.wait(
            [main_task, shutdown_task],
            return_when=asyncio.FIRST_COMPLETED
        )
        
        # 남은 태스크 취소
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # 이벤트 루프 실행
    try:
        asyncio.run(run_with_shutdown())
    except KeyboardInterrupt:
        print("\n  프로그램 종료")
    except Exception as e:
        print(f" 프로그램 실행 오류: {e}")
        sys.exit(1)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
로그 파일 자동 순환 모듈
12시간마다 새로운 로그 파일 생성
"""

import os
import logging
from logging.handlers import TimedRotatingFileHandler
from datetime import datetime
from pathlib import Path

class LogRotator:
    """12시간 간격 로그 순환"""
    
    @staticmethod
    def setup_rotating_logger(logger_name: str = "ALTTrading") -> logging.Logger:
        """12시간마다 순환하는 로거 설정"""
        
        # 로그 디렉토리
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # 현재 시간 기반 파일명
        current_time = datetime.now()
        hour_suffix = "AM" if current_time.hour < 12 else "PM"
        base_filename = log_dir / f"alt_trading_{current_time.strftime('%Y%m%d')}_{hour_suffix}.log"
        
        # 로거 설정
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        
        # 기존 핸들러 제거
        logger.handlers.clear()
        
        # TimedRotatingFileHandler 설정 (12시간마다 순환)
        file_handler = TimedRotatingFileHandler(
            filename=base_filename,
            when='H',  # 시간 단위
            interval=12,  # 12시간
            backupCount=14,  # 최대 14개 파일 보관 (7일치)
            encoding='utf-8'
        )
        
        # 파일명 포맷 설정
        file_handler.suffix = "%Y%m%d_%H"
        
        # 포맷터 설정
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # 콘솔 핸들러
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)  # 콘솔은 경고 이상만
        console_handler.setFormatter(formatter)
        
        # 핸들러 추가
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    @staticmethod
    def get_current_log_file() -> str:
        """현재 로그 파일 경로 반환"""
        current_time = datetime.now()
        hour_suffix = "AM" if current_time.hour < 12 else "PM"
        log_dir = Path("logs")
        return str(log_dir / f"alt_trading_{current_time.strftime('%Y%m%d')}_{hour_suffix}.log")
    
    @staticmethod
    def cleanup_old_logs(days_to_keep: int = 7):
        """오래된 로그 파일 정리"""
        log_dir = Path("logs")
        if not log_dir.exists():
            return
        
        current_time = datetime.now()
        
        for log_file in log_dir.glob("alt_trading_*.log*"):
            # 파일 수정 시간 확인
            file_time = datetime.fromtimestamp(log_file.stat().st_mtime)
            age_days = (current_time - file_time).days
            
            if age_days > days_to_keep:
                try:
                    log_file.unlink()
                    print(f"오래된 로그 파일 삭제: {log_file.name}")
                except Exception as e:
                    print(f"로그 파일 삭제 실패: {log_file.name} - {e}")
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
포지션 상태 초기화 스크립트
"""

import sqlite3
import sys

try:
    # 데이터베이스 연결
    conn = sqlite3.connect('alt_trading_bot.db')
    cursor = conn.cursor()
    
    # 포지션 상태 초기화
    cursor.execute("DELETE FROM position_state")
    cursor.execute("INSERT INTO position_state (id, position_data) VALUES (1, NULL)")
    
    conn.commit()
    conn.close()
    
    print("✅ 포지션 상태가 초기화되었습니다.")
    print("프로그램을 재시작하세요.")
    
except Exception as e:
    print(f"❌ 오류 발생: {e}")
    sys.exit(1)
#!/usr/bin/env python3
"""
수동으로 부분익절 상태를 설정하는 유틸리티 스크립트
사용법: python set_partial_closed.py <order_id>
"""

import sqlite3
import sys

def set_partial_closed(order_id):
    """특정 주문의 부분익절 상태를 1로 설정"""
    conn = sqlite3.connect('sol_trading_bot.db')
    cursor = conn.cursor()
    
    # 해당 주문이 존재하는지 확인
    cursor.execute("SELECT * FROM trades WHERE order_id = ? AND exit_price IS NULL", (order_id,))
    result = cursor.fetchone()
    
    if not result:
        print(f"❌ 열린 포지션을 찾을 수 없습니다: order_id = {order_id}")
        return False
    
    # 부분익절 상태 업데이트
    cursor.execute("UPDATE trades SET partial_closed = 1 WHERE order_id = ?", (order_id,))
    conn.commit()
    
    print(f"✅ 부분익절 상태가 설정되었습니다: order_id = {order_id}")
    
    # 업데이트된 정보 표시
    cursor.execute("SELECT symbol, side, quantity, entry_price, partial_closed FROM trades WHERE order_id = ?", (order_id,))
    updated = cursor.fetchone()
    print(f"   - Symbol: {updated[0]}")
    print(f"   - Side: {updated[1]}")
    print(f"   - Quantity: {updated[2]}")
    print(f"   - Entry Price: {updated[3]}")
    print(f"   - Partial Closed: {updated[4]}")
    
    conn.close()
    return True

def list_open_positions():
    """현재 열린 포지션 목록 표시"""
    conn = sqlite3.connect('sol_trading_bot.db')
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT order_id, symbol, side, quantity, entry_price, partial_closed 
        FROM trades 
        WHERE exit_price IS NULL 
        ORDER BY id DESC 
        LIMIT 10
    """)
    
    positions = cursor.fetchall()
    
    if not positions:
        print("열린 포지션이 없습니다.")
        return
    
    print("\n=== 최근 열린 포지션 (최대 10개) ===")
    print(f"{'Order ID':20} {'Symbol':10} {'Side':6} {'Quantity':10} {'Entry':10} {'Partial'}")
    print("-" * 70)
    
    for pos in positions:
        partial = "✅" if pos[5] == 1 else "❌"
        print(f"{pos[0]:20} {pos[1]:10} {pos[2]:6} {pos[3]:10.3f} ${pos[4]:9.2f} {partial}")
    
    conn.close()

if __name__ == "__main__":
    if len(sys.argv) == 2:
        order_id = sys.argv[1]
        set_partial_closed(order_id)
    else:
        print("사용법: python set_partial_closed.py <order_id>")
        print("\n현재 열린 포지션:")
        list_open_positions()
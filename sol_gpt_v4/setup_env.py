#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
환경 변수 설정 도우미
.env 파일을 자동으로 로드하여 환경 변수 설정
"""

import os
import sys
from pathlib import Path

def load_env_file(env_file='.env'):
    """
    .env 파일에서 환경 변수를 로드
    """
    env_path = Path(env_file)
    
    if not env_path.exists():
        print(f"⚠️  {env_file} 파일이 없습니다.")
        print(f"📝 .env.example 파일을 복사하여 {env_file} 파일을 생성하세요.")
        
        # .env.example이 있으면 복사 안내
        if Path('.env.example').exists():
            print("\n다음 명령어를 실행하세요:")
            print(f"copy .env.example {env_file}")
            print(f"\n그 다음 {env_file} 파일을 열어 실제 API 키를 입력하세요.")
        return False
    
    # .env 파일 읽기
    with open(env_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            # 주석이나 빈 줄 무시
            if not line or line.startswith('#'):
                continue
            
            # KEY=VALUE 형식 파싱
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
                
    return True

def check_required_keys():
    """
    필수 API 키 확인
    """
    required_keys = ['BINANCE_API_KEY', 'BINANCE_API_SECRET', 'OPENAI_API_KEY']
    missing_keys = []
    
    for key in required_keys:
        value = os.getenv(key)
        if not value or value.startswith('your_'):
            missing_keys.append(key)
    
    if missing_keys:
        print("❌ 다음 API 키를 설정해주세요:")
        for key in missing_keys:
            print(f"   - {key}")
        print("\n.env 파일을 열어 실제 API 키를 입력하세요.")
        return False
    
    return True

def setup_environment():
    """
    환경 설정 메인 함수
    """
    print("🔧 환경 변수 설정 중...")
    
    # .env 파일 로드
    if not load_env_file():
        return False
    
    # 필수 키 확인
    if not check_required_keys():
        return False
    
    print("✅ 환경 변수 설정 완료!")
    return True

if __name__ == "__main__":
    if setup_environment():
        print("\n이제 다음 명령어로 프로그램을 실행할 수 있습니다:")
        print("python sol_ai_main_v4.py")
    else:
        sys.exit(1)
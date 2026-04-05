## ETH/USDT 선물 백테스트 — GPU/CUDA 가속 규칙

### 환경
- **GPU**: NVIDIA GeForce RTX 4060 Ti 8GB
- **CUDA Toolkit**: 12.6 (`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6`)
- **Numba**: 0.64.0 (JIT + CUDA 지원)
- **CuPy**: 14.0.1
- **Python**: 3.12

### 백테스트 최적화 시 필수 규칙
1. **대량 파라미터 최적화** (1,000개 이상 조합)는 반드시 **Numba CUDA GPU**로 실행할 것
2. **단건 백테스트**는 **Numba @njit** (CPU JIT)로 실행할 것 (0.001초/건)
3. **실시간 트레이딩 봇**은 기존 Python 그대로
4. GPU 사용 시 **배치 크기 1,500** (8GB VRAM 기준), 초과 시 자동 분할

### 검증된 CUDA 백테스트 엔진 (이 디렉토리)
- `eth_v7_3_numba.py` — Numba JIT 버전 (CPU, 460배 가속, 검증 완료)
- `eth_v7_3_cuda.py` — CUDA GPU 버전 (417 jobs/s, 검증 완료)
- 기존 원본: `eth_v7_3_gpt_collab.py` (수정 안 함)

### 새 백테스트/최적화 스크립트 작성 시
1. `backtest()` 함수는 **Numba @njit** 데코레이터 적용
2. dict.get() 대신 **개별 float 인자**로 전달
3. 문자열/Boolean → **숫자 코드**로 변환 (예: dual_mode='BOTH' → 1.0)
4. 대량 최적화는 `numba.cuda` 커널로 작성, `run_cuda_batch()` 패턴 사용
5. 결과 검증: 기존 Python 버전과 **1:1 비교** 후 사용

### 기존 백테스트 엔진 구조
| 엔진 파일 | 주요 함수 | 사용 스크립트 |
|-----------|----------|-------------|
| `fast_backtest.py` | `run_fast()`, `fast_bt()` | `run_fast_opt.py`, `run_fixed1000_opt.py` |
| `eth_v6_backtest.py` | `v6_backtest()`, `compute_all()` | `eth_v7_fixed1000.py`, `eth_v7_optimize.py` |
| `eth_v7_backtest.py` | `backtest()` | `eth_v7_3_gpt_collab.py` |
| `backtest_engine.py` | 지표 유틸리티 | 여러 스크립트 |

### 성능 비교 (검증 완료)
| 방식 | 1건 속도 | 64,800건 | CPU 부하 |
|------|---------|---------|---------|
| 원본 Python | 0.461초 | ~50분 | 96% |
| Numba JIT | 0.001초 | ~7초 | 96% |
| CUDA GPU | 0.002초/건 | ~2.5분 | ~5% |


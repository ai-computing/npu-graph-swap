# npu-graph-swap

**가중치를 올려 둔 채로 NPU가 컴파일된 그래프(시퀀스 길이·배치)를 바꿔 쓸 수 있는가?**

[English](README.md) · 한국어

> 상태: **가능성 검증.** 실험만 있으며 아직 라이브러리가 아닙니다.

## 왜

NPU는 길이 구간·배치마다 미리 컴파일한 그래프로 동작합니다. 부하가 바뀌는 서버는 그래프를 바꾸고 싶어 합니다: 한가할
때는 배치 1, 요청이 밀리면 배치 8, 짧은 프롬프트에는 짧은 길이 그래프. GPU는 같은 가중치 위에 배치별 CUDA graph를
잡아 두는 방식으로 이미 이렇게 합니다.

[onggi](https://github.com/ai-computing/onggi/blob/main/docs/ko/artifact-formats.md) 측정은 NPU에서도 가능할 것임을
보여줍니다.

- Rebellions ATOM의 decode 빌드는 길이·배치가 달라도 바이트의 **98.5–99.7%**를 공유하며, 약 15 GB 가중치 위에 변형마다
  50–190 MB만 다릅니다. 가중치 배열은 같고 프로그램만 다릅니다.
- FuriosaAI RNGD는 이미 그래프(`.edf`, KB–MB)를 가중치와 분리해 두고, furiosa-llm이 가중치를 공유하는 여러 bucket 중에서
  고릅니다.

그러나 지금 ATOM에서 따로 컴파일한 변형은 각자 가중치를 통째로 품고 있어, 바꾸려면 약 14 GB를 다시 올려야 하고 두 변형을
동시에 올려 두면 장치 메모리가 두 배가 됩니다.

## 질문

| # | 질문 | 장치 |
|---|---|---|
| 1 | decoder 배치 여러 개(와 prefill)를 한 번에 컴파일하면 장치에서 가중치 한 벌을 공유하는가? 메모리, 그래프 간 전환 지연, 따로 컴파일한 경우 대비 step 시간 | ATOM |
| 2 | furiosa-llm은 bucket 사이를 얼마나 빨리 바꾸며, bucket 수에 따라 장치 메모리는 어떻게 느는가? | RNGD |
| 3 | 따로 컴파일한 그래프를 이미 장치에 올라간 가중치에 붙일 수 있는가(런타임 API, 가중치 바인딩, 공유 버퍼)? | ATOM, RNGD |
| 4 | 기준선: 변형 하나를 통째로 다시 올리는 시간(적재 + 첫 step) | ATOM, RNGD |
| 5 | 1–3이 가능하면: 부하 기반 그래프 선택기(이력 현상, 웜 풀, 프리페치)와 처리량·SLO 효과 | 둘 다 |

질문마다 가설을 담은 Issue가 있고, 실행마다 `experiments/`에 기록을 남깁니다.

[공통 교체 벤치마크](experiments/README.md#common-replacement-benchmark-issue-12)는 ATOM·RNGD에서
전체 복원, Onggi delta 복원, 지원되는 상주 그래프 선택을 같은 절차로 비교합니다.
파일 복원·런타임 로딩·첫 정상 응답 시간을 나누며, 미지원 경로와 정확성 실패도 기록합니다.
파일 delta 복원이 장치 메모리의 부분 로딩을 뜻하지는 않습니다.

## 결과의 행방

- 가능 → 엔진용 명세: [Optimus Prime](https://github.com/ai-computing/aicomp)에 그래프 전환 기능 요청(이 저장소는 엔진·벤더
  코드를 고치지 않음), onggi에는 그래프 차이 메타데이터
- 불가능 → 벤더 런타임 요구사항과 대안별 측정 비용을 정리

## 관련

[ai-rm](https://github.com/ai-computing/ai-rm) · [onggi](https://github.com/ai-computing/onggi) ·
[rngd-pd](https://github.com/ai-computing/rngd-pd) · [npu-train](https://github.com/ai-computing/npu-train)

라이선스: [Apache-2.0](LICENSE).

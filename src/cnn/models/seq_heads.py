"""순서를 보는 헤드를 붙인 변형들 — "그룹 구조를 안 쓰는 게 풀링 탓인가"를 가르기 위한 통제.

simple_cnn 은 합성곱 몸통 뒤에 AdaptiveAvgPool2d(1) 로 시간축 전체를 평균한다. 이 연산은
설계상 순서를 지우므로, 모델이 자릿수 그룹 경계를 결정에 쓰지 못하는 것이 (a) 표현의 한계인지
(b) 풀링이 지운 것인지 구분할 수 없다. 아래 세 변형은 그 사다리를 만든다.

  simple_cnn      균등 평균          순서 무시                시간 해상도 T'=9
  attn_cnn        가중 평균(학습)     순서 무시                시간 해상도 T'=9
  gru_cnn         BiGRU + 가중 평균   순서 인지                시간 해상도 T'=9
  gru_cnn_hires   BiGRU + 가중 평균   순서 인지                시간 해상도 T'=39

attn_cnn 은 "평균이 희석해서 못 쓴 것"이라는 가설을 분리한다. 어텐션은 가중치를 학습하지만
여전히 치환 불변이므로, attn_cnn 이 그룹 구조를 쓰기 시작하면 원인은 순서가 아니라 희석이다.

시간 해상도: hop 512 표본 = 32 ms 이므로 T'=9 는 스텝 간격 512 ms, T'=39 는 128 ms 이다.
자릿수 그룹 사이 휴지가 200--300 ms 이므로 T'=9 에서는 휴지가 스텝 하나에 못 미쳐 묻힐 수 있다.
gru_cnn_hires 는 그 가능성을 배제하기 위해 뒤 두 블록의 시간축 풀링을 없앤 것이다.
"""
import torch
import torch.nn as nn

# 채널 구성은 simple_cnn 과 동일하게 두어 몸통의 용량을 맞춘다.
_CHANNELS = [16, 32, 64, 64]


def _trunk(in_ch, time_strides):
    """simple_cnn 과 같은 4블록. MaxPool 커널은 (주파수, 시간) 순서이다."""
    layers, prev = [], in_ch
    for out, ts in zip(_CHANNELS, time_strides):
        layers += [
            nn.Conv2d(prev, out, 3, padding=1),
            nn.BatchNorm2d(out),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, ts)),
        ]
        prev = out
    return nn.Sequential(*layers)


class _AttnPool(nn.Module):
    """시간축에 대한 학습된 가중 평균. 균등 평균(GAP)의 최소 확장이다."""

    def __init__(self, dim):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, h):                      # h: [B, T, D]
        w = torch.softmax(self.score(h), dim=1)   # [B, T, 1]
        return (w * h).sum(dim=1)                 # [B, D]


class SeqHeadCNN(nn.Module):
    """합성곱 몸통 → 주파수축 평균 → (선택) BiGRU → 어텐션 풀링 → 선형 분류기."""

    def __init__(self, in_ch=1, n_classes=2, use_gru=False, hires=False, gru_hidden=32):
        super().__init__()
        time_strides = (2, 2, 1, 1) if hires else (2, 2, 2, 2)
        self.features = _trunk(in_ch, time_strides)
        dim = _CHANNELS[-1]
        # BiGRU 는 양방향 hidden 을 이어 붙이므로 출력 차원을 dim 으로 맞춘다.
        self.gru = nn.GRU(dim, gru_hidden, batch_first=True, bidirectional=True) if use_gru else None
        pooled = gru_hidden * 2 if use_gru else dim
        self.pool = _AttnPool(pooled)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(pooled, n_classes))

    def forward(self, x):
        z = self.features(x)                   # [B, C, F', T']
        z = z.mean(dim=2)                      # 주파수축 평균 → [B, C, T']
        z = z.transpose(1, 2)                  # [B, T', C]
        if self.gru is not None:
            z, _ = self.gru(z)                 # [B, T', 2*hidden]
        return self.head(self.pool(z))


def attn_cnn(in_ch=1):
    return SeqHeadCNN(in_ch, use_gru=False, hires=False)


def gru_cnn(in_ch=1):
    return SeqHeadCNN(in_ch, use_gru=True, hires=False)


def gru_cnn_hires(in_ch=1):
    return SeqHeadCNN(in_ch, use_gru=True, hires=True)

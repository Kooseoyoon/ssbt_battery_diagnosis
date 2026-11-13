from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib import font_manager


OUTPUT = Path(__file__).with_name("pinn_poster_2026-11-13.pdf")
FONT_CANDIDATES = ("Malgun Gothic", "AppleGothic", "Noto Sans CJK KR", "NanumGothic")
FONT_NAME = None
for font_name in FONT_CANDIDATES:
    try:
        font_manager.findfont(font_name, fallback_to_default=False)
        FONT_NAME = font_name
        break
    except ValueError:
        continue
if FONT_NAME is None:
    raise RuntimeError("Install a Korean font such as Malgun Gothic or AppleGothic.")
plt.rcParams.update(
    {
        "font.family": FONT_NAME,
        "font.size": 10,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": False,
    }
)

NAVY = "#13243A"
INK = "#203247"
MUTED = "#5D6B7A"
CANVAS = "#F3F6FA"
WHITE = "#FFFFFF"
TEAL = "#148A80"
TEAL_LIGHT = "#E5F5F2"
BLUE = "#5276B8"
BLUE_LIGHT = "#EDF2FB"
GOLD = "#D99519"
GOLD_LIGHT = "#FFF4D9"
GRAY = "#8793A1"
LINE = "#DCE3EA"

fig = plt.figure(figsize=(16.54, 11.69), facecolor=CANVAS)


def text(x, y, value, size=10, color=INK, weight="normal", ha="left", va="top", **kwargs):
    fig.text(
        x,
        y,
        value,
        fontsize=size,
        color=color,
        fontweight=weight,
        ha=ha,
        va=va,
        **kwargs,
    )


def card(x, y, width, height, face=WHITE, edge=None, radius=0.012, lw=0.8):
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0.008,rounding_size={radius}",
        transform=fig.transFigure,
        facecolor=face,
        edgecolor=edge or face,
        linewidth=lw,
        zorder=0,
    )
    fig.patches.append(patch)
    return patch


# Header
text(0.045, 0.968, "배터리 재사용 · 연구 포스터  |  2026.11.13", 10, TEAL, "bold")
text(0.045, 0.925, "구성방정식 PINN, 실제 팩 가치를 높였을까?", 25, NAVY, "bold")
text(
    0.045,
    0.883,
    "일반 신경망보다 V는 9.3% 높았지만, 간단한 Current-only 기준선은 아직 넘지 못했다.",
    13,
    INK,
)
card(0.777, 0.895, 0.18, 0.055, NAVY, radius=0.015)
text(0.867, 0.922, "실제 86셀 데이터", 12, WHITE, "bold", ha="center", va="center")

# Three content panels
panel_y, panel_h = 0.205, 0.635
card(0.035, panel_y, 0.295, panel_h)
card(0.345, panel_y, 0.335, panel_h)
card(0.695, panel_y, 0.270, panel_h)

# Left: research context and previous handoff
text(0.055, 0.811, "01  연구 질문과 실험 조건", 14, NAVY, "bold")
text(
    0.055,
    0.766,
    "적은 진단으로 셀을 잘 묶어,\n팩의 실제 가치를 높일 수 있을까?",
    11,
    INK,
    "bold",
)
text(
    0.055,
    0.710,
    "이전 물리제약 NN은 추세를 부드럽게 했습니다.\n"
    "이번 PINN은 열화식이 요구하는 변화율도\n"
    "학습에 반영해 두 방식을 구분합니다.",
    9.5,
    MUTED,
    linespacing=1.5,
)

card(0.055, 0.550, 0.255, 0.090, BLUE_LIGHT, radius=0.010)
text(0.070, 0.622, "팩 가치 V (Ah·사이클)", 10, BLUE, "bold")
text(
    0.070,
    0.593,
    "각 팩의 가장 낮은 실제 용량을\n사용 종료까지 누적한 값",
    10,
    INK,
    linespacing=1.4,
)

# Study setup chips
chips = [(0.055, "86개 셀"), (0.142, "8직렬 × 10팩"), (0.238, "6개 미사용")]
for x, label in chips:
    width = 0.074 if x < 0.14 else (0.087 if x < 0.23 else 0.072)
    card(x, 0.496, width, 0.038, "#F5F7F9", edge=LINE, radius=0.008, lw=0.6)
    text(x + width / 2, 0.515, label, 8.3, INK, "bold", ha="center", va="center")

text(0.055, 0.462, "진단 10사이클  ·  평가 입력은 사이클 1·6만", 9.2, INK, "bold")
text(
    0.055,
    0.451,
    "사이클 11 이후 실제 용량으로 V를 계산합니다.\n"
    "미래값은 예측·배정이 아니라 사후 평가에만 사용.",
    8.4,
    MUTED,
    linespacing=1.3,
)

card(0.055, 0.215, 0.255, 0.198, GOLD_LIGHT, radius=0.010)
text(0.070, 0.395, "이번 PINN에서 쓴 식과 λ", 9.1, GOLD, "bold")
text(
    0.070,
    0.370,
    "x=(N-6)/5, eps=0.05 RPT; Q(0)=Q6\n"
    "sqrt+선형: Q=Q6-ks*(sqrt(x+eps)-sqrt(eps))-kl*x\n"
    "지수+선형: Q=Q6-ke*(1-exp(-x/tau))-kl*x\n"
    "계수는 양수, tau>=0.1 RPT. r(Ah/RPT)은\n"
    "예측 변화율과 식이 요구하는 변화율의 차이입니다.\n"
    "L=Huber((Q_pred-Q)/0.05Ah)+lambda*\n"
    "mean[(r/(0.02Ah/RPT))^2]\n"
    "lambda=.01/.1/1은 약·중·강 제약 (100배 범위).\n"
    "물리식 잔차의 상대 비중이지, 퍼센트나 팩 V 보상이 아닙니다.",
    7.6,
    INK,
    linespacing=1.18,
)

# Center: comparison chart
text(0.365, 0.811, "02  오늘의 비교: 실제 V", 14, NAVY, "bold")
text(0.365, 0.777, "Current-only를 100으로 환산한 값  |  막대 끝은 실제 V", 9, MUTED)

ax = fig.add_axes([0.405, 0.303, 0.255, 0.430])
ax.set_zorder(2)
labels = [
    "Current-only",
    "일반 NN (같은 실행)",
    "√ + 선형  λ=0.01",
    "√ + 선형  λ=0.1",
    "√ + 선형  λ=1",
    "지수 + 선형  λ=0.01",
    "지수 + 선형  λ=0.1",
    "지수 + 선형  λ=1",
]
indices = [100.0, 90.8, 98.3, 92.4, 99.2, 89.8, 95.9, 90.1]
values = [3374.804, 3064.715, 3318.358, 3116.915, 3348.210, 3029.731, 3235.051, 3040.329]
colors = [NAVY, GRAY, TEAL, TEAL, TEAL, BLUE, BLUE, BLUE]
ypos = list(range(len(labels)))[::-1]
ax.barh(ypos, [value - 85 for value in indices], left=85, height=0.61, color=colors, zorder=3)
ax.set_yticks(ypos, labels=labels, fontsize=8.7, color=INK)
ax.set_xlim(85, 103.5)
ax.set_xticks([85, 90, 95, 100])
ax.tick_params(axis="x", labelsize=8, colors=MUTED, length=0, pad=5)
ax.tick_params(axis="y", length=0, pad=7)
ax.axvline(100, color=GOLD, linewidth=1.4, linestyle=(0, (3, 2)), zorder=2)
ax.grid(axis="x", color=LINE, linewidth=0.7, zorder=0)
for y, score, value in zip(ypos, indices, values):
    ax.text(
        score + 0.22,
        y,
        f"{score:.1f}  |  {value:,.1f}",
        va="center",
        ha="left",
        fontsize=7.6,
        color=INK,
        fontweight="bold",
    )
for spine in ax.spines.values():
    spine.set_visible(False)
ax.set_xlabel("상대 V 지수 (Current-only = 100)", fontsize=8.5, color=MUTED, labelpad=6)

text(
    0.365,
    0.272,
    "두 식은 경험적 열화 추세 후보입니다. 제곱근식은\n"
    "초기에 빠르게, 포화 지수식은 초기에 감소한 뒤\n"
    "완만해지는 형태를 가정합니다. λ는 방정식 잔차\n"
    "가중치이며, 6개 후보를 모두 보고했습니다.",
    8.3,
    MUTED,
    linespacing=1.38,
)

# Right: key outcome and trade-offs
text(0.715, 0.811, "03  숫자의 해석", 14, NAVY, "bold")
card(0.715, 0.674, 0.230, 0.093, TEAL_LIGHT, radius=0.012)
text(0.733, 0.746, "같은 실행의 일반 NN 대비", 9, TEAL, "bold")
text(0.733, 0.711, "+9.3%", 22, TEAL, "bold")
text(0.846, 0.721, "V가 상승\n3,064.7 → 3,348.2", 9, INK, linespacing=1.4)

card(0.715, 0.558, 0.230, 0.093, GOLD_LIGHT, radius=0.012)
text(0.733, 0.630, "Current-only 기준선 대비", 9, GOLD, "bold")
text(0.733, 0.595, "-0.8%", 22, GOLD, "bold")
text(0.846, 0.605, "아직 낮음\n3,374.8 → 3,348.2", 9, INK, linespacing=1.4)

text(0.715, 0.518, "오차 지표는 한 방향으로 좋아지지 않았습니다.", 9.1, NAVY, "bold")
text(
    0.715,
    0.489,
    "· 제곱근 PINN(λ=1) 궤적 오차: 0.0644 Ah\n"
    "  일반 NN: 0.0629 Ah (NN이 약간 작음)\n"
    "· RUL 오차 하한: 454.6 vs 454.9사이클\n"
    "  (거의 같고, 86셀 중 16·18개는 검열됨)\n"
    "· 가장 작은 궤적 오차는 지수 PINN(λ=1)\n"
    "  0.0613 Ah지만 상대 V는 90.1",
    8.5,
    INK,
    linespacing=1.43,
)
card(0.715, 0.253, 0.230, 0.073, NAVY, radius=0.010)
text(
    0.830,
    0.290,
    "결론: NN 대비 유망 신호는 있지만\n실제 기준선 초과 효용은 아직 입증 전",
    9.4,
    WHITE,
    "bold",
    ha="center",
    va="center",
    linespacing=1.35,
)

# Footer: limitations and next steps
card(0.035, 0.045, 0.930, 0.128, "#E8EDF3", radius=0.012)
text(0.055, 0.151, "한계와 다음 단계", 10.2, NAVY, "bold")
text(
    0.055,
    0.124,
    "86셀 OOF 결과는 같은 재고에서의 탐색 비교이지, 독립된 새 재고 검증이 아닙니다. "
    "100점은 Current-only 대비 비율이지 최적 정답(oracle)이 아닙니다. 두 식은 검증된 "
    "전기화학 법칙이 아니며, 관측 회복을 강제로 지우지는 않았지만 별도 회복항도 아직 없습니다.",
    8.1,
    INK,
    linespacing=1.4,
)
text(
    0.055,
    0.070,
    "다음: 회복항을 포함한 식의 근거를 검토하고, 후보·가중치를 사전에 고정한 뒤 독립 셀 재고에서 "
    "재검증합니다. Stage 2 특징 파일은 이번 PINN에 포함되지 않았습니다.",
    8.1,
    MUTED,
    linespacing=1.35,
)
text(
    0.952,
    0.022,
    "자료: C_full 실제 86셀 · 8직렬 · 10진단사이클 · OOF 탐색",
    7.3,
    MUTED,
    ha="right",
    va="bottom",
)

fig.savefig(
    OUTPUT,
    format="pdf",
    bbox_inches="tight",
    pad_inches=0.06,
    metadata={
        "Title": "배터리 구성방정식 PINN 연구 포스터",
        "Author": "Battery reuse research",
        "Subject": "실제 86셀 데이터에서 PINN의 팩 가치 비교",
    },
)
print(OUTPUT)
print(f"bytes={OUTPUT.stat().st_size}")

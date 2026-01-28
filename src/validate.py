import pandas as pd
from sklearn.metrics import f1_score

# 경로
ANS_PATH = "data/test/test_ans.csv"
PRED_PATH = "data/test/test_set.csv"

# CSV 로드
df_ans = pd.read_csv(ANS_PATH)
df_pred = pd.read_csv(PRED_PATH)

# id 기준으로 병합 (정답 / 예측 정렬 보장)
df = df_ans.merge(
    df_pred,
    on="id",
    suffixes=("_true", "_pred")
)

# F1 score 계산 (binary classification 기준)
f1 = f1_score(
    df["label_true"],
    df["label_pred"]
)

print(f"F1 Score: {f1:.4f}")

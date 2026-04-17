import os
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt

from pandas.plotting import scatter_matrix
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error


# 결과 이미지 저장 폴더
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# 1. 데이터 불러오기
fires = pd.read_csv("./sanbul2district-divby100.csv", sep=",")

print("\n==============================")
print("1-1 Data 불러오기 완료")
print("==============================")
print(fires.head())


# 2. 기본 정보 출력
print("\n==============================")
print("1-2 fires.head()")
print("==============================")
print(fires.head())

print("\n==============================")
print("1-2 fires.info()")
print("==============================")
fires.info()

print("\n==============================")
print("1-2 fires.describe()")
print("==============================")
print(fires.describe())

print("\n==============================")
print("1-2 month value_counts()")
print("==============================")
print(fires["month"].value_counts())

print("\n==============================")
print("1-2 day value_counts()")
print("==============================")
print(fires["day"].value_counts())


# 3. 데이터 시각화 - 히스토그램 저장
print("\n==============================")
print("1-3 히스토그램 저장")
print("==============================")
fires.hist(figsize=(12, 8), bins=20)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "01_histograms.png"))
plt.close()


# 4. burned_area 로그 변환 비교 저장
print("\n==============================")
print("1-4 burned_area 로그 변환")
print("==============================")

fires_before_log = fires.copy()

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
fires_before_log["burned_area"].hist(bins=30)
plt.title("Before log transform")

fires["burned_area"] = np.log(fires["burned_area"] + 1)

plt.subplot(1, 2, 2)
fires["burned_area"].hist(bins=30)
plt.title("After log transform")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "02_log_transform_comparison.png"))
plt.close()


# 5. train_test_split
print("\n==============================")
print("1-5 train_test_split")
print("==============================")
train_set, test_set = train_test_split(fires, test_size=0.2, random_state=42)

print("전체 데이터 개수:", len(fires))
print("훈련 세트 개수:", len(train_set))
print("테스트 세트 개수:", len(test_set))
print("테스트 세트 비율:", len(test_set) / len(fires))


# 6. month 기준 계층 분할
print("\n==============================")
print("1-5 StratifiedShuffleSplit")
print("==============================")
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(fires, fires["month"]):
    strat_train_set = fires.loc[train_index].copy()
    strat_test_set = fires.loc[test_index].copy()

print("\nMonth category proportion in stratified test set")
print(strat_test_set["month"].value_counts() / len(strat_test_set))

print("\nOverall month category proportion")
print(fires["month"].value_counts() / len(fires))


# 7. scatter_matrix 저장
print("\n==============================")
print("1-6 scatter_matrix 저장")
print("==============================")
attributes = ["burned_area", "max_temp", "avg_temp", "max_wind_speed", "avg_wind"]
scatter_matrix(fires[attributes], figsize=(12, 8))
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "03_scatter_matrix.png"))
plt.close()


# 8. 지역별 burned_area 산점도 저장
print("\n==============================")
print("1-7 지역별 burned_area 산점도 저장")
print("==============================")
ax = fires.plot(
    kind="scatter",
    x="longitude",
    y="latitude",
    alpha=0.4,
    s=fires["max_temp"] * 10,
    label="max_temp",
    c="burned_area",
    cmap=plt.get_cmap("jet"),
    colorbar=True,
    figsize=(10, 7)
)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "04_regional_burned_area_plot.png"))
plt.close()


# 9. 학습용 데이터 / 라벨 분리
fires_data = strat_train_set.drop("burned_area", axis=1)
fires_labels = strat_train_set["burned_area"].copy()
fires_num = fires_data.drop(["month", "day"], axis=1)

print("\n==============================")
print("1-8 수치형 데이터 일부 출력")
print("==============================")
print(fires_num.head())


# 10. OneHotEncoder
print("\n==============================")
print("1-8 OneHotEncoder")
print("==============================")
cat_encoder = OneHotEncoder(handle_unknown="ignore")
fires_cat = fires_data[["month", "day"]]
fires_cat_1hot = cat_encoder.fit_transform(fires_cat)

print("OneHot 인코딩 결과 shape:", fires_cat_1hot.shape)
print("\nmonth/day categories:")
print(cat_encoder.categories_)


# 11. Pipeline + StandardScaler + ColumnTransformer
print("\n==============================")
print("1-9 Pipeline + StandardScaler + ColumnTransformer")
print("==============================")
num_attribs = ["longitude", "latitude", "avg_temp", "max_temp", "max_wind_speed", "avg_wind"]
cat_attribs = ["month", "day"]

num_pipeline = Pipeline([
    ("scaler", StandardScaler())
])

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_attribs)
])

fires_prepared = full_pipeline.fit_transform(fires_data)

if hasattr(fires_prepared, "toarray"):
    fires_prepared_dense = fires_prepared.toarray()
else:
    fires_prepared_dense = fires_prepared

print("전처리 완료된 데이터 shape:", fires_prepared_dense.shape)
print("전처리된 데이터 일부:")
print(np.round(fires_prepared_dense[:5], 2))


# 12. 테스트 데이터 전처리
X_test_df = strat_test_set.drop("burned_area", axis=1)
y_test = strat_test_set["burned_area"].copy()

X_test_prepared = full_pipeline.transform(X_test_df)

if hasattr(X_test_prepared, "toarray"):
    X_test_prepared = X_test_prepared.toarray()


# 13. train / validation 분리
X_train, X_valid, y_train, y_valid = train_test_split(
    fires_prepared_dense,
    fires_labels,
    test_size=0.2,
    random_state=42
)

print("\n==============================")
print("train / validation 분리")
print("==============================")
print("X_train shape:", X_train.shape)
print("X_valid shape:", X_valid.shape)


# 14. 시드 고정
np.random.seed(42)
tf.random.set_seed(42)


# 15. MLP 회귀 모델 생성
print("\n==============================")
print("2단계 MLP 모델 생성")
print("==============================")
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(1)
])

model.summary()

model.compile(
    loss="mean_squared_error",
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=["mae"]
)


# 16. 모델 학습
history = model.fit(
    X_train,
    y_train,
    epochs=100,
    validation_data=(X_valid, y_valid),
    verbose=1
)


# 17. 학습 손실 그래프 저장
print("\n==============================")
print("학습 결과 그래프 저장")
print("==============================")
plt.figure(figsize=(10, 5))
plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "05_training_validation_loss.png"))
plt.close()


# 18. 모델 평가
print("\n==============================")
print("모델 평가")
print("==============================")
test_loss, test_mae = model.evaluate(X_test_prepared, y_test, verbose=0)
test_pred = model.predict(X_test_prepared, verbose=0).reshape(-1)

rmse = np.sqrt(mean_squared_error(y_test, test_pred))
mae = mean_absolute_error(y_test, test_pred)

print("테스트 MSE:", round(float(test_loss), 4))
print("테스트 MAE:", round(float(test_mae), 4))
print("RMSE:", round(float(rmse), 4))
print("MAE:", round(float(mae), 4))

print("\n샘플 예측값 5개")
print("예측:", np.round(test_pred[:5], 2))
print("실제:", np.round(y_test.iloc[:5].values, 2))


# 19. 모델 / 전처리기 저장
model.save("fires_model.keras")
joblib.dump(full_pipeline, "preprocess_pipeline.pkl")

print("\n==============================")
print("저장 완료")
print("==============================")
print("fires_model.keras")
print("preprocess_pipeline.pkl")

print("\n==============================")
print("저장된 그래프 파일")
print("==============================")
print("outputs/01_histograms.png")
print("outputs/02_log_transform_comparison.png")
print("outputs/03_scatter_matrix.png")
print("outputs/04_regional_burned_area_plot.png")
print("outputs/05_training_validation_loss.png")
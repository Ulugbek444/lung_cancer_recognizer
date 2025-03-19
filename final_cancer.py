import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, roc_auc_score
import joblib
# Загрузка и подготовка данных
df = pd.read_csv("survey lung cancer.csv")
df['GENDER'] = df['GENDER'].replace({'M': 1, 'F': 0})
df['LUNG_CANCER'] = df['LUNG_CANCER'].replace({'YES': 1, 'NO': 0})
df.dropna(inplace=True)
X = df.drop(columns="LUNG_CANCER")
y = df["LUNG_CANCER"]

# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# Балансировка с помощью SMOTE
smote = SMOTE(sampling_strategy=0.7, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Масштабирование данных
scaler = StandardScaler()
X_resampled = scaler.fit_transform(X_resampled)
X_test_scaled = scaler.transform(X_test)

# Настройка и обучение RandomForest
param_grid = {'n_estimators': [100, 200, 300], 'max_depth': [5, 10, 20]}
rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='f1')
grid_search.fit(X_resampled, y_resampled)

# Оценка модели
best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test_scaled)
y_pred_proba = best_rf.predict_proba(X_test_scaled)[:, 1]

print("\nЛучшие параметры:", grid_search.best_params_)
print("\nПодробный отчет по метрикам:\n", classification_report(y_test, y_pred))
print(f"ROC-AUC на тестовом наборе: {roc_auc_score(y_test, y_pred_proba):.4f}")

# Кросс-валидация
cv_scores = cross_val_score(best_rf, X_resampled, y_resampled, cv=5, scoring='f1')
print(f"Средний F1-score на кросс-валидации: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Предсказание для новых данных
new_data = pd.DataFrame({
    'GENDER': [0, 1, 1],
    'AGE': [61, 53, 73],
    'SMOKING': [1, 2, 1],
    'YELLOW_FINGERS': [2, 2, 1],
    'ANXIETY': [2, 2, 1],
    'PEER_PRESSURE': [2, 2, 1],
    'CHRONIC DISEASE': [1, 2, 2],
    'FATIGUE ': [1, 1, 1],
    'ALLERGY ': [2, 2, 2],
    'WHEEZING': [2, 1, 1],
    'ALCOHOL CONSUMING': [1, 2, 2],
    'COUGHING': [2, 1, 2],
    'SHORTNESS OF BREATH': [1, 1, 2],
    'SWALLOWING DIFFICULTY': [2, 2, 2],
    'CHEST PAIN': [1, 2, 2]
})

new_data_scaled = scaler.transform(new_data)
predictions = best_rf.predict(new_data_scaled)
prediction_proba = best_rf.predict_proba(new_data_scaled)[:, 1]

for i, (pred, prob) in enumerate(zip(predictions, prediction_proba), 1):
    print(f"Пациент {i}: {'Да' if pred == 1 else 'Нет'}, вероятность: {prob:.2f}")
# Сохраняем модель
joblib.dump(best_rf, 'lung_cancer_model.pkl')

# Сохраняем скейлер
joblib.dump(scaler, 'scaler.pkl')
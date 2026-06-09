# =============================================================================
# Классификация типа вина по химическим характеристикам
# Датасет: wine_exam_example_dirty.csv
# =============================================================================

# --- Импорт всех необходимых библиотек ---

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, validation_curve, learning_curve
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import joblib

# --- Загрузка данных и первичный осмотр: размер, типы, пропуски, дубликаты ---

DATA_PATH = 'wine_exam_example_dirty.csv'

try:
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    df = pd.read_csv('/mnt/data/wine_exam_example_dirty.csv')

print('Размер датасета:', df.shape)
df.head()

# --- Вывод информации о типах данных и пропусках по каждому столбцу ---

print('Информация о датасете:')
df.info()

# --- Подсчёт пропущенных значений по каждому признаку ---

print('Количество пропусков по столбцам:')
df.isna().sum().sort_values(ascending=False)

# --- Удаление дубликатов ---

clean_df = df.copy()

print('Количество строк до удаления дубликатов:', clean_df.shape[0])
print('Количество полных дубликатов:', clean_df.duplicated().sum())

clean_df = clean_df.drop_duplicates()

print('Количество строк после удаления дубликатов:', clean_df.shape[0])

# --- Просмотр уникальных значений категориальных признаков ---

cat_cols_initial = clean_df.select_dtypes(include='object').columns

for col in cat_cols_initial:
    print(f'{col}:', clean_df[col].unique())

# --- Очистка грязных значений в признаке AlcoholLevel: оставляем только low/medium/high ---

if 'AlcoholLevel' in clean_df.columns:
    allowed_levels = ['low', 'medium', 'high']
    clean_df['AlcoholLevel'] = clean_df['AlcoholLevel'].where(
        clean_df['AlcoholLevel'].isin(allowed_levels),
        np.nan
    )

print('Уникальные значения AlcoholLevel после очистки:')
print(clean_df['AlcoholLevel'].unique() if 'AlcoholLevel' in clean_df.columns else 'Признак отсутствует')

# --- Удаление технического идентификатора SampleID и утечки целевой переменной TargetCodeLeak ---

drop_cols = []

for col in ['SampleID', 'TargetCodeLeak']:
    if col in clean_df.columns:
        drop_cols.append(col)

clean_df = clean_df.drop(columns=drop_cols)

print('Удалённые признаки:', drop_cols)
print('Размер после удаления лишних признаков:', clean_df.shape)

# --- Проверка распределения классов целевой переменной WineClass ---

print('Распределение классов:')
print(clean_df['WineClass'].value_counts())

# --- EDA: столбчатая диаграмма распределения объектов по классам ---

plt.figure(figsize=(6, 4))
clean_df['WineClass'].value_counts().sort_index().plot(kind='bar')
plt.title('Распределение объектов по классам WineClass')
plt.xlabel('Класс')
plt.ylabel('Количество объектов')
plt.grid(axis='y')
plt.show()

# --- EDA: описательная статистика числовых признаков ---

clean_df.describe().T

# --- EDA: boxplot числовых признаков для обнаружения выбросов ---

numeric_cols = clean_df.select_dtypes(include=np.number).columns.tolist()

plt.figure(figsize=(14, 6))
clean_df[numeric_cols].boxplot(rot=45)
plt.title('Boxplot числовых признаков')
plt.ylabel('Значения признаков')
plt.grid(True)
plt.show()

# --- EDA: boxplot признака alcohol в разбивке по классам ---

if 'alcohol' in clean_df.columns:
    plt.figure(figsize=(7, 5))
    clean_df.boxplot(column='alcohol', by='WineClass', grid=True)
    plt.title('Распределение alcohol по классам')
    plt.suptitle('')
    plt.xlabel('Класс вина')
    plt.ylabel('Alcohol')
    plt.show()

# --- EDA: корреляционная матрица числовых признаков ---

corr = clean_df[numeric_cols].corr()

plt.figure(figsize=(10, 8))
plt.imshow(corr, cmap='Reds')
plt.colorbar()
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.title('Корреляционная матрица числовых признаков')
plt.show()

# --- EDA: гистограммы распределений всех числовых признаков ---

clean_df[numeric_cols].hist(bins=20, figsize=(15, 15))
plt.suptitle('Распределения числовых признаков')
plt.tight_layout()
plt.show()

# --- Формирование X и y, кодирование меток классов, разбивка на train/valid ---

TARGET = 'WineClass'

X = clean_df.drop(columns=[TARGET])
y = clean_df[TARGET]

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print('Классы:', list(label_encoder.classes_))
print('Размер X:', X.shape)
print('Размер y:', y_encoded.shape)

# --- Стратифицированное разбиение на обучающую (75%) и валидационную (25%) выборки ---

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y_encoded,
    test_size=0.25,
    random_state=42,
    stratify=y_encoded
)

print('Размер обучающей выборки:', X_train.shape)
print('Размер валидационной выборки:', X_valid.shape)

# --- Определение числовых и категориальных признаков ---

numeric_features = X_train.select_dtypes(include=np.number).columns.tolist()
categorical_features = X_train.select_dtypes(include='object').columns.tolist()

print('Числовые признаки:', numeric_features)
print('Категориальные признаки:', categorical_features)

# --- Построение пайплайна предобработки: импутация + масштабирование/OHE ---

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# --- Базовая логистическая регрессия: обучение и оценка на валидации ---

log_reg_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LogisticRegression(max_iter=5000, random_state=42))
])

log_reg_model.fit(X_train, y_train)

y_pred_lr = log_reg_model.predict(X_valid)

print('Accuracy Logistic Regression:', accuracy_score(y_valid, y_pred_lr))
print(classification_report(y_valid, y_pred_lr, target_names=label_encoder.classes_))

# --- GridSearchCV для логистической регрессии: подбор C, penalty, solver ---

param_grid_lr = [
    {
        'model__C': [0.01, 0.1, 1, 10, 100],
        'model__penalty': ['l1', 'l2'],
        'model__solver': ['liblinear']
    },
    {
        'model__C': [0.01, 0.1, 1, 10, 100],
        'model__penalty': ['l2'],
        'model__solver': ['lbfgs']
    }
]

grid_lr = GridSearchCV(
    estimator=log_reg_model,
    param_grid=param_grid_lr,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_lr.fit(X_train, y_train)

print('Лучшие параметры Logistic Regression:')
print(grid_lr.best_params_)
print('Лучшая accuracy на кросс-валидации:', grid_lr.best_score_)

y_pred_lr_opt = grid_lr.predict(X_valid)

print('Accuracy оптимизированной Logistic Regression:', accuracy_score(y_valid, y_pred_lr_opt))
print(classification_report(y_valid, y_pred_lr_opt, target_names=label_encoder.classes_))

# --- Матрица ошибок для оптимизированной логистической регрессии ---

cm_lr = confusion_matrix(y_valid, y_pred_lr_opt)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm_lr,
    display_labels=label_encoder.classes_
)

disp.plot()
plt.title('Матрица ошибок: оптимизированная Logistic Regression')
plt.show()

# --- Кривая валидации логистической регрессии по параметру C ---

param_range_C = np.logspace(-3, 3, 7)

train_scores_lr, valid_scores_lr = validation_curve(
    estimator=Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', LogisticRegression(max_iter=5000, solver='liblinear', penalty='l2', random_state=42))
    ]),
    X=X_train,
    y=y_train,
    param_name='model__C',
    param_range=param_range_C,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

train_mean_lr = train_scores_lr.mean(axis=1)
valid_mean_lr = valid_scores_lr.mean(axis=1)

plt.figure(figsize=(8, 5))
plt.semilogx(param_range_C, train_mean_lr, marker='o', label='Train')
plt.semilogx(param_range_C, valid_mean_lr, marker='o', label='Validation')
plt.title('Кривая валидации Logistic Regression по параметру C')
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# --- Кривая обучения для оптимизированной логистической регрессии ---

train_sizes_lr, train_scores_lc_lr, valid_scores_lc_lr = learning_curve(
    estimator=grid_lr.best_estimator_,
    X=X_train,
    y=y_train,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    train_sizes=np.linspace(0.2, 1.0, 5)
)

plt.figure(figsize=(8, 5))
plt.plot(train_sizes_lr, train_scores_lc_lr.mean(axis=1), marker='o', label='Train')
plt.plot(train_sizes_lr, valid_scores_lc_lr.mean(axis=1), marker='o', label='Validation')
plt.title('Кривая обучения Logistic Regression')
plt.xlabel('Размер обучающей выборки')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# --- Базовая нейронная сеть MLPClassifier: обучение и оценка на валидации ---

mlp_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        alpha=0.001,
        learning_rate_init=0.001,
        max_iter=1000,
        early_stopping=True,
        random_state=42
    ))
])

mlp_model.fit(X_train, y_train)

y_pred_mlp = mlp_model.predict(X_valid)

print('Accuracy MLP:', accuracy_score(y_valid, y_pred_mlp))
print(classification_report(y_valid, y_pred_mlp, target_names=label_encoder.classes_))

# --- GridSearchCV для MLP: подбор архитектуры, активации, alpha, learning_rate ---

param_grid_mlp = {
    'model__hidden_layer_sizes': [(32,), (64,), (64, 32), (128, 64)],
    'model__activation': ['relu', 'tanh'],
    'model__alpha': [0.0001, 0.001, 0.01],
    'model__learning_rate_init': [0.001, 0.005]
}

grid_mlp = GridSearchCV(
    estimator=mlp_model,
    param_grid=param_grid_mlp,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_mlp.fit(X_train, y_train)

print('Лучшие параметры MLP:')
print(grid_mlp.best_params_)
print('Лучшая accuracy на кросс-валидации:', grid_mlp.best_score_)

y_pred_mlp_opt = grid_mlp.predict(X_valid)

print('Accuracy оптимизированной MLP:', accuracy_score(y_valid, y_pred_mlp_opt))
print(classification_report(y_valid, y_pred_mlp_opt, target_names=label_encoder.classes_))

# --- Матрица ошибок для оптимизированной нейронной сети ---

cm_mlp = confusion_matrix(y_valid, y_pred_mlp_opt)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm_mlp,
    display_labels=label_encoder.classes_
)

disp.plot()
plt.title('Матрица ошибок: оптимизированная MLP')
plt.show()

# --- Кривая валидации нейронной сети по параметру alpha (L2-регуляризация) ---

param_range_alpha = np.logspace(-5, 0, 6)

train_scores_mlp, valid_scores_mlp = validation_curve(
    estimator=Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation='relu',
            max_iter=1000,
            early_stopping=True,
            random_state=42
        ))
    ]),
    X=X_train,
    y=y_train,
    param_name='model__alpha',
    param_range=param_range_alpha,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

train_mean_mlp = train_scores_mlp.mean(axis=1)
valid_mean_mlp = valid_scores_mlp.mean(axis=1)

plt.figure(figsize=(8, 5))
plt.semilogx(param_range_alpha, train_mean_mlp, marker='o', label='Train')
plt.semilogx(param_range_alpha, valid_mean_mlp, marker='o', label='Validation')
plt.title('Кривая валидации MLP по параметру alpha')
plt.xlabel('alpha')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# --- Кривая обучения для оптимизированной нейронной сети ---

train_sizes_mlp, train_scores_lc_mlp, valid_scores_lc_mlp = learning_curve(
    estimator=grid_mlp.best_estimator_,
    X=X_train,
    y=y_train,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    train_sizes=np.linspace(0.2, 1.0, 5)
)

plt.figure(figsize=(8, 5))
plt.plot(train_sizes_mlp, train_scores_lc_mlp.mean(axis=1), marker='o', label='Train')
plt.plot(train_sizes_mlp, valid_scores_lc_mlp.mean(axis=1), marker='o', label='Validation')
plt.title('Кривая обучения MLP')
plt.xlabel('Размер обучающей выборки')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# --- График функции потерь (loss curve) лучшей нейронной сети ---

best_mlp = grid_mlp.best_estimator_.named_steps['model']

plt.figure(figsize=(8, 5))
plt.plot(best_mlp.loss_curve_)
plt.title('График функции потерь MLP')
plt.xlabel('Эпоха')
plt.ylabel('Loss')
plt.grid(True)
plt.show()

# --- Сравнительная таблица accuracy всех четырёх моделей ---

results = pd.DataFrame({
    'Модель': [
        'Logistic Regression',
        'Optimized Logistic Regression',
        'MLP',
        'Optimized MLP'
    ],
    'Accuracy': [
        accuracy_score(y_valid, y_pred_lr),
        accuracy_score(y_valid, y_pred_lr_opt),
        accuracy_score(y_valid, y_pred_mlp),
        accuracy_score(y_valid, y_pred_mlp_opt)
    ]
})

results

# --- Столбчатая диаграмма сравнения accuracy моделей ---

plt.figure(figsize=(8, 5))
plt.bar(results['Модель'], results['Accuracy'])
plt.title('Сравнение accuracy моделей')
plt.xlabel('Модель')
plt.ylabel('Accuracy')
plt.xticks(rotation=30, ha='right')
plt.ylim(0, 1)
plt.grid(axis='y')
plt.show()

# --- Выбор лучшей модели (MLP vs LogReg) и сохранение на диск через joblib ---

acc_lr_opt = accuracy_score(y_valid, y_pred_lr_opt)
acc_mlp_opt = accuracy_score(y_valid, y_pred_mlp_opt)

if acc_mlp_opt >= acc_lr_opt:
    best_model = grid_mlp.best_estimator_
    best_model_name = 'optimized_mlp'
    best_accuracy = acc_mlp_opt
else:
    best_model = grid_lr.best_estimator_
    best_model_name = 'optimized_logistic_regression'
    best_accuracy = acc_lr_opt

print('Лучшая модель:', best_model_name)
print('Accuracy лучшей модели:', best_accuracy)

joblib.dump(best_model, 'best_exam_model.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

print('Модель сохранена в файл best_exam_model.pkl')
print('Кодировщик целевой переменной сохранён в файл label_encoder.pkl')

# --- Проверка загрузки сохранённой модели и тестовый прогон предсказаний ---

loaded_model = joblib.load('best_exam_model.pkl')
loaded_encoder = joblib.load('label_encoder.pkl')

sample_prediction = loaded_model.predict(X_valid.head(5))
print('Предсказания для первых 5 объектов:', loaded_encoder.inverse_transform(sample_prediction))

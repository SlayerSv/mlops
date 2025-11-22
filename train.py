import pandas as pd
import json
import yaml
import re
import pymorphy3
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import pickle
import nltk
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
from nltk.corpus import stopwords

# Список русских стоп-слов
ru_stopwords = set(stopwords.words('russian'))

def remove_stopwords_aug(text):
    """Оставляет только значимые слова"""
    if not isinstance(text, str): return str(text)
    
    words = text.split()
    # Оставляем слова, которых НЕТ в списке стоп-слов
    filtered = [w for w in words if w.lower() not in ru_stopwords]
    
    # Если вдруг удалили вообще всё (бывает в коротких фразах), вернем оригинал
    if len(filtered) == 0:
        return text
        
    return ' '.join(filtered)
    
# --- НАСТРОЙКИ И ИНИЦИАЛИЗАЦИЯ ---
morph = pymorphy3.MorphAnalyzer()

def lemmatize_text(text):
    """Лемматизация: приведение к начальной форме"""
    # Оставляем только буквы
    words = re.findall(r'[а-яА-ЯёЁ]+', str(text))
    res = []
    for word in words:
        p = morph.parse(word)[0]
        res.append(p.normal_form)
    return ' '.join(res)

def simple_clean(text):
    """Простая очистка: только нижний регистр, без лемматизации"""
    # Оставляем буквы и цифры, переводим в нижний регистр
    # Это нужно, чтобы Tfidf не сходил с ума от знаков препинания
    words = re.findall(r'[а-яА-ЯёЁ]+', str(text))
    return ' '.join(words).lower()

# --- ОСНОВНОЙ ПАЙПЛАЙН ---

# 1. Загружаем параметры
params = yaml.safe_load(open("params.yaml"))["train"]
model_name = params["model"]
use_augmentation = params.get("augment", False)
use_lemmatization = params.get("lemmatize", False)

# 2. Загрузка данных
df = pd.read_csv('anecdotes.csv')

# 3. Предобработка текста (Ветвление логики)
if use_lemmatization:
    print("Лемматизация ВКЛЮЧЕНА")
    df['text_processed'] = df['text'].apply(lemmatize_text)
else:
    print("Лемматизация ВЫКЛЮЧЕНА (только нижний регистр)")
    df['text_processed'] = df['text'].apply(simple_clean)

# 4. Кодирование категорий
print("Кодирование категорий...")
le = LabelEncoder()
df['category_encoded'] = le.fit_transform(df['category'])

X = df['text_processed']
y = df['category_encoded']

# 5. РАЗДЕЛЕНИЕ ДАННЫХ
# Делим ДО аугментации, чтобы избежать утечки!
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. АУГМЕНТАЦИЯ (Только на Train)
if use_augmentation:
    print("Аугментация ВКЛЮЧЕНА (Random Deletion)")
    print(f"Размер Train до: {len(X_train)}")
    
    # Аугментируем
    X_train_aug = X_train.apply(lambda x: remove_stopwords_aug(x))
    y_train_aug = y_train.copy()
    
    # Объединяем
    X_train = pd.concat([X_train, X_train_aug])
    y_train = pd.concat([y_train, y_train_aug])
    
    print(f"Размер Train после: {len(X_train)}")
else:
    print("Аугментация ВЫКЛЮЧЕНА")

# 7. Векторизация
vectorizer = TfidfVectorizer(max_features=params["max_features"], ngram_range=(1, 1))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 8. Обучение
print(f"Обучаем модель: {model_name}")
if model_name == 'xgb':
    model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', n_estimators=params["n_estimators"])
elif model_name == 'rf':
    model = RandomForestClassifier(n_estimators=params["n_estimators"])
elif model_name == 'nb':
    model = MultinomialNB()

model.fit(X_train_vec, y_train)

# 9. Оценка
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print(f"Точность: {accuracy}")
metrics = {'accuracy': accuracy}
with open("results.json", "w") as f:
    json.dump(metrics, f)

pickle.dump(model, open("model.pkl", "wb"))
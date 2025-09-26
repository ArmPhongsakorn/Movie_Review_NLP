import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

# คอมพิวเตอร์ไม่เข้าใจคำศัพท์ แต่เข้าใจตัวเลข เราจึงต้องแปลงข้อความที่ทำความสะอาดแล้วให้เป็นเวกเตอร์ตัวเลข เทคนิคที่ง่ายที่สุดคือ TF-IDF Vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

import pickle
import os

# ดาวน์โหลด Stop Words และ Lemmatizer ที่จำเป็น (ทำแค่ครั้งแรก)
# nltk.download('stopwords')
# nltk.download('wordnet')

# data set from kaggle https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews/data
# กำหนด Path เต็มของไฟล์ โดยเพิ่มชื่อไฟล์เข้าไปในส่วนท้าย
file_path = r'C:\Users\Asus\.cache\kagglehub\datasets\lakshmi25npathi\imdb-dataset-of-50k-movie-reviews\versions\1\IMDB Dataset.csv'

# โหลดข้อมูลจากไฟล์ Excel
df = pd.read_csv(file_path)

# ลองดูข้อมูล 5 แถวแรกเพื่อตรวจสอบ
print(df.head())

# ดาวน์โหลดชุดข้อมูล NLTK ที่จำเป็น (รันแค่ครั้งแรก)
# nltk.download('stopwords')
# nltk.download('wordnet')
    
# ฟังก์ชันสำหรับทำความสะอาดข้อความ
def preprocess_text(text):
    # เพิ่มโค้ดสำหรับลบ HTML tags
    text = re.sub(r'<.*?>', '', text)
    # ลบตัวอักษรพิเศษและตัวเลข
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # แปลงเป็นตัวพิมพ์เล็กทั้งหมด
    text = text.lower()
    # ตัดคำเป็น Token
    tokens = text.split()
    # ลบ Stop Words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # ทำให้คำกลับไปสู่รูปพื้นฐาน (Lemmatization)
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# ใช้ฟังก์ชันกับข้อมูล
df['cleaned_review'] = df['review'].apply(preprocess_text)
print(df)

# สร้าง TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer()

# แปลงข้อความให้เป็นเวกเตอร์
X = tfidf_vectorizer.fit_transform(df['cleaned_review'])

# กำหนดตัวแปรเป้าหมาย (label)
y = df['sentiment']

# ใช้โมเดล Naive Bayes ซึ่งเป็นโมเดลที่นิยมและทำงานได้ดีกับงานจำแนกข้อความ
# แบ่งข้อมูลเป็นชุดฝึก (Training Set) และชุดทดสอบ (Testing Set)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# สร้างและฝึกโมเดล
model = MultinomialNB()
model.fit(X_train, y_train)

# ทำนายผลบนชุดข้อมูลทดสอบ
y_pred = model.predict(X_test)

# ประเมินผลการทำงานของโมเดล
print(classification_report(y_test, y_pred))

# กำหนด Path ของโฟลเดอร์ models
# MODEL_DIR = 'models'

# # ตรวจสอบว่าโฟลเดอร์ models มีอยู่หรือไม่ ถ้าไม่มีให้สร้างขึ้น
# if not os.path.exists(MODEL_DIR):
#     os.makedirs(MODEL_DIR)
#     print(f"สร้างโฟลเดอร์: {MODEL_DIR} เรียบร้อยแล้ว")

# # กำหนดชื่อไฟล์โดยระบุโฟลเดอร์นำหน้า
# MODEL_FILENAME = os.path.join(MODEL_DIR, 'naive_bayes_model.pkl')
# VECTORIZER_FILENAME = os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl')

# # บันทึกโมเดล (Naive Bayes)
# with open(MODEL_FILENAME, 'wb') as file:
#     pickle.dump(model, file)
# print(f"บันทึกโมเดลเป็น: {MODEL_FILENAME}")

# # บันทึก Vectorizer (TF-IDF)
# with open(VECTORIZER_FILENAME, 'wb') as file:
#     pickle.dump(tfidf_vectorizer, file)
# print(f"บันทึก Vectorizer เป็น: {VECTORIZER_FILENAME}")

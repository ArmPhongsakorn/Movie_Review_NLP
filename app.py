from flask import Flask, render_template, request, redirect, url_for
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import pandas as pd # ใช้เฉพาะในตัวอย่างนี้เพื่อจำลองฐานข้อมูล

# ตั้งค่าโมเดลและไฟล์
MODEL_PATH = 'models/naive_bayes_model.pkl'
VECTORIZER_PATH = 'models/tfidf_vectorizer.pkl'

# โหลดโมเดลและ Vectorizer ทันทีที่เซิร์ฟเวอร์เริ่มต้น
try:
    with open(MODEL_PATH, 'rb') as f:
        loaded_model = pickle.load(f)
    with open(VECTORIZER_PATH, 'rb') as f:
        loaded_vectorizer = pickle.load(f)
    # ดาวน์โหลด NLTK resource ถ้ายังไม่มี
    try:
        _ = stopwords.words('english')
        _ = WordNetLemmatizer()
    except:
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
    
except FileNotFoundError:
    print("!!! ERROR: ไม่พบไฟล์โมเดลหรือ Vectorizer โปรดตรวจสอบโฟลเดอร์ models !!!")
    exit()


app = Flask(__name__)

# --- จำลองฐานข้อมูลสำหรับเก็บข้อมูลรีวิว ---
# ใช้ Dictionary เพื่อเก็บข้อมูลรีวิวของแต่ละหนัง (ในเว็บจริงต้องใช้ Database)
movie_data = {
    'm1': {'title': 'The Code Whisperer', 'reviews': []},
    'm2': {'title': 'Data Deluge', 'reviews': []},
    'm3': {'title': 'The Vector Journey', 'reviews': []},
    'm4': {'title': 'NLP Dreams', 'reviews': []},
    'm5': {'title': 'The Risen of Computer Science', 'reviews': []},
}

# ฟังก์ชัน Preprocessing (ต้องเหมือนกับที่ใช้ในการฝึกโมเดล)
def preprocess_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# ฟังก์ชันหลักสำหรับการทำนายและคำนวณคะแนน
def calculate_score(movie_id):
    reviews = movie_data[movie_id]['reviews']
    if not reviews:
        return 0  # ไม่มีรีวิว

    positive_count = 0
    
    for review in reviews:
        # 1. Preprocess ข้อความ
        cleaned_text = preprocess_text(review)
        # 2. Vectorize ข้อความ (ใช้ Vectorizer ที่โหลดมา)
        vector = loaded_vectorizer.transform([cleaned_text])
        # 3. ทำนายผล (0: Negative, 1: Positive)
        prediction = loaded_model.predict(vector)[0]
        
        # Naive Bayes Model มักจะคืนค่าเป็น 'positive' หรือ 'negative' โดยตรง
        if prediction == 'positive':
            positive_count += 1
        # *Note: ถ้าโมเดลของคุณถูก Label Encode ให้เป็น 0/1 ต้องปรับเงื่อนไขตรงนี้*

    # คำนวณคะแนนเป็น %
    score = (positive_count / len(reviews)) * 100
    return round(score, 1)

# --- หน้าเว็บไซต์หลัก (แสดง Card หนัง) ---
@app.route('/')
def index():
    # สร้างรายการหนังพร้อมคะแนนปัจจุบัน
    movies_with_scores = []
    for mid, data in movie_data.items():
        score = calculate_score(mid)
        movies_with_scores.append({
            'id': mid,
            'title': data['title'],
            'score': score
        })
        
    return render_template('index.html', movies=movies_with_scores)

# --- หน้าสำหรับเขียนรีวิว ---
@app.route('/movie/<movie_id>', methods=['GET', 'POST'])
def movie_page(movie_id):
    if movie_id not in movie_data:
        return "Movie Not Found", 404
    
    movie_title = movie_data[movie_id]['title']
    current_score = calculate_score(movie_id)
    reviews_list = movie_data[movie_id]['reviews']
    
    if request.method == 'POST':
        # รับข้อมูลรีวิวใหม่
        new_review = request.form.get('review_text')
        if new_review:
            # เพิ่มรีวิวใหม่เข้าไปในฐานข้อมูลจำลอง
            movie_data[movie_id]['reviews'].append(new_review)
            # Redirect กลับไปหน้าเดิมเพื่อแสดงผลลัพธ์ที่อัปเดต
            return redirect(url_for('movie_page', movie_id=movie_id))
    
    return render_template('movie.html', 
                           movie_id=movie_id,
                           title=movie_title, 
                           score=current_score,
                           reviews=reviews_list)

if __name__ == '__main__':
    # รันเว็บเซิร์ฟเวอร์ Flask
    app.run(debug=True)
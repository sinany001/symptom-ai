
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr, Field
import joblib
import psycopg2
from psycopg2.extras import RealDictCursor
from passlib.context import CryptContext
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired

from symptom_extractor import extract_symptoms
from risk_assessment import assess_risk
from ai_agent import extract_symptom_information


# ========================================
# FASTAPI
# ========================================

app = FastAPI(
    title="SymptomAI API",
    description="AI-based preliminary health assessment system",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Swagger Authorize butonu için
security = HTTPBearer()


# ========================================
# MODEL VE VECTORIZER
# ========================================

model = joblib.load("models/symptom_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")


# ========================================
# PASSWORD VE TOKEN AYARLARI
# ========================================

pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto"
)

# Geliştirme ortamı için geçici anahtar.
# Gerçek dağıtım öncesinde ortam değişkeninden alınmalı.
SECRET_KEY = "CHANGE_THIS_TO_A_LONG_RANDOM_SECRET_BEFORE_DEPLOYMENT"
serializer = URLSafeTimedSerializer(SECRET_KEY)


# ========================================
# DATABASE
# ========================================

def get_connection():
    return psycopg2.connect(
        dbname="symptom_ai",
        user="sinanyilmaz"
    )


# ========================================
# REQUEST MODELS
# ========================================

class RegisterRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    email: EmailStr
    password: str = Field(min_length=8, max_length=72)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=72)


class SymptomRequest(BaseModel):
    symptoms: str = Field(min_length=3, max_length=5000)


# ========================================
# AUTHENTICATION HELPER
# ========================================

def get_current_user(token: str):
    try:
        payload = serializer.loads(
            token,
            max_age=60 * 60 * 24
        )
        user_id = payload.get("user_id")

        if not user_id:
            raise HTTPException(
                status_code=401,
                detail="Invalid token."
            )

    except (BadSignature, SignatureExpired):
        raise HTTPException(
            status_code=401,
            detail="Session expired or invalid. Please log in again."
        )

    conn = get_connection()

    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                """
                SELECT id, name, email
                FROM users
                WHERE id = %s
                """,
                (user_id,)
            )
            user = cursor.fetchone()

        if not user:
            raise HTTPException(
                status_code=401,
                detail="User not found."
            )

        return dict(user)

    finally:
        conn.close()


# ========================================
# HOME
# ========================================

@app.get("/")
def home():
    return {
        "message": "SymptomAI API is running!"
    }


# ========================================
# REGISTER
# ========================================

@app.post("/register")
def register(request: RegisterRequest):
    password_hash = pwd_context.hash(request.password)
    conn = get_connection()

    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO users (name, email, password_hash)
                VALUES (%s, %s, %s)
                RETURNING id, name, email
                """,
                (
                    request.name.strip(),
                    request.email.lower(),
                    password_hash
                )
            )

            user = cursor.fetchone()

        conn.commit()

        return {
            "message": "User registered successfully",
            "user": {
                "id": user[0],
                "name": user[1],
                "email": user[2]
            }
        }

    except psycopg2.errors.UniqueViolation:
        conn.rollback()
        raise HTTPException(
            status_code=409,
            detail="This email is already registered."
        )

    finally:
        conn.close()


# ========================================
# LOGIN
# ========================================

@app.post("/login")
def login(request: LoginRequest):
    conn = get_connection()

    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                """
                SELECT id, name, email, password_hash
                FROM users
                WHERE email = %s
                """,
                (request.email.lower(),)
            )
            user = cursor.fetchone()

        if not user or not pwd_context.verify(
            request.password,
            user["password_hash"]
        ):
            raise HTTPException(
                status_code=401,
                detail="Incorrect email or password."
            )

        token = serializer.dumps({
            "user_id": user["id"]
        })

        return {
            "message": "Login successful",
            "access_token": token,
            "token_type": "bearer",
            "user": {
                "id": user["id"],
                "name": user["name"],
                "email": user["email"]
            }
        }

    finally:
        conn.close()


# ========================================
# CURRENT USER
# ========================================

@app.get("/me")
def me(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    return get_current_user(credentials.credentials)


# ========================================
# PREDICTION
# ========================================

@app.post("/predict")
def predict(
    request: SymptomRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    # Kullanıcının kimliğini doğrula
    user = get_current_user(credentials.credentials)

    user_input = request.symptoms.strip()

    # 1. Semptomları tespit et
    detected_symptoms = extract_symptoms(user_input)

    # 2. Metni TF-IDF ile dönüştür
    symptom_tfidf = vectorizer.transform([user_input])

    # 3. SVM modeliyle sınıflandır
    prediction = model.predict(symptom_tfidf)[0]

    # 4. Prototip risk değerlendirmesi
    risk_result = assess_risk(detected_symptoms)

    # 5. Analiz sonucunu PostgreSQL'e kaydet
    conn = get_connection()

    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO analyses (
                    user_id,
                    symptoms,
                    classification,
                    risk_level,
                    risk_message
                )
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    user["id"],
                    user_input,
                    prediction,
                    risk_result["level"],
                    risk_result["message"]
                )
            )

            analysis_id = cursor.fetchone()[0]

        conn.commit()

    finally:
        conn.close()

    # 6. Sonucu döndür
    return {
        "analysis_id": analysis_id,
        "input": user_input,
        "detected_symptoms": detected_symptoms,
        "classification": prediction,
        "risk_level": risk_result["level"],
        "risk_message": risk_result["message"],
        "disclaimer": (
            "This is a dataset-based preliminary classification, "
            "not a medical diagnosis."
        )
    }


# ========================================
# USER ANALYSIS HISTORY
# ========================================

@app.get("/analyses")
def analysis_history(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    user = get_current_user(credentials.credentials)
    conn = get_connection()

    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                """
                SELECT
                    id,
                    symptoms,
                    classification,
                    risk_level,
                    risk_message,
                    created_at
                FROM analyses
                WHERE user_id = %s
                ORDER BY id DESC
                """,
                (user["id"],)
            )

            rows = cursor.fetchall()

        return {
            "analyses": [
                dict(row) for row in rows
            ]
        }

    finally:
        conn.close()

# ========================================
# AI AGENT - SYMPTOM EXTRACTION
# ========================================

@app.post("/agent/extract")
def agent_extract(
    request: SymptomRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    # Kullanıcının oturumunu doğrula
    get_current_user(credentials.credentials)

    try:
        result = extract_symptom_information(request.symptoms)
        return result.model_dump()

    except ValueError as error:
        raise HTTPException(
            status_code=422,
            detail=str(error),
        )

    except Exception:
        raise HTTPException(
            status_code=503,
            detail="AI Agent is temporarily unavailable. Please try again later.",
        )


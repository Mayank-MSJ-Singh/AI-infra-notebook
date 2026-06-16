from fastapi import FastAPI
import httpx
import redis
import psycopg2
import os

app = FastAPI()

# Configuration from environment
MODEL_URL = os.getenv("MODEL_SERVER_URL")
redis_client = redis.from_url(os.getenv("REDIS_URL"))

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
async def predict(data: dict):
    # 1. Check Redis cache

    # Create cache key from input
    cache_key = f"predict:{data['text']}"

    # Check Redis
    result = redis_client.get(cache_key)

    # 2. If not cached, call model server

    if not result:
        async with httpx.AsyncClient() as client:
            response = await client.post(MODEL_URL, json=data)
            result = response.json()  # model returns JSON with "prediction"
        
        redis_client.set(cache_key, result["prediction"])


    # 3. Cache result
    
    # 4. Log to PostgreSQL
    conn = psycopg2.connect(
        os.getenv("DATABASE_URL")
    )

    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO predictions (text, prediction)
        VALUES (%s, %s)
        """,
        (
            data["text"],
            result["prediction"]
        )
    )       

    cursor.close()
    conn.close()

    # 5. Return prediction
    return {"prediction":result["prediction"]}
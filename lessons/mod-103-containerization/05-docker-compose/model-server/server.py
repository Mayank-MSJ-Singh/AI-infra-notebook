from fastapi import FastAPI

app = FastAPI()

# TODO: Load model
# model = torch.load("/models/model.pth")

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
def predict(data: dict):
    return {"prediction":"hello"}
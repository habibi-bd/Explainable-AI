from fastapi import FastAPI, Request
import httpx

app = FastAPI()

# URL of your local LLM server
LOCAL_LLM_URL = "http://localhost:1234/v1/chat/completions"

@app.post("/generate")
async def generate(request: Request):
    # Receive JSON body from client
    input_data = await request.json()

    # Prepare payload for local LLM (you can customize here)
    payload = {
        "model": "google/gemma-3-4b",
        "messages": input_data.get("messages", [])
    }

    # Forward the request to your local LLM server
    async with httpx.AsyncClient() as client:
        response = await client.post(LOCAL_LLM_URL, json=payload)
        llm_response = response.json()

    # Optionally modify or log the response here

    return llm_response

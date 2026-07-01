import json
import os
from contextlib import asynccontextmanager
from typing import List, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google import genai
from google.genai import types

# In-memory storage for the parsed catalog data
CATALOG_DATA = []

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI.
    Loads the product catalog from shl_product_catalogue.json on startup.
    """
    global CATALOG_DATA
    try:
        with open("shl_product_catalogue.json", "r", encoding="utf-8") as f:
            raw_catalog = json.load(f, strict=False)
            for item in raw_catalog:
                keys = item.get("keys", [])
                test_type = None
                
                # Map keys to the required test_type
                if "Knowledge & Skills" in keys:
                    test_type = "K"
                elif "Personality & Behavior" in keys or "Competencies" in keys:
                    test_type = "P"
                
                # Only include valid items with a determined test_type
                if test_type and "name" in item and "link" in item:
                    CATALOG_DATA.append({
                        "name": item["name"],
                        "url": item["link"],
                        "test_type": test_type
                    })
        print(f"Successfully loaded {len(CATALOG_DATA)} items from the catalog.")
    except Exception as e:
        print(f"Error loading catalog: {e}")
    
    yield
    # Cleanup resources if necessary during shutdown

app = FastAPI(lifespan=lifespan)

# --- Pydantic Models for Request and Response ---

class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]

class Recommendation(BaseModel):
    name: str
    url: str
    test_type: Literal["K", "P"]

class ChatResponse(BaseModel):
    reply: str
    recommendations: List[Recommendation]
    end_of_conversation: bool

# --- API Endpoints ---

@app.get("/health")
async def health_check():
    """Health check endpoint to verify the service is running."""
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Stateless chat endpoint. Uses the Gemini SDK to process conversation history
    and return structured JSON matching ChatResponse.
    """
    # Ensure API key is set
    if not os.environ.get("GEMINI_API_KEY"):
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY environment variable is missing.")

    try:
        client = genai.Client()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize Gemini client: {e}")

    # Build the system instruction, embedding the available catalog
    catalog_json = json.dumps(CATALOG_DATA)
    system_instruction = (
        "You are an SHL conversational assessment recommender assistant. "
        "Your goal is to converse with the user, understand their assessment needs, "
        "and recommend the most appropriate assessments from the provided catalog. "
        "If you have gathered enough information and made your recommendations, set end_of_conversation to true. "
        "Always respond in JSON format matching the requested schema. "
        "Here is the catalog of available assessments:\n"
        f"{catalog_json}"
    )

    # Convert FastAPI incoming messages to Gemini Content types
    contents = []
    for msg in request.messages:
        # Map "assistant" role from our API to "model" for Gemini
        role = "user" if msg.role == "user" else "model"
        contents.append(types.Content(role=role, parts=[types.Part.from_text(text=msg.content)]))

    # Configure the Gemini API call to enforce JSON output using the Pydantic model
    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        response_mime_type="application/json",
        response_schema=ChatResponse,
        temperature=0.2, # Lower temperature for reliable schema adherence
    )

    try:
        # Call Gemini (using gemini-2.5-flash)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            config=config,
        )
        
        # The SDK returns a JSON string conforming to ChatResponse
        response_data = json.loads(response.text)
        return response_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating AI response: {e}")

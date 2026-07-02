import os
import json
import logging
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("capcore")

# Load environment variables
load_dotenv()

app = FastAPI(title="CapCore Backend", description="AI-powered stateless caption generator")

# Allow CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Gemini Client
# Client will automatically retrieve GEMINI_API_KEY from environment variables
try:
    client = genai.Client()
except Exception as e:
    logger.error(f"Failed to initialize Gemini Client: {e}")
    client = None

# System instruction as strictly required by the prompt, expanded for Gen-Z tags and medium captions
SYSTEM_INSTRUCTION = (
    "You are the AI engine for 'CapCore'. Transform the user's input into three distinct social media captions and a list of social tags. "
    "Prioritize linguistic precision and natural phrasing. NEVER use generic AI clichés. "
    "Output ONLY a valid JSON object matching this schema: "
    "{'literary': 'eloquent and deep', 'medium': 'one or two sentences, well-balanced', 'micro': '1-4 words max', "
    "'tags': '5-8 modern aesthetic Gen-Z tags and hashtags for Instagram, Snapchat, X, etc.'}."
)

@app.get("/", response_class=HTMLResponse)
async def read_index():
    """Serves the frontend application."""
    index_path = os.path.join(os.path.dirname(__file__), "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="index.html not found")
    with open(index_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@app.post("/generate-captions")
async def generate_captions(
    text: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None)
):
    """
    Accepts text and/or an image, calls Gemini API, and returns caption options.
    Strictly stateless: files processed in-memory and closed/flushed immediately.
    """
    # 1. Validation
    if not client:
        raise HTTPException(
            status_code=500,
            detail="Gemini Client is not initialized. Please configure GEMINI_API_KEY in the environment."
        )

    # Validate that we received at least one input
    has_text = text is not None and len(text.strip()) > 0
    has_image = image is not None and image.filename != ""

    if not has_text and not has_image:
        raise HTTPException(
            status_code=400,
            detail="Provide either a text prompt or an image to generate captions."
        )

    contents = []
    
    try:
        # 2. In-Memory Image Processing
        if has_image:
            logger.info(f"Processing uploaded image in-memory: {image.filename} ({image.content_type})")
            image_bytes = await image.read()
            
            if len(image_bytes) == 0:
                raise HTTPException(
                    status_code=400,
                    detail="The uploaded image is empty."
                )

            # Append the binary data directly as an inline Part using the SDK
            contents.append(
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type=image.content_type or "image/jpeg"
                )
            )

        # 3. Add text prompt if provided
        if has_text:
            contents.append(text)
        else:
            # If only image is provided, add a default prompt to instruct the model to describe/capture it
            contents.append("Generate captions for this image.")

        # 4. Generate Content via Gemini API
        logger.info("Sending request to Gemini API (gemini-2.5-flash)")
        
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                response_mime_type="application/json",
            )
        )
        
        # 5. Parse JSON response
        response_text = response.text
        if not response_text:
            raise HTTPException(
                status_code=502,
                detail="Empty response received from the Gemini model."
            )
            
        logger.info("Successfully received response from Gemini API")
        
        try:
            parsed_json = json.loads(response_text)
            return JSONResponse(content=parsed_json)
        except json.JSONDecodeError as jde:
            logger.error(f"Failed to parse model response as JSON: {response_text}. Error: {jde}")
            raise HTTPException(
                status_code=502,
                detail="Model response was not valid JSON. Please try again."
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during caption generation: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred during caption generation: {str(e)}"
        )
        
    finally:
        # 6. Strict memory cleanup: close upload file and drop reference
        if image:
            logger.info(f"Flushing and closing image memory stream for {image.filename}")
            await image.close()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import groq


# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with specific domains for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Welcome to the Article AI Chat API"}


# Remove proxy environment variables
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)

# Initialize the Groq client
api_key = os.getenv("GROQ_API_KEY")

try:
    # Pass only required arguments to avoid the 'proxies' error
    client = groq.Client(api_key=api_key)
except TypeError as e:
    print(f"Error initializing Groq Client: {e}")
    client = None  # Safeguard against uninitialized client
except Exception as e:
    print(f"Unexpected error initializing Groq Client: {e}")
    client = None


# Helper function to fetch content from a news article link
def fetch_news_article_content(url: str) -> str:
    try:
        print(f"Fetching article content from URL: {url}")
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
        }
        response = requests.get(url, headers=headers)
        print(f"Response status code: {response.status_code}")
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "html.parser")
            article_text = " ".join(p.get_text() for p in soup.find_all("p"))
            print(
                f"Extracted article content: {article_text[:100]}..."
            )  # Log the first 100 characters
            return article_text
        else:
            print("Failed to fetch article content.")
            raise HTTPException(
                status_code=400, detail="Failed to retrieve content from the article."
            )
    except Exception as e:
        print(f"Error fetching article: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching article: {str(e)}")


# Models
class AIRequest(BaseModel):
    article_link: str
    message: str


# Routes
@app.post("/api/extract-article")
async def extract_article_content(request: AIRequest):
    """Extract content from an article link."""
    article_link = request.article_link
    if not article_link:
        raise HTTPException(status_code=400, detail="Article link is required.")

    # Fetch the article content
    article_content = fetch_news_article_content(article_link)
    if not article_content:
        raise HTTPException(
            status_code=400, detail="Failed to retrieve content from the article."
        )

    return {
        "article_content": article_content[:1000]
    }  # Return the first 1000 characters as a preview


@app.post("/api/interact")
async def interact_with_article(request: AIRequest):
    try:
        print(f"Received API request: {request}")
        if client is None:
            raise HTTPException(
                status_code=500, detail="Groq Client is not initialized."
            )

        article_link = request.article_link
        user_message = request.message

        # Validate input
        if not article_link or not user_message:
            print("Validation failed: Missing article link or message.")
            raise HTTPException(
                status_code=400, detail="Both article link and message are required."
            )

        # Fetch the article content
        article_content = fetch_news_article_content(article_link)
        if not article_content:
            print("Failed to retrieve article content.")
            raise HTTPException(
                status_code=400, detail="Failed to retrieve content from the article."
            )

        # Generate the prompt
        prompt = f"Based on the following article content:\n\n{article_content[:1500]}...\n\nAnswer this question: {user_message}"
        print(
            f"Generated prompt: {prompt[:100]}..."
        )  # Log the first 100 characters of the prompt

        # Query the Groq Llama model
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="llama3-70b-8192",  # Replace with the appropriate model
        )
        response = chat_completion.choices[0].message.content
        print(f"Groq response: {response[:100]}...")  # Log the first 100 characters
        return {"reply": response}

    except Exception as e:
        print(f"Error in /api/interact: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error querying AI: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

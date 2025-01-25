from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from pymongo import MongoClient
from dotenv import load_dotenv
import groq
from bson import ObjectId
import requests
from bs4 import BeautifulSoup

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

# MongoDB connection
MONGO_URI = "mongodb+srv://nishantbaruah3:Iwp93v4ZKUNXUPdG@cluster0.afoul.mongodb.net/"
client = MongoClient(MONGO_URI)
db = client["test"]  # Database name
articles_collection = db["articles"]  # Collection name

@app.get("/")
async def root():
    return {"message": "Welcome to the Article AI Chat API"}

# Initialize the Groq client
api_key = os.getenv("GROQ_API_KEY")

try:
    groq_client = groq.Client(api_key=api_key)
except Exception as e:
    print(f"Error initializing Groq Client: {e}")
    groq_client = None

# Models
class AIRequest(BaseModel):
    article_content: str
    message: str

class ArticleLinkRequest(BaseModel):
    url: str
    message: str

# Models
class EnhanceRequest(BaseModel):
    content: str


@app.post("/api/articles/enhance")
async def enhance_article_content(request: EnhanceRequest):
    """
    Route to enhance the article content using AI.
    """
    try:
        if groq_client is None:
            raise HTTPException(
                status_code=500, detail="Groq Client is not initialized."
            )

        content = request.content

        if not content.strip():
            raise HTTPException(status_code=400, detail="Content cannot be empty.")

        # AI prompt to enhance content
        prompt = f"Rewrite the following text to be more professional, engaging, and polished:\n\n{content[:3000]}"

        response = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-70b-8192",  # Replace with the appropriate model
        )
        enhanced_content = response.choices[0].message.content.strip()

        return {"enhanced_content": enhanced_content}

    except Exception as e:
        print(f"Error enhancing content: {e}")
        raise HTTPException(status_code=500, detail=f"Error enhancing content: {str(e)}")


def fetch_article_content_from_url(url: str) -> str:
    """
    Fetches the content of an article from the given URL using Beautiful Soup.
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")

        # Extract the main content of the article (common tags for articles)
        paragraphs = soup.find_all("p")
        content = " ".join([p.get_text(strip=True) for p in paragraphs])

        if not content.strip():
            raise ValueError("Unable to extract content from the provided URL.")

        return content.strip()

    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Error fetching article content: {str(e)}"
        )


@app.post("/api/interact")
async def interact_with_article(request: AIRequest):
    try:
        if groq_client is None:
            raise HTTPException(
                status_code=500, detail="Groq Client is not initialized."
            )

        article_content = request.article_content
        user_message = request.message

        if not article_content or not user_message:
            raise HTTPException(
                status_code=400, detail="Both article content and message are required."
            )

        prompt = f"Based on the following article content:\n\n{article_content[:1500]}...\n\nAnswer this question professionally with a structured format: {user_message}"

        chat_completion = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-70b-8192",  # Replace with the appropriate model
        )
        response = chat_completion.choices[0].message.content

        formatted_response = f"### Response:\n\n{response.strip()}"
        return {"reply": formatted_response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying AI: {str(e)}")


@app.post("/api/interact_from_url")
async def interact_with_article_from_url(request: ArticleLinkRequest):
    """
    Allows users to interact with an article by providing its URL.
    """
    try:
        if groq_client is None:
            raise HTTPException(
                status_code=500, detail="Groq Client is not initialized."
            )

        # Fetch article content from URL
        article_content = fetch_article_content_from_url(request.url)
        user_message = request.message

        if not user_message:
            raise HTTPException(
                status_code=400, detail="A message is required to ask a question."
            )

        prompt = f"Based on the following article content:\n\n{article_content[:1500]}...\n\nAnswer this question professionally with a structured format: {user_message}"

        chat_completion = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-70b-8192",  # Replace with the appropriate model
        )
        response = chat_completion.choices[0].message.content

        formatted_response = f"### Response:\n\n{response.strip()}"
        return {"reply": formatted_response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying AI: {str(e)}")


@app.get("/api/articles/classify_one/{article_id}")
async def classify_one_article(article_id: str):
    try:
        if groq_client is None:
            raise HTTPException(
                status_code=500, detail="Groq Client is not initialized."
            )

        # Fetch the article by ID
        article = articles_collection.find_one({"_id": ObjectId(article_id)})
        if not article:
            raise HTTPException(status_code=404, detail="Article not found.")

        content = article.get("content", "")
        if not content:
            category = "Uncategorized"
        else:
            # Groq classification prompt
            prompt = f"Classify the following article into a category such as Sports, Technology, Food, etc., based on its content:\n\n{content[:1000]}\n\nCategory:"
            response = groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama3-70b-8192",
            )
            category = response.choices[0].message.content.strip()

        # Return classification for the article
        return {
            "id": str(article["_id"]),
            "title": article.get("title", "Untitled"),
            "category": category,
        }

    except Exception as e:
        print(f"Error classifying article {article_id}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error processing article {article_id}."
        )


@app.get("/api/articles/summarize/{article_id}")
async def summarize_article(article_id: str):
    """
    Route to summarize an article's content in 50 to 100 words.
    """
    try:
        if groq_client is None:
            raise HTTPException(
                status_code=500, detail="Groq Client is not initialized."
            )

        # Fetch the article by ID
        article = articles_collection.find_one({"_id": ObjectId(article_id)})
        if not article:
            raise HTTPException(status_code=404, detail="Article not found.")

        content = article.get("content", "")
        if not content:
            summary = "No content available to summarize."
        else:
            # Groq summary prompt
            prompt = f"Summarize the following article content in 50 to 100 words:\n\n{content[:3000]}"
            response = groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama3-70b-8192",
            )
            summary = response.choices[0].message.content.strip()

        # Return the summary for the article
        return {
            "id": str(article["_id"]),
            "title": article.get("title", "Untitled"),
            "summary": summary,
        }

    except Exception as e:
        print(f"Error summarizing article {article_id}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error processing article {article_id}."
        )


@app.get("/api/articles")
async def get_all_articles():
    """
    Route to fetch all articles with their IDs and titles.
    """
    try:
        articles = list(articles_collection.find({}, {"_id": 1, "title": 1}))
        return {
            "articles": [
                {"id": str(article["_id"]), "title": article["title"]}
                for article in articles
            ]
        }
    except Exception as e:
        print(f"Error fetching articles: {e}")
        raise HTTPException(status_code=500, detail="Error fetching articles.")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)




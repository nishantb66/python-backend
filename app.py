from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from pymongo import MongoClient
from dotenv import load_dotenv
import groq
from bson import ObjectId

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


# Route to classify articles one by one
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



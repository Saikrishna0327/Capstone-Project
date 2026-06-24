import os
import random 
from typing import Literal
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load local environment variables from your secure .env file
load_dotenv()

# Initialize the GenAI Client
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client()

# Define the Blueprint structure the AI MUST follow using Pydantic
class MarketNewsOutput(BaseModel):
    ticker: str = Field(description="The exact stock ticker symbol picked from the available list.")
    headline: str = Field(description="A short, simple breaking news headline about this company.")
    sentiment: Literal["POSITIVE", "NEGATIVE", "NEUTRAL"] = Field(
        description="The market effect direction of this news event text on the stock price."
    )
    intensity: int = Field(
        description="The magnitude scale of market impact from 1 to 5."
    )

# Define the primary gateway function that app.py attempts to import
def generate_market_news(tickers):
    """
    Selects a ticker fairly using Python, asks Gemini for a short, clear headline,
    and returns a structured data record matching our Pydantic schema.
    """
    # Defensive programming check: If no tickers exist, provide fallback targets
    if not tickers:
        tickers = ["AAPL", "TSLA", "MSFT", "AMZN", "WMT"]

    #Python selects the target ticker explicitly here to treat all stocks equally
    chosen_ticker = random.choice(tickers)

    #  Strict rules forcing Gemini to use basic, clean English
    prompt = f"""
    You are an automated stock market news generator.
    Create a realistic breaking news headline for this exact stock ticker: {chosen_ticker}.
    
    CRITICAL INSTRUCTIONS FOR SIMPLE LANGUAGE:
    1. Write the headline using short, plain, everyday English.
    2. Avoid all complicated Wall Street jargon, technical economic phrases, or dense vocabulary.
    3. Keep it under 12 words total. It should read like a normal, easy-to-understand news alert.
    
    Tasks:
    1. Output the selected ticker symbol: {chosen_ticker}
    2. Write the short, simple headline about it.
    3. Determine the market sentiment (POSITIVE, NEGATIVE, or NEUTRAL).
    4. Gauge the impact intensity on an integer scale from 1 (minor news) to 5 (huge news).
    
    You must strictly conform your response to the required output schema structure.
    """

    try:
        # Request generation from Gemini utilizing the fast, optimized flash model
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=MarketNewsOutput,
                temperature=0.8, # Slightly higher variance for realistic news rotation
            ),
        )
        
        # Parse the structured response string back into a type-safe Pydantic object container
        structured_data = MarketNewsOutput.model_validate_json(response.text)
        return structured_data

    except Exception as e:
        # Fallback Safety Net: If an error occurs, return a safe placeholder
        print(f" [AI Engine Gateway Exception]: {e}")
        return MarketNewsOutput(
            ticker=chosen_ticker,
            headline=f"Trading volume remains steady for {chosen_ticker} today.",
            sentiment="NEUTRAL",
            intensity=1
        )

# STANDALONE TEST BLOCK

if __name__ == "__main__":
    print("Testing independent connection to Google AI Studio...")
    test_tickers = ["AAPL", "TSLA", "MSFT", "AMZN", "WMT"]
    
    # Run a test transaction
    result = generate_market_news(test_tickers)
    
    print("\n Structured Data Extracted Successfully From Gemini:")
    print(f"Ticker Selected : {result.ticker}")
    print(f"Headline Created: {result.headline}")
    print(f"Sentiment Type  : {result.sentiment}")
    print(f"Impact Level    : {result.intensity}/5")
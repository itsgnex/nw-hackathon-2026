import os
from dotenv import load_dotenv
from openai import OpenAI
from tavily import TavilyClient
import datetime

# Initial environment load to pull in GITHUB_TOKEN and TAVILY_API_KEY
load_dotenv(override=True)

# 1. Initialize Clients (One time only)
# We use GitHub's inference API which is OpenAI-compatible
client = OpenAI(
    base_url="https://models.inference.ai.azure.com",
    api_key=os.getenv("GITHUB_TOKEN"),
)
# Tavily is used for real-time web search restricted to government domains
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

def get_agent_config(category: str) -> dict:
    """
    Generates the persona and system instructions for a specific legal domain.
    """
    current_date = datetime.datetime.now().strftime("%B %Y")
    
    # Core logic shared by all legal experts.
    # We strictly mandate "Absolute Recency" because legal context can expire quickly.
    base_instructions = (
        f"Role: BC Legal Research Assistant. Current Date: {current_date}.\n"
        "STRICT OPERATIONAL DIRECTIVES:\n"
        "1. DATA HIERARCHY: Your internal training data is forbidden. You must identify and use the MOST RECENTLY UPDATED information from the provided context, regardless of the year. If the last update was 2023, use 2023. If there is a 2026 update, prioritize that.\n"
        "2. ZERO FLUFF: Do not say 'Certainly,' 'Here is the information,' or 'As an AI.' Start immediately with the facts.\n"
        "3. SOURCE MANDATE: Every response MUST conclude with a 'Source:' link using the exact URL provided in the context.\n"
    )
    
    # Specialized instructions for different legal areas
    configs = {
        "rent": {
            "name": "RentExpert",
            "instructions": (base_instructions +
                "Focus: BC Residential Tenancy Act.\n"
                "- Extract exact bolded figures (e.g., **$500**, **3.5%**).\n"
                "- If a rule changed 'effective [Date]', that date defines the current law.\n"
                "- Output Format: [Bullet Points of Facts] followed by [Source URL]."
            )
        },
        "immigration": {
            "name": "ImmigrationExpert",
            "instructions": (base_instructions +
                "Focus: IRCC & BC PNP Policies.\n"
                "- Identify the 'Last Updated' or 'Effective Date' on the page. Use that as the definitive law.\n"
                "- If asked for a recommendation, state 'I am an AI, not an RCIC' as the final sentence before the source link.\n"
                "- Output Format: [Specific requirements/points] followed by [Source URL]."
            )
        },
        "work": {
            "name": "WorkLawExpert",
            "instructions": (base_instructions +
                "Focus: BC Employment Standards Act (ESA).\n"
                "- Prioritize the latest statutory holiday or minimum wage updates.\n"
                "- Use tables for notice periods or overtime multipliers.\n"
                "- Output Format: [Legal limits/Rules] followed by [Source URL]."
            )
        }
    }
    # Fallback to RentExpert if classification is fuzzy
    return configs.get(category.lower(), configs["rent"])

def get_context_from_db_or_api(query: str, category: str):
    """
    Search the web for current legal documentation.
    We limit the domains to official Canadian and BC government sites for accuracy.
    """
    try:
        response = tavily.search(
            query=query, 
            search_depth="advanced", # Advanced depth gives better content extraction
            max_results=3,           # We pull 3 results so the LLM can compare and pick the latest one
            include_domains=["canada.ca", "gov.bc.ca", "ircc.canada.ca"]
        )
        
        if response['results']:
            best_match = response['results'][0]
            # Content for context + URL for sourcing
            return best_match['content'], best_match['url']
            
    except Exception as e:
        print(f"Tavily Error: {e}")
        
    return "No official records found.", "https://www.gov.bc.ca"


def classify_intent(user_input: str) -> str:
    """
    Determines which legal department the user's question belongs to.
    This routes the query to the correct system prompt.
    """
    try:
        system_prompt = (
            "You are a classification assistant for a BC legal bot. "
            "Categorize the user's query into one of these three labels: 'rent', 'work', 'immigration' or 'other.\n"
            "Rules:\n"
            "- 'rent': for anything related to housing, landlords, tenants, or evictions.\n"
            "- 'work': for anything related to jobs, wages, labor rights, or firing.\n"
            "- 'immigration': for visas, PNP, PR, or work permits.\n"
            "- 'other': for general chat, greetings\n"
            "- Respond with ONLY the word (e.g., 'work')."
        )

        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Classify this query: {user_input}"}
            ],
            model=os.getenv("GITHUB_MODEL", "gpt-4o"), # Using gpt-4o for high classification accuracy
            temperature=0 # Absolute precision
        )
        
        # Clean up the response to get just the keyword
        category = response.choices[0].message.content.strip().lower()
        
        valid_categories = ["rent", "work", "immigration"]
        return category if category in valid_categories else "other"
    except Exception as e:
        print(f"Error in classify_intent: {str(e)}")
        return "rent"  # Safe default fallback

def legal_bot_response(user_input: str, category: str):
    """
    Main RAG (Retrieval-Augmented Generation) pipeline:
    1. Retrieval: Get latest context from official sites.
    2. Generation: Produce a structured answer based strictly on that context.
    """
    try:
        config = get_agent_config(category)
        
        # SEARCH QUERY: We pivot the search to look for 'latest' updates specifically.
        search_query = f"latest official BC {category} law update rules {user_input} site:gov.bc.ca OR site:canada.ca"
        
        # Step 1: Context Retrieval
        context_text, source_url = get_context_from_db_or_api(search_query, category)
        
        # Step 2: Final prompt assembly with the injected context
        prompt_content = f"""
        CONTEXT DATA:
        {context_text}
        
        SOURCE URL: {source_url}
        
        USER QUESTION: {user_input}
        
        TASK:
        1. Scan the CONTEXT DATA for the most recent date mentioned (e.g., 'As of Jan 2025', 'Effective 2026').
        2. Extract the rules/numbers associated with that LATEST date.
        3. Provide the answer in bullet points.
        4. End the response with: Source: {source_url}
        """

        # Step 3: Call the LLM with the specialist's persona
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": config["instructions"]},
                {"role": "user", "content": prompt_content}
            ],
            model=os.getenv("GITHUB_MODEL", "gpt-4o"),
            temperature=0
        )
        
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error in legal_bot_response: {e}")
        raise


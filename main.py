import os
import json
import logging
import uuid
import datetime
from typing import Optional, Dict, List, Any
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from pypdf import PdfReader
from dotenv import load_dotenv
from groq import Groq

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load local .env if present
load_dotenv()

app = FastAPI(
    title="Startup Intelligence Agent API",
    description="Backend API for automated startup idea validation and fundraising readiness assessment",
    version="1.3.0"
)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database file configuration
HISTORY_FILE = "history.json"

# In-memory configuration store for local session persistence (fallback if .env is write-protected)
CONFIG = {
    "groq_api_key": os.getenv("GROQ_API_KEY", "")
}

class StartupProfile(BaseModel):
    name: str
    industry: str
    problem: str
    solution: str
    business_model: str
    target_market: str
    team: str
    product_stage: str
    monthly_revenue: str
    customer_count: str
    growth_rate: str
    funding_stage: str
    competitors: str

class IdeaValidationProfile(BaseModel):
    name: str  # Idea / Project Name
    industry: str
    idea: str
    problem: str
    target_customer: str
    revenue_model: str

class SettingsUpdateRequest(BaseModel):
    groq_api_key: str

# Helper functions for database persistence
def load_history() -> List[Dict[str, Any]]:
    """Loads all records from the history JSON file."""
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error reading history.json: {e}")
        return []

def save_history(records: List[Dict[str, Any]]):
    """Saves all records back to the history JSON file."""
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Error writing to history.json: {e}")

def get_groq_client(custom_key: Optional[str] = None) -> Groq:
    """Helper to initialize Groq client using custom key, env var, or CONFIG."""
    api_key = custom_key or CONFIG.get("groq_api_key") or os.getenv("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=400,
            detail="Groq API Key not found. Please provide it in the settings panel or set GROQ_API_KEY in the environment."
        )
    return Groq(api_key=api_key)

@app.get("/api/config")
async def get_config():
    """Check if the Groq API Key is configured."""
    has_key = bool(CONFIG.get("groq_api_key") or os.getenv("GROQ_API_KEY"))
    return {"configured": has_key}

@app.post("/api/config")
async def update_config(req: SettingsUpdateRequest):
    """Dynamically save the Groq API Key for the active session and write to .env if writable."""
    CONFIG["groq_api_key"] = req.groq_api_key
    try:
        with open(".env", "w") as f:
            f.write(f"GROQ_API_KEY={req.groq_api_key}\n")
        logger.info("Successfully updated API Key in local .env file")
    except Exception as e:
        logger.warning(f"Could not write to .env (expected in some permission environments): {e}")
    return {"status": "success", "message": "API Key saved successfully."}

@app.post("/api/extract-deck")
async def extract_deck(
    file: UploadFile = File(...),
    x_groq_api_key: Optional[str] = Header(None)
):
    """
    Extracts text from an uploaded pitch deck PDF and uses Llama 3.3 70B to pre-fill the startup form.
    """
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    try:
        # 1. Read PDF content
        try:
            pdf_reader = PdfReader(file.file)
            text_content = ""
            # Limit extraction to first 15 pages to stay within token/processing boundaries
            pages_to_read = min(len(pdf_reader.pages), 15)
            for page_idx in range(pages_to_read):
                text = pdf_reader.pages[page_idx].extract_text()
                if text:
                    text_content += f"\n--- Page {page_idx + 1} ---\n" + text
        except Exception as pdf_err:
            raise HTTPException(status_code=400, detail=f"Could not extract text from the PDF: {str(pdf_err)}")
                
        if not text_content.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from the PDF. The file might contain only images.")
        
        # 2. Setup Groq Client
        client = get_groq_client(x_groq_api_key)
        
        # 3. Request Llama 3.3 70B to parse and map properties
        system_prompt = (
            "You are an expert startup analyst. You extract structured company profiles from raw pitch deck text. "
            "Your task is to analyze the text and extract details into the specified JSON format. "
            "Fill in empty strings for fields that cannot be inferred from the text. Be concise but descriptive."
        )
        
        user_prompt = f"""Analyze the following pitch deck text extract and map it to these exact 13 keys:
- name (Startup Name)
- industry (General sector, e.g., Fintech, B2B SaaS, Healthtech)
- problem (Problem statement being solved)
- solution (The startup's core product/service solution)
- business_model (How they make money, pricing strategy)
- target_market (Target audience, market size/TAM if available)
- team (Key founders, advisors, and their backgrounds)
- product_stage (e.g., Idea, Prototype, MVP, Launching, Scaling)
- monthly_revenue (e.g., $0, $5k MRR, $50k MRR or 'Not mentioned')
- customer_count (Number of users or clients or 'Not mentioned')
- growth_rate (e.g., 10% MoM, 2x YoY or 'Not mentioned')
- funding_stage (e.g., Bootstrapped, Pre-Seed, Seed, Series A)
- competitors (Competitors listed or competitor analysis)

Pitch Deck Text Extract (Truncated):
{text_content[:8000]}

Response format:
Respond ONLY with a valid JSON object matching the keys above. Do not include markdown codeblocks (like ```json), introduction, or extra comments.
"""
        
        model_name = "llama-3.3-70b-versatile"
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model=model_name,
                temperature=0.1,
                response_format={"type": "json_object"}
            )
        except Exception as api_err:
            logger.warning(f"Error calling {model_name}, trying llama3-70b-8192: {api_err}")
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model="llama3-70b-8192",
                temperature=0.1,
                response_format={"type": "json_object"}
            )

        raw_response = chat_completion.choices[0].message.content
        extracted_data = json.loads(raw_response)
        
        # Ensure all required keys exist
        expected_keys = [
            "name", "industry", "problem", "solution", "business_model", 
            "target_market", "team", "product_stage", "monthly_revenue", 
            "customer_count", "growth_rate", "funding_stage", "competitors"
        ]
        response_data = {k: extracted_data.get(k, "") for k in expected_keys}
        
        return response_data

    except HTTPException as he:
        raise he
    except json.JSONDecodeError as jde:
        logger.error(f"JSON Decode Error from Groq response: {jde}")
        raise HTTPException(status_code=500, detail="Failed to parse structured JSON data from Groq analysis.")
    except Exception as e:
        logger.error(f"Error extracting pitch deck: {e}")
        raise HTTPException(status_code=500, detail=f"Error parsing pitch deck: {str(e)}")

@app.post("/api/extract-idea")
async def extract_idea(
    file: UploadFile = File(...),
    x_groq_api_key: Optional[str] = Header(None)
):
    """
    Extracts text from an uploaded idea document PDF and uses Llama 3.3 70B to pre-fill the startup idea form.
    If details are missing, it formulates tailored suggestions.
    """
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    try:
        # 1. Read PDF content
        try:
            pdf_reader = PdfReader(file.file)
            text_content = ""
            # Limit extraction to first 15 pages to stay within token/processing boundaries
            pages_to_read = min(len(pdf_reader.pages), 15)
            for page_idx in range(pages_to_read):
                text = pdf_reader.pages[page_idx].extract_text()
                if text:
                    text_content += f"\n--- Page {page_idx + 1} ---\n" + text
        except Exception as pdf_err:
            raise HTTPException(status_code=400, detail=f"Could not extract text from the PDF: {str(pdf_err)}")
                
        if not text_content.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from the PDF. The file might contain only images.")
        
        # 2. Setup Groq Client
        client = get_groq_client(x_groq_api_key)
        
        # 3. Request Llama 3.3 70B to parse and map properties
        system_prompt = (
            "You are an expert startup analyst, business model designer, and incubator director. "
            "You analyze early-stage startup idea notes, proposals, and concept documents, and extract key variables. "
            "Your task is to analyze the text and extract details into the specified JSON format. "
            "If any information is not explicitly mentioned in the text (e.g. expected revenue model or target customers), "
            "use your startup expertise to suggest a highly logical, standard value for that field based on the rest of the text. "
            "You must also assign a confidence level ('High', 'Medium', or 'Low') for each extracted/suggested field, "
            "based on whether the information was explicitly found in the text or suggested by you."
        )
        
        user_prompt = f"""Analyze the following startup idea text extract and map it to these exact 6 keys:
- name (Idea / Project Name)
- industry (General sector, e.g., GreenTech, B2B SaaS, Healthtech)
- idea (Core startup idea / product value proposition)
- problem (Problem statement being solved)
- target_customer (Target customer segment)
- revenue_model (Expected revenue model)

And include a nested object 'confidences' containing a confidence score ('High', 'Medium', or 'Low') for each of the 6 fields above:
- name
- industry
- idea
- problem
- target_customer
- revenue_model

If you had to formulate a suggestion for a field because it wasn't explicitly mentioned, assign 'Low' confidence to it. Otherwise, use 'High' or 'Medium'.

Startup Idea Text Extract (Truncated):
{text_content[:8000]}

Response format:
Respond ONLY with a valid JSON object matching the specification above. Do not include markdown codeblocks (like ```json), introduction, or extra comments.
"""
        
        model_name = "llama-3.3-70b-versatile"
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model=model_name,
                temperature=0.1,
                response_format={"type": "json_object"}
            )
        except Exception as api_err:
            logger.warning(f"Error calling {model_name}, trying llama3-70b-8192: {api_err}")
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model="llama3-70b-8192",
                temperature=0.1,
                response_format={"type": "json_object"}
            )

        raw_response = chat_completion.choices[0].message.content
        extracted_data = json.loads(raw_response)
        
        # Ensure all required keys exist
        expected_keys = ["name", "industry", "idea", "problem", "target_customer", "revenue_model"]
        response_data = {k: extracted_data.get(k, "") for k in expected_keys}
        
        # Ensure confidences exist
        extracted_conf = extracted_data.get("confidences", {})
        response_data["confidences"] = {k: extracted_conf.get(k, "Low") for k in expected_keys}
        
        return response_data

    except HTTPException as he:
        raise he
    except json.JSONDecodeError as jde:
        logger.error(f"JSON Decode Error from Groq response in extract_idea: {jde}")
        raise HTTPException(status_code=500, detail="Failed to parse structured JSON data from Groq analysis.")
    except Exception as e:
        logger.error(f"Error extracting startup idea: {e}")
        raise HTTPException(status_code=500, detail=f"Error parsing startup idea PDF: {str(e)}")

@app.post("/api/evaluate")
async def evaluate_startup(
    profile: StartupProfile,
    x_groq_api_key: Optional[str] = Header(None)
):
    """
    Evaluates the startup details against investor criteria, generates
    scores, and automatically appends the assessment to the history log.
    """
    try:
        client = get_groq_client(x_groq_api_key)
        
        system_prompt = (
            "You are a sophisticated Venture Capital (VC) general partner and startup readiness advisor. "
            "You perform rigorous, objective, and constructive evaluations of early-stage startups. "
            "Your feedback must be insightful, highlighting critical gaps while giving tactical recommendations. "
            "You MUST output your response as a single, valid JSON object containing specific fields."
        )
        
        user_prompt = f"""Perform a complete, professional investor evaluation of the following startup profile.

Startup Profile Details:
- Startup Name: {profile.name}
- Industry: {profile.industry}
- Problem Statement: {profile.problem}
- Solution: {profile.solution}
- Business Model: {profile.business_model}
- Target Market: {profile.target_market}
- Team Details: {profile.team}
- Product Stage: {profile.product_stage}
- Monthly Revenue: {profile.monthly_revenue}
- Customer Count: {profile.customer_count}
- Growth Rate: {profile.growth_rate}
- Funding Stage: {profile.funding_stage}
- Competitor Information: {profile.competitors}

You must return a JSON object with this exact structure:
{{
  "overall_score": <Integer from 0 to 100 representing fundraising readiness>,
  "category_scores": {{
    "problem_solution_fit": <Integer 0-100>,
    "market_opportunity": <Integer 0-100>,
    "team_strength": <Integer 0-100>,
    "product_maturity": <Integer 0-100>,
    "traction": <Integer 0-100>,
    "revenue_potential": <Integer 0-100>,
    "scalability": <Integer 0-100>,
    "competitive_advantage": <Integer 0-100>
  }},
  "category_analyses": {{
    "problem_solution_fit": "<Brief professional critique, highlighting positive and negative aspects>",
    "market_opportunity": "<Analysis of TAM, target user validation, and headroom>",
    "team_strength": "<Critique of team backgrounds, expertise gaps, and key execution risks>",
    "product_maturity": "<Evaluation of product stage relative to funding stage goals>",
    "traction": "<Assessment of current revenue, customers, and growth momentum>",
    "revenue_potential": "<Critique of business model viability and future scalability limits>",
    "scalability": "<Analysis of operational leverage and technology scaling capabilities>",
    "competitive_advantage": "<Critique of defensibility, barriers to entry, and competitor awareness>"
  }},
  "swot": {{
    "strengths": ["<Strength 1>", "<Strength 2>", "<Strength 3>"],
    "weaknesses": ["<Weakness 1>", "<Weakness 2>", "<Weakness 3>"],
    "opportunities": ["<Opportunity 1>", "<Opportunity 2>", "<Opportunity 3>"],
    "threats": ["<Threat 1>", "<Threat 2>", "<Threat 3>"]
  }},
  "gaps": [
    {{
      "area": "<e.g., Traction or Team or Market>",
      "issue": "<Description of the specific weakness or missing detail>",
      "severity": "<High, Medium, or Low>",
      "recommendation": "<Direct actionable remedy>"
    }}
  ],
  "investor_questions": [
    {{
      "question": "<High-probability question a VC would ask during due diligence>",
      "explanation": "<Why the investor is asking this question (their underlying concern)>",
      "key_to_answer": "<Specific metrics, proof, or narrative strategy the founder should prepare to answer successfully>"
    }}
  ],
  "recommendations": [
    "<High-impact actionable step 1>",
    "<High-impact actionable step 2>",
    "<High-impact actionable step 3>",
    "<High-impact actionable step 4>",
    "<High-impact actionable step 5>"
  ],
  "roadmap": {{
    "30_days": ["<Action 1>", "<Action 2>", "<Action 3>"],
    "60_days": ["<Action 1>", "<Action 2>", "<Action 3>"],
    "90_days": ["<Action 1>", "<Action 2>", "<Action 3>"]
  }}
}}

Make sure:
1. Category scores average out logically to match the overall score.
2. Under gaps, identify at least 4 items.
3. Under investor_questions, compile 6 to 8 realistic questions.
4. Under roadmap, list exactly 3 actions for each timeline (30, 60, and 90 days).
5. Return ONLY the JSON object. Do not wrap the JSON output in markdown ```json or include text outside the JSON.
"""

        model_name = "llama-3.3-70b-versatile"
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model=model_name,
                temperature=0.2,
                response_format={"type": "json_object"}
            )
        except Exception as api_err:
            logger.warning(f"Error calling {model_name}, trying llama3-70b-8192: {api_err}")
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model="llama3-70b-8192",
                temperature=0.2,
                response_format={"type": "json_object"}
            )

        raw_response = chat_completion.choices[0].message.content
        report_data = json.loads(raw_response)

        # Automatically log to history database
        try:
            record = {
                "id": str(uuid.uuid4()),
                "name": profile.name,
                "date": datetime.datetime.now().isoformat(),
                "type": "Funding Readiness",
                "overall_score": report_data.get("overall_score", 0),
                "category_scores": report_data.get("category_scores", {}),
                "profile": profile.dict(),
                "report": report_data
            }
            history = load_history()
            history.append(record)
            save_history(history)
            logger.info(f"Saved assessment history for startup: {profile.name}")
        except Exception as save_err:
            logger.error(f"Error auto-saving assessment to history: {save_err}")

        return report_data

    except json.JSONDecodeError as jde:
        logger.error(f"JSON Decode Error from Groq response during evaluation: {jde}")
        raise HTTPException(status_code=500, detail="Failed to parse structured JSON report from Groq API.")
    except Exception as e:
        logger.error(f"Error during startup evaluation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/validate-idea")
async def validate_idea(
    profile: IdeaValidationProfile,
    x_groq_api_key: Optional[str] = Header(None)
):
    """
    Validates a raw startup idea against viability criteria, generates scores,
    and automatically appends the validation assessment to the history log.
    """
    try:
        client = get_groq_client(x_groq_api_key)
        
        system_prompt = (
            "You are a startup incubator director, business model strategist, and expert validator. "
            "You critically assess early-stage startup ideas to check viability, market demand, and business model fit. "
            "You provide highly tactical feedback, outlining risks, suggesting business models, and building roadmap validation plans. "
            "You MUST output your response as a single, valid JSON object containing specific fields."
        )
        
        user_prompt = f"""Perform a comprehensive viability evaluation of the following startup idea.

Startup Idea Details:
- Project / Idea Name: {profile.name}
- Industry: {profile.industry}
- Core Startup Idea: {profile.idea}
- Problem Being Solved: {profile.problem}
- Target Customer Segment: {profile.target_customer}
- Expected Revenue Model: {profile.revenue_model}

Evaluate the startup idea against the following 8 criteria, grading each from 0-100:
1. Problem-Solution Fit (fit of the solution to the identified customer pain)
2. Market Opportunity (estimated TAM, growth potential of the segment)
3. Execution Complexity (difficulty in building, compliance barriers, resource needs)
4. Business Model Viability (scalability of the expected revenue stream)
5. Competitor Density (crowdedness, ease of customer acquisition)
6. Revenue Potential (likelihood of capturing high margin, pricing power)
7. Growth Potential (opportunities to scale up operations and leverage network effects)
8. Defensibility (IP, data moats, locking mechanisms)

You must return a JSON object with this exact structure:
{{
  "overall_score": <Integer from 0 to 100 representing startup idea viability>,
  "category_scores": {{
    "problem_solution_fit": <Integer 0-100 for Problem-Solution Fit>,
    "market_opportunity": <Integer 0-100 for Market Opportunity>,
    "team_strength": <Integer 0-100 for Execution Complexity>,
    "product_maturity": <Integer 0-100 for Business Model Viability>,
    "traction": <Integer 0-100 for Competitor Density>,
    "revenue_potential": <Integer 0-100 for Revenue Potential>,
    "scalability": <Integer 0-100 for Growth Potential>,
    "competitive_advantage": <Integer 0-100 for Defensibility>
  }},
  "category_analyses": {{
    "problem_solution_fit": "<Critique of the solution relative to target customer pain points>",
    "market_opportunity": "<Assessment of target customer segment size, access, and scale headroom>",
    "team_strength": "<Critique of execution barriers, build complexity, and technical requirements>",
    "product_maturity": "<Critique of the expected revenue model and suggestions for improvement>",
    "traction": "<Critique of competitor presence, direct options, and market entry barriers>",
    "revenue_potential": "<Analysis of pricing limits, margin potential, and customer lifetime value potential>",
    "scalability": "<Analysis of distribution leverage, marginal unit cost behavior, and scale potential>",
    "competitive_advantage": "<Suggestions on how to build defensibility, barriers to entry, or data moats>"
  }},
  "swot": {{
    "strengths": ["<Strength 1>", "<Strength 2>", "<Strength 3>"],
    "weaknesses": ["<Weakness 1>", "<Weakness 2>", "<Weakness 3>"],
    "opportunities": ["<Opportunity 1>", "<Opportunity 2>", "<Opportunity 3>"],
    "threats": ["<Threat 1>", "<Threat 2>", "<Threat 3>"]
  }},
  "gaps": [
    {{
      "area": "<e.g., Market Validation or Technical Risk or Defensibility>",
      "issue": "<Specific risk, challenge, or critical assumption that remains unproven>",
      "severity": "<High, Medium, or Low>",
      "recommendation": "<Actionable way to validate or mitigate this risk>"
    }}
  ],
  "investor_questions": [
    {{
      "question": "<Validation question a founder must answer during testing or interviews>",
      "explanation": "<Why answering this question is critical to verifying the business idea>",
      "key_to_answer": "<Specific validation experiment, landing page metric, or survey tactic to get this proof>"
    }}
  ],
  "recommendations": [
    "<High-impact validation recommendation 1>",
    "<High-impact validation recommendation 2>",
    "<High-impact validation recommendation 3>",
    "<High-impact validation recommendation 4>",
    "<High-impact validation recommendation 5>"
  ],
  "roadmap": {{
    "30_days": ["<Action 1>", "<Action 2>", "<Action 3>"],
    "60_days": ["<Action 1>", "<Action 2>", "<Action 3>"],
    "90_days": ["<Action 1>", "<Action 2>", "<Action 3>"]
  }}
}}

Make sure:
1. Category scores average out logically to match the overall score.
2. Under gaps, identify at least 4 critical risks and challenges.
3. Under investor_questions, compile 6 to 8 validation/testing questions.
4. Under roadmap, list exactly 3 actions to test the idea for each timeframe (30, 60, and 90 days).
5. Return ONLY the JSON object. Do not wrap the JSON output in markdown ```json or include text outside the JSON.
"""

        model_name = "llama-3.3-70b-versatile"
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model=model_name,
                temperature=0.2,
                response_format={"type": "json_object"}
            )
        except Exception as api_err:
            logger.warning(f"Error calling {model_name}, trying llama3-70b-8192: {api_err}")
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model="llama3-70b-8192",
                temperature=0.2,
                response_format={"type": "json_object"}
            )

        raw_response = chat_completion.choices[0].message.content
        report_data = json.loads(raw_response)

        # Automatically log to history database
        try:
            record = {
                "id": str(uuid.uuid4()),
                "name": profile.name,
                "date": datetime.datetime.now().isoformat(),
                "type": "Idea Validation",
                "overall_score": report_data.get("overall_score", 0),
                "category_scores": report_data.get("category_scores", {}),
                "profile": profile.dict(),
                "report": report_data
            }
            history = load_history()
            history.append(record)
            save_history(history)
            logger.info(f"Saved validation history for idea: {profile.name}")
        except Exception as save_err:
            logger.error(f"Error auto-saving validation to history: {save_err}")

        return report_data

    except json.JSONDecodeError as jde:
        logger.error(f"JSON Decode Error from Groq response during validation: {jde}")
        raise HTTPException(status_code=500, detail="Failed to parse structured JSON report from Groq API.")
    except Exception as e:
        logger.error(f"Error during idea validation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# History Management API Routes
@app.get("/api/history")
async def get_history():
    """Returns summaries of all saved assessments (for tables and charts)."""
    history = load_history()
    summaries = []
    for r in history:
        summaries.append({
            "id": r.get("id"),
            "name": r.get("name"),
            "date": r.get("date"),
            "type": r.get("type"),
            "overall_score": r.get("overall_score", 0),
            "category_scores": r.get("category_scores", {})
        })
    return summaries

@app.get("/api/history/{record_id}")
async def get_history_detail(record_id: str):
    """Retrieves full details of a specific assessment."""
    history = load_history()
    for r in history:
        if r.get("id") == record_id:
            return r
    raise HTTPException(status_code=404, detail="Assessment record not found")

@app.delete("/api/history/{record_id}")
async def delete_history_record(record_id: str):
    """Deletes a specific assessment record from history."""
    history = load_history()
    updated = [r for r in history if r.get("id") != record_id]
    if len(updated) == len(history):
        raise HTTPException(status_code=404, detail="Assessment record not found")
    save_history(updated)
    return {"status": "success", "message": "Record deleted successfully"}

@app.post("/api/export-report")
async def export_report(data: Dict[str, Any]):
    """
    Generates a beautifully formatted Markdown report file that the client can download.
    """
    try:
        startup_name = data.get("name", "Startup")
        overall_score = data.get("overall_score", 0)
        mode = data.get("mode", "funding")  # "funding" or "idea"
        
        if mode == "idea":
            title = "STARTUP IDEA VALIDATION REPORT"
            score_label = "Overall Startup Viability Score"
            category_mapping = {
                "problem_solution_fit": "Problem-Solution Fit",
                "market_opportunity": "Market Opportunity",
                "team_strength": "Execution Complexity",
                "product_maturity": "Business Model Viability",
                "traction": "Competitor Density",
                "revenue_potential": "Revenue Potential",
                "scalability": "Growth Potential",
                "competitive_advantage": "Defensibility"
            }
            gaps_title = "RISKS & CHALLENGES DETECTED"
            q_title = "CRITICAL VALIDATION QUESTIONS"
            rec_title = "VALIDATION RECOMMENDATIONS"
            roadmap_title = "NEXT STEPS VALIDATION ROADMAP (90-DAY PREPARATION TIMELINE)"
        else:
            title = "FUNDING READINESS REPORT"
            score_label = "Overall Funding Readiness Score"
            category_mapping = {
                "problem_solution_fit": "Problem-Solution Fit",
                "market_opportunity": "Market Opportunity",
                "team_strength": "Team Strength",
                "product_maturity": "Product Maturity",
                "traction": "Traction",
                "revenue_potential": "Revenue Potential",
                "scalability": "Scalability",
                "competitive_advantage": "Competitive Advantage"
            }
            gaps_title = "GAP ANALYSIS & BOTTLENECK DETECTION"
            q_title = "PREPARED INVESTOR QUESTIONS (DUE DILIGENCE SIMULATOR)"
            rec_title = "STRATEGIC RECOMMENDATIONS"
            roadmap_title = "ACTION ROADMAP (90-DAY PREPARATION TIMELINE)"

        md_content = f"""# {title}
## Project: {startup_name}
## {score_label}: {overall_score}/100

Generated on: {datetime.datetime.now().strftime('%Y-%m-%d')} (Automated AI Evaluation)

---

## 1. CATEGORY SCORE SUMMARY
"""
        
        cat_scores = data.get("category_scores", {})
        cat_analyses = data.get("category_analyses", {})
        
        for key, name in category_mapping.items():
            score = cat_scores.get(key, 0)
            analysis = cat_analyses.get(key, "No analysis provided.")
            md_content += f"### {name}: {score}/100\n{analysis}\n\n"
            
        md_content += "---\n\n## 2. SWOT ANALYSIS\n\n"
        swot = data.get("swot", {})
        
        for aspect in ["strengths", "weaknesses", "opportunities", "threats"]:
            items = swot.get(aspect, [])
            md_content += f"### {aspect.capitalize()}\n"
            for item in items:
                md_content += f"- {item}\n"
            md_content += "\n"
            
        md_content += f"---\n\n## 3. {gaps_title}\n\n"
        gaps = data.get("gaps", [])
        for gap in gaps:
            md_content += f"### Area: {gap.get('area', 'General')} (Severity: {gap.get('severity', 'Medium')})\n"
            md_content += f"**Issue/Risk:** {gap.get('issue', '')}\n"
            md_content += f"**Remedy:** {gap.get('recommendation', '')}\n\n"
            
        md_content += f"---\n\n## 4. {q_title}\n\n"
        questions = data.get("investor_questions", [])
        for idx, q in enumerate(questions, 1):
            md_content += f"### Q{idx}: {q.get('question', '')}\n"
            md_content += f"*Why this is asked/critical:* {q.get('explanation', '')}\n"
            md_content += f"*How to validate/answer:* {q.get('key_to_answer', '')}\n\n"
            
        md_content += f"---\n\n## 5. {rec_title}\n\n"
        recs = data.get("recommendations", [])
        for rec in recs:
            md_content += f"- {rec}\n"
            
        md_content += f"\n---\n\n## 6. {roadmap_title}\n\n"
        roadmap = data.get("roadmap", {})
        
        for period in ["30_days", "60_days", "90_days"]:
            clean_period = period.replace("_", " ").title()
            items = roadmap.get(period, [])
            md_content += f"### {clean_period}\n"
            for item in items:
                md_content += f"- {item}\n"
            md_content += "\n"
            
        # Write temporary markdown report to scratch space
        report_filename = f"Viability_Report_{startup_name.replace(' ', '_')}.md" if mode == "idea" else f"Readiness_Report_{startup_name.replace(' ', '_')}.md"
        report_path = os.path.join(os.getcwd(), report_filename)
        
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(md_content)
            
        return {"filename": report_filename, "content": md_content}
        
    except Exception as e:
        logger.error(f"Error exporting report: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Mount static files (served at the root path or inside React SPA)
if os.path.exists("static"):
    app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    # Start on localhost:8000
    uvicorn.run(app, host="127.0.0.1", port=8000)

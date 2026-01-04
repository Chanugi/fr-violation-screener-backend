from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os, json
import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

load_dotenv()

app = FastAPI(title="AI Fundamental Rights Violation Screener")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# ---------- GLOBAL HOLDERS (EMPTY INIT) ----------
embed_model = None
index = None
articles = None
gemini = None


class ScenarioInput(BaseModel):
    scenario: str


# ---------- INITIALIZE AI (ON FIRST REQUEST) ----------
def init_ai():
    global embed_model, index, articles, gemini

    if embed_model is not None:
        return  # already loaded

    print("🔄 Loading AI models...")

    # Load data
    with open("data/cleaned_constitution_articles.json", encoding="utf-8") as f:
        articles = json.load(f)

    # Embeddings
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [
        f"{a['article_id']} - {a.get('summary','')} - {a.get('text','')}"
        for a in articles
    ]
    embeddings = embed_model.encode(texts, convert_to_numpy=True)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # Gemini: prefer gemini-2.5-flash first (user confirmed it's working)
    genai.configure(api_key=GOOGLE_API_KEY)
    preferred = [
        "models/gemini-2.5-flash",
        "models/gemini-1.5-flash",
        "models/gemini-1.5-pro",
        "models/gemini-pro",
    ]

    model_name = None
    # Try preferred models directly first (fast path)
    for p in preferred:
        try:
            _tmp = genai.GenerativeModel(p)
            model_name = p
            print("Using preferred model:", p)
            break
        except Exception as e:
            print("Preferred model not available:", p, e)

    # If none of the preferred worked, fall back to ListModels discovery
    if model_name is None:
        try:
            models_info = genai.list_models()
            models_list = models_info.get("models") if isinstance(models_info, dict) and "models" in models_info else models_info
            available = []
            if isinstance(models_list, list):
                for m in models_list:
                    name = m.get("name") if isinstance(m, dict) else getattr(m, "name", None)
                    supported = None
                    if isinstance(m, dict):
                        supported = m.get("supported_generation_methods") or m.get("supportedMethods")
                    else:
                        supported = getattr(m, "supported_generation_methods", None) or getattr(m, "supportedMethods", None)

                    supports_generate = False
                    if supported is not None:
                        s = supported if isinstance(supported, (list, tuple)) else [supported]
                        for item in s:
                            if "generate" in str(item).lower() or "content" in str(item).lower() or "text" in str(item).lower() or "chat" in str(item).lower():
                                supports_generate = True
                                break

                    if name and supports_generate:
                        available.append((name, m))

            if available:
                print("Available Gemini models:")
                for n, _ in available:
                    print(" -", n)
                for p in preferred:
                    if p in [n for n, _ in available]:
                        model_name = p
                        break
                if model_name is None:
                    model_name = available[0][0]
        except Exception as e:
            print("ListModels failed:", e)

    if model_name is None:
        raise Exception("No suitable generative model available. Run ListModels to see options.")

    gemini = genai.GenerativeModel(model_name)
    print("Using Gemini model:", model_name)

    print("✅ AI models loaded successfully")


def search_relevant_articles(user_input, top_k=3):
    query_embedding = embed_model.encode([user_input])
    _, indices = index.search(query_embedding, top_k)
    return [articles[i] for i in indices[0]]


def build_prompt(user_scenario, matched_articles):
    article_text = ""
    for art in matched_articles:
        article_text += f"""
Article: {art['article_id']}
Summary: {art.get('summary','')}
Full Text: {art.get('text','')}
---
"""

    # Provide an explicit plain-text template matching the Colab output
    return f"""
You are a legal assistant specialized in Sri Lankan Fundamental Rights.

USER SCENARIO:
"{user_scenario}"

RELEVANT CONSTITUTION ARTICLES:
{article_text}

TASK:
- Decide whether this is a Fundamental Rights violation or not.
- If YES, state the violated Article(s) using the format: ARTICLE <number> – <short summary title>.
- Provide a short plain-language Explanation (1-3 short paragraphs).
- Provide "What the person can do next:" with practical steps and mention Article 17 remedy if applicable.

RESPONSE FORMAT (STRICT):
Produce ONLY plain text (no markdown, no bold, no lists characters like '*', no HTML). Use the exact headings below followed by content.

Violation Status: Yes or No

Violated Article(s):
ARTICLE <number> – <short summary title>

Explanation:
<one or two short paragraphs explaining why this situation does or does not violate the Article>

What the person can do next:
<practical next steps; mention Article 17 remedy if applicable and simple evidence-collection suggestions>

Keep language simple and concise, suitable for ordinary citizens. Do not include extra sections or commentary.
"""


def normalize_analysis_text(text: str) -> str:
    """If the model determines there is no violation, replace the verbose
    'What the person can do next' section with a short, practical note.
    """
    if text is None:
        return text

    # Detect a clear 'Violation Status: No' (case-insensitive)
    if re.search(r"Violation Status:\\s*No\\b", text, re.I):
        # Normalize the Violation Status line
        text = re.sub(r"(?i)Violation Status:.*", "Violation Status: No", text, count=1)

        # Set a concise Explanation
        if re.search(r"(?s)Explanation:\\s*.*?(?=\\n\\nWhat the person can do next:|\\Z)", text):
            text = re.sub(
                r"(?s)Explanation:\\s*.*?(?=\\n\\nWhat the person can do next:|\\Z)",
                "Explanation:\nNo fundamental rights violation detected.",
                text,
            )
        else:
            text = text.strip() + "\\n\\nExplanation:\nNo fundamental rights violation detected."

        # Replace or add short action note
        if re.search(r"(?s)What the person can do next:\\s*.*", text):
            text = re.sub(
                r"(?s)What the person can do next:\\s*.*",
                "What the person can do next:\\nNo fundamental rights violation detected; no immediate action required.",
                text,
            )
        else:
            text = text.strip() + "\\n\\nWhat the person can do next:\\nNo fundamental rights violation detected; no immediate action required."

    return text


@app.post("/analyze")
def analyze(data: ScenarioInput):
    init_ai()  # 👈 CRITICAL FIX
    matched_articles = search_relevant_articles(data.scenario)
    prompt = build_prompt(data.scenario, matched_articles)
    try:
        response = gemini.generate_content(prompt)
        analysis_text = getattr(response, "text", str(response))
    except Exception as e:
        print("Generative API error:", e)
        raise HTTPException(status_code=500, detail=str(e))

    return {
        #"matched_articles": matched_articles,
        "analysis": analysis_text
    }

@app.post("/screen-scenario")
def screen_scenario(data: ScenarioInput):
    """Screen a scenario for FR violations and return structured results"""
    init_ai()
    
    # Search for relevant articles
    matched_articles = search_relevant_articles(data.scenario, top_k=5)
    prompt = build_prompt(data.scenario, matched_articles)
    
    try:
        response = gemini.generate_content(prompt)
        analysis_text = getattr(response, "text", str(response))
        analysis_text = normalize_analysis_text(analysis_text)
    except Exception as e:
        print("Generative API error:", e)
        raise HTTPException(status_code=500, detail=str(e))
    
    # Parse the response into structured format
    violations = []
    has_violation = "Violation Status:" in analysis_text and "Yes" in analysis_text.split("Violation Status:")[1].split("\n")[0]
    
    if has_violation:
        # Extract violated articles
        articles_section = ""
        if "Violated Article(s):" in analysis_text:
            articles_section = analysis_text.split("Violated Article(s):")[1].split("Explanation:")[0].strip()
        
        # Extract explanation
        explanation = ""
        if "Explanation:" in analysis_text:
            explanation = analysis_text.split("Explanation:")[1].split("What the person can do next:")[0].strip()
        
        # Extract guidance
        guidance = ""
        if "What the person can do next:" in analysis_text:
            guidance = analysis_text.split("What the person can do next:")[1].strip()
        
        violations.append({
            "status": "Violation Detected",
            "article": articles_section,
            "explanation": explanation,
            "guidance": guidance,
            "confidence": 0.95
        })
    else:
        violations.append({
            "status": "No Violation",
            "article": "N/A",
            "explanation": "No fundamental rights violation detected in this scenario.",
            "guidance": "No immediate action required.",
            "confidence": 0.95
        })
    
    return {
        "violations": violations,
        "summary": {
            "total_violations": len([v for v in violations if v["status"] == "Violation Detected"]),
            "risk_level": "High" if has_violation else "Low",
            "recommendations": [
                "Consult with a legal advisor for detailed guidance" if has_violation else "Continue monitoring the situation"
            ]
        },
        "raw_analysis": analysis_text
    }
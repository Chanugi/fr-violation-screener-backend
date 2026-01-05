# FR Violation Screener Backend

AI-powered Fundamental Rights Violation Screener for Sri Lanka.

## 📋 Overview

Analyzes real-life incidents to detect violations of Fundamental Rights under Sri Lankan Constitution (Articles 10–17). Returns violation status, relevant articles, plain language explanations, related Supreme Court cases, and legal guidance.

## 🛠 Tech Stack

- **Framework**: FastAPI
- **AI**: Google Gemini API
- **Embeddings**: Sentence-Transformers
- **Search**: FAISS
- **Server**: Uvicorn

## 📁 Project Structure

```
fr-violation-screener-backend/
├── main.py                        # API endpoints & core logic
├── requirements.txt               # Python dependencies
├── .env                          # Environment variables (not in git)
├── data/
│   ├── constitution_articles.json
│   └── supreme_court_cases.json
└── README.md
```

## 🚀 Setup

### 1. Create Virtual Environment

```bash
python -m venv venv

# Activate
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment

Create `.env` file:

```
GOOGLE_API_KEY=your-api-key-here
```

Get API key: [Google AI Studio](https://makersuite.google.com/app/apikey)

## ▶️ Running

### Development Mode

```bash
python -m uvicorn main:app --host 0.0.0.0 --port 8016 --reload
```

### Access

- API: `http://localhost:8016`
- Docs: `http://localhost:8016/docs`
- Health: `http://localhost:8016/health`

## 📡 API Endpoints

### POST `/screen-scenario`

Analyze a scenario for FR violations.

**Request**:
```json
{
  "scenario": "I was arrested without a warrant and not allowed to contact a lawyer"
}
```

**Response**:
```json
{
  "violations": [
    {
      "article": "Article 13(1)",
      "title": "Freedom of the Person",
      "status": "Violation Detected",
      "explanation": "No person can be arrested without a warrant or reasonable suspicion",
      "guidance": "File habeas corpus petition in Court of Appeal",
      "confidence": 0.95,
      "related_cases": [
        {
          "case_name": "Case Name vs State",
          "year": 2020,
          "summary": "Court held that arrest without warrant is illegal..."
        }
      ]
    }
  ],
  "summary": {
    "total_violations": 1,
    "risk_level": "High",
    "recommendations": ["File habeas corpus petition", "Contact legal aid"]
  }
}
```

### POST `/screen-document`

Upload PDF and screen for FR violations.

### GET `/health`

Check server status.

## 🔄 Workflow

```
User Scenario
    ↓
Extract Text & Create Embedding
    ↓
Search Relevant Articles (FAISS)
    ↓
Search Similar Past Cases
    ↓
Call Google Gemini API
    ↓
Parse & Structure Response
    ↓
Return JSON with Violations + Cases
```

## 🐛 Troubleshooting

### Port Already in Use
```bash
# macOS/Linux: lsof -i :8016 | kill -9 <PID>
# Windows: netstat -ano | findstr :8016 | taskkill /PID <PID> /F
```

### API Key Error
- Check `.env` file exists in root directory
- Verify API key is correct
- Restart server after changing `.env`

### Module Not Found
```bash
pip install --upgrade -r requirements.txt
```

## 📚 Frontend Integration

Frontend connects via environment variables:

```
VITE_FR_API_URL=http://localhost:8016
```

## 🔐 Security

- ⚠️ Never commit `.env` file
- API key in environment variables only
- CORS configured for allowed domains

---

**Status**: ✅ Ready  
**Version**: 1.0.0


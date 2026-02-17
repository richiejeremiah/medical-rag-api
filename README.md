# Medical RAG API

Flask API for retrieving medical codes (ICD-10, CPT, HCPCS) using Pinecone vector search.

## Quick Deploy to Render

### 1. Push to GitHub
```bash
git add .
git commit -m "Add Flask RAG API"
git push origin main
```

### 2. Deploy on Render
1. Go to https://render.com
2. Sign up/login with GitHub
3. Click **New +** → **Web Service**
4. Connect repository: `richiejeremiah/medical-rag-api`
5. Settings:
   - **Name:** `medical-rag-api`
   - **Environment:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app`
   - **Instance Type:** `Free` (or `Starter` for better performance)

### 3. Environment Variables
In Render dashboard → **Environment** tab, add:
```
PINECONE_API_KEY=your_pinecone_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
INDEX_NAME=doctorlittle
```

**⚠️ Important:** Never commit API keys to Git. Set them in Render dashboard only.

### 4. Deploy
Click **Create Web Service** and wait 5-10 minutes.

You'll get a URL like: `https://medical-rag-api.onrender.com`

## Test

```bash
# Health check
curl https://medical-rag-api.onrender.com/health

# Test query
curl -X POST https://medical-rag-api.onrender.com/api/retrieve \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "patient with chest pain and shortness of breath",
    "specialty": "cardiology",
    "top_k": 10
  }'
```

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export PINECONE_API_KEY=your_key
export OPENAI_API_KEY=your_key
export INDEX_NAME=doctorlittle

# Run locally
python app.py
# Or with gunicorn
gunicorn app:app
```

## API Endpoints

### GET /health
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "index": "doctorlittle",
  "total_vectors": 69195,
  "dimension": 1536,
  "terminology_loaded": 8746
}
```

### POST /api/retrieve
Retrieve medical codes from RAG.

**Request:**
```json
{
  "query": "patient with chest pain",
  "specialty": "cardiology",
  "region": "US",
  "exclusion_terms": [],
  "top_k": 20
}
```

**Response:**
```json
{
  "icd10": [
    {
      "code": "R06.02",
      "description": "Shortness of breath",
      "score": 0.89
    }
  ],
  "cpt": [
    {
      "code": "99213",
      "description": "Office visit",
      "score": 0.85
    }
  ],
  "hcpcs": [],
  "metadata": {
    "query": "patient with chest pain",
    "specialty": "cardiology",
    "region": "US",
    "total_results": 45,
    "filtered_results": 12,
    "source": "render_rag_v1"
  }
}
```

## Update Middleware

Once deployed, update `middleware-platform/.env`:

```bash
RAG_API_URL=https://medical-rag-api.onrender.com/api/rag
```

Or if using the proxy (recommended):
```bash
RAG_API_URL=http://localhost:4000/api/rag
```

The middleware proxy will forward to Render automatically.

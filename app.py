from flask import Flask, request, jsonify
from pinecone import Pinecone
from openai import OpenAI
import json
import os
import re

app = Flask(__name__)

# Environment variables (set in Render dashboard)
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
INDEX_NAME = os.environ.get('INDEX_NAME', 'doctorlittle')

if not PINECONE_API_KEY or not OPENAI_API_KEY:
    raise ValueError("PINECONE_API_KEY and OPENAI_API_KEY must be set")

# Initialize clients
pc = Pinecone(api_key=PINECONE_API_KEY)
pinecone_index = pc.Index(INDEX_NAME)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Load terminology from file
print("Loading terminology...")
terminology = {}
try:
    with open('terminology_lookup.json', 'r') as f:
        terminology_data = json.load(f)
    
    if isinstance(terminology_data, list):
        for entry in terminology_data:
            code = entry.get('code')
            if code:
                terminology[code] = entry
    else:
        terminology = terminology_data
    
    print(f"✅ Loaded {len(terminology)} terminology entries")
except FileNotFoundError:
    print("⚠️  terminology_lookup.json not found - continuing without terminology")
except Exception as e:
    print(f"⚠️  Error loading terminology: {e}")

APP_VERSION = "1.0.0"

@app.route('/', methods=['GET'])
def index():
    """Root: list endpoints (confirms this app is deployed)."""
    return jsonify({
        "service": "medical-rag-api",
        "version": APP_VERSION,
        "endpoints": {
            "GET /health": "Health check (Pinecone + terminology)",
            "GET /api/debug_metadata": "Inspect raw Pinecone metadata (?query=...)",
            "POST /api/retrieve": "Retrieve ICD-10/CPT/HCPCS (body: query, specialty?, top_k?)",
            "POST /retrieve": "Same as /api/retrieve (alias)",
        },
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check"""
    try:
        stats = pinecone_index.describe_index_stats()
        return jsonify({
            "status": "healthy",
            "version": APP_VERSION,
            "index": INDEX_NAME,
            "total_vectors": stats.total_vector_count,
            "dimension": stats.dimension,
            "terminology_loaded": len(terminology)
        })
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e)}), 503

@app.route('/api/retrieve', methods=['POST'])
@app.route('/retrieve', methods=['POST'])
def retrieve_codes():
    """
    Retrieve medical codes from RAG
    
    Body: {query, specialty?, region?, exclusion_terms?, top_k?}
    Returns: {icd10: [...], cpt: [...], hcpcs: [...], metadata: {...}}
    """
    try:
        data = request.json or {}
        query = data.get('query', '').strip()
        specialty = data.get('specialty', 'general')
        region = data.get('region', 'US')
        exclusion_terms = data.get('exclusion_terms', [])
        top_k = int(data.get('top_k', 20))

        if not query:
            return jsonify({"error": "Query is required"}), 400

        # Generate embedding
        response = openai_client.embeddings.create(
            input=query,
            model="text-embedding-3-small"
        )
        query_embedding = response.data[0].embedding

        # Build filter
        filter_dict = {}
        if specialty and specialty != 'general':
            filter_dict['specialty'] = specialty

        # Query Pinecone
        results = pinecone_index.query(
            vector=query_embedding,
            top_k=min(top_k * 3, 100),
            include_metadata=True,
            filter=filter_dict if filter_dict else None
        )

        # Extract codes - try multiple strategies (metadata keys + regex from text)
        icd10_codes = {}
        cpt_codes = {}
        hcpcs_codes = {}

        for match in results.matches:
            metadata = match.metadata or {}
            chunk_text = metadata.get('text', '')
            score = match.score

            # Check exclusions
            if exclusion_terms and any(term.lower() in chunk_text.lower() for term in exclusion_terms):
                continue

            # STRATEGY 1: Look for codes in metadata (all possible key names)
            icd10_str = (
                metadata.get('icd10_codes') or
                metadata.get('icd10') or
                metadata.get('icd_10') or
                metadata.get('icd-10') or
                ''
            )
            cpt_str = (
                metadata.get('cpt_codes') or
                metadata.get('cpt') or
                metadata.get('procedure_codes') or
                ''
            )
            hcpcs_str = (
                metadata.get('hcpcs_codes') or
                metadata.get('hcpcs') or
                ''
            )
            from_meta_icd10 = bool(icd10_str)
            from_meta_cpt = bool(cpt_str)
            from_meta_hcpcs = bool(hcpcs_str)

            # STRATEGY 2: If no codes in metadata, extract from text
            if not icd10_str and chunk_text:
                icd10_matches = re.findall(r'\b[A-TV-Z]\d{2}(?:\.\d{1,4})?\b', chunk_text)
                if icd10_matches:
                    icd10_str = ','.join(icd10_matches)
            if not cpt_str and chunk_text:
                cpt_matches = re.findall(r'\b\d{5}\b', chunk_text)
                if cpt_matches:
                    cpt_str = ','.join(cpt_matches)

            # Process ICD-10
            if icd10_str:
                for code in str(icd10_str).replace(';', ',').split(','):
                    code = code.strip()
                    if code and code not in icd10_codes and len(code) >= 3:
                        if re.match(r'^[A-TV-Z]\d{2}', code):
                            description = f"ICD-10 code {code}"
                            if code in terminology:
                                terms = terminology[code].get('positive_terms', [])
                                if terms:
                                    description = terms[0]
                            icd10_codes[code] = {
                                'code': code,
                                'description': description,
                                'score': float(score),
                                'source': 'metadata' if from_meta_icd10 else 'text_extraction'
                            }

            # Process CPT
            if cpt_str:
                for code in str(cpt_str).replace(';', ',').split(','):
                    code = code.strip()
                    if code and code not in cpt_codes and len(code) == 5 and code.isdigit():
                        description = f"CPT code {code}"
                        if code in terminology:
                            terms = terminology[code].get('positive_terms', [])
                            if terms:
                                description = terms[0]
                        cpt_codes[code] = {
                            'code': code,
                            'description': description,
                            'score': float(score),
                            'source': 'metadata' if from_meta_cpt else 'text_extraction'
                        }

            # Process HCPCS
            if hcpcs_str:
                for code in str(hcpcs_str).replace(';', ',').split(','):
                    code = code.strip()
                    if code and code not in hcpcs_codes:
                        description = f"HCPCS code {code}"
                        if code in terminology:
                            terms = terminology[code].get('positive_terms', [])
                            if terms:
                                description = terms[0]
                        hcpcs_codes[code] = {
                            'code': code,
                            'description': description,
                            'score': float(score),
                            'source': 'metadata' if from_meta_hcpcs else 'text_extraction'
                        }

        # Sort and limit
        icd10_list = sorted(icd10_codes.values(), key=lambda x: x['score'], reverse=True)[:20]
        cpt_list = sorted(cpt_codes.values(), key=lambda x: x['score'], reverse=True)[:15]
        hcpcs_list = sorted(hcpcs_codes.values(), key=lambda x: x['score'], reverse=True)[:10]

        return jsonify({
            'icd10': icd10_list,
            'cpt': cpt_list,
            'hcpcs': hcpcs_list,
            'metadata': {
                'query': query,
                'specialty': specialty,
                'region': region,
                'total_results': len(results.matches),
                'filtered_results': len(icd10_codes) + len(cpt_codes) + len(hcpcs_codes),
                'source': 'render_rag_v1',
                'extraction_methods_used': {
                    'metadata_fields': True,
                    'text_extraction': True
                }
            }
        })

    except Exception as e:
        print(f"❌ Error in retrieve_codes: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/debug_metadata', methods=['GET'])
def debug_metadata():
    """
    Inspect raw Pinecone metadata for the first few matches.
    Query param: query (e.g. ?query=anxiety). Returns metadata keys and sample values.
    """
    try:
        query = (request.args.get('query') or 'anxiety').strip()
        response = openai_client.embeddings.create(
            input=query,
            model="text-embedding-3-small"
        )
        query_embedding = response.data[0].embedding
        results = pinecone_index.query(
            vector=query_embedding,
            top_k=5,
            include_metadata=True
        )
        out = {
            "query": query,
            "num_matches": len(results.matches),
            "match_metadata_samples": []
        }
        for i, match in enumerate(results.matches):
            meta = match.metadata or {}
            out["match_metadata_samples"].append({
                "match_index": i,
                "score": getattr(match, 'score', None),
                "metadata_keys": list(meta.keys()),
                "metadata": {k: (v[:200] + "..." if isinstance(v, str) and len(v) > 200 else v) for k, v in meta.items()}
            })
        return jsonify(out)
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

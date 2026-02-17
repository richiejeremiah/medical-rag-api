from flask import Flask, request, jsonify
from pinecone import Pinecone
from openai import OpenAI
import json
import os

app = Flask(__name__)

# Environment variables (set in Render dashboard)
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
INDEX_NAME = os.environ.get('INDEX_NAME', 'doctorlittle')

if not PINECONE_API_KEY or not OPENAI_API_KEY:
    raise ValueError("PINECONE_API_KEY and OPENAI_API_KEY must be set")

# Initialize clients
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)
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

@app.route('/health', methods=['GET'])
def health():
    """Health check"""
    try:
        stats = index.describe_index_stats()
        return jsonify({
            "status": "healthy",
            "index": INDEX_NAME,
            "total_vectors": stats.total_vector_count,
            "dimension": stats.dimension,
            "terminology_loaded": len(terminology)
        })
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e)}), 503

@app.route('/api/retrieve', methods=['POST'])
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
        results = index.query(
            vector=query_embedding,
            top_k=min(top_k * 3, 100),  # Cap at 100 for performance
            include_metadata=True,
            filter=filter_dict if filter_dict else None
        )

        # Extract codes
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

            # Extract ICD-10
            icd10_str = metadata.get('icd10_codes', '') or metadata.get('icd10', '')
            if icd10_str:
                for code in str(icd10_str).split(','):
                    code = code.strip()
                    if code and code not in icd10_codes:
                        description = f"ICD-10 code {code}"
                        if code in terminology:
                            terms = terminology[code].get('positive_terms', [])
                            if terms:
                                description = terms[0]
                        
                        icd10_codes[code] = {
                            'code': code,
                            'description': description,
                            'score': float(score)
                        }

            # Extract CPT
            cpt_str = metadata.get('cpt_codes', '') or metadata.get('cpt', '')
            if cpt_str:
                for code in str(cpt_str).split(','):
                    code = code.strip()
                    if code and code not in cpt_codes:
                        description = f"CPT code {code}"
                        if code in terminology:
                            terms = terminology[code].get('positive_terms', [])
                            if terms:
                                description = terms[0]
                        
                        cpt_codes[code] = {
                            'code': code,
                            'description': description,
                            'score': float(score)
                        }

            # Extract HCPCS
            hcpcs_str = metadata.get('hcpcs_codes', '') or metadata.get('hcpcs', '')
            if hcpcs_str:
                for code in str(hcpcs_str).split(','):
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
                            'score': float(score)
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
                'source': 'render_rag_v1'
            }
        })

    except Exception as e:
        print(f"❌ Error in retrieve_codes: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

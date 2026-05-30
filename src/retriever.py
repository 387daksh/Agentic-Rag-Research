import json
import os
import chromadb
from sentence_transformers import SentenceTransformer,CrossEncoder
from rank_bm25 import BM25Okapi
from tqdm import tqdm
from groq import Groq
from nltk.stem import PorterStemmer
import pickle

CHUNKS_FILE="data/chunks.json"
CHROMA_DIR="data/chroma"
BM25_FILE = "data/BM25"
print("loading embedding model")
EMBEDDER=SentenceTransformer("BAAI/bge-large-en-v1.5",device="cuda")
print("loading reranker")
RERANKER = CrossEncoder("BAAI/bge-reranker-base")
STEMMER = PorterStemmer()
# GROQ_CLIENT=Groq(api_key=os.environ.get("GROQ_API_KEY"))
GROQ_CLIENT=Groq(api_key="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")


def setup_chromadb(chunks):
    client=chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        collection = client.get_collection("papers")
        print("Using existing Chroma collection")
        return collection
    except:
        print("Creating new Chroma collection")
    collection=client.create_collection("papers")
    batch_size=100
    for i in tqdm(range(0,len(chunks),batch_size)):
        batch=chunks[i:i+batch_size]
        texts=[c["text"] for c in batch]
        ids=[c["chunk_id"] for c in batch]
        metadatas=[{"arxiv_id": c["arxiv_id"], "title": c["title"]} for c in batch]
        
        embeddings=EMBEDDER.encode(texts,normalize_embeddings=True).tolist()
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )
    return collection
def setup_bm25(chunks):
    tokenized_chunks=[[STEMMER.stem(word) for word in c["text"].lower().split()] for c in chunks]
    bm25= BM25Okapi(tokenized_chunks)
    return bm25

def search_chromadb(collection,query,n_results=10):
    query_embedding=EMBEDDER.encode(query,normalize_embeddings=True).tolist()
    results=collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    # print(results.keys())
    chunks=[]
    for i in range(len(results["ids"][0])):
        chunks.append({
            "chunk_id":results["ids"][0][i],
            "arxiv_id":results["metadatas"][0][i]["arxiv_id"],
            "title":results["metadatas"][0][i]["title"],
            "text":results["documents"][0][i],
            "chroma_distance": results["distances"][0][i]
        })
    return chunks
def search_bm25(bm25,query,chunks,n_results=10):
    tokenized_query=[STEMMER.stem(word) for word in query.lower().split()]
    scores=bm25.get_scores(tokenized_query)
    
    top_indices=sorted(range(len(scores)),key=lambda i:scores[i], reverse=True)[:n_results]
    results=[]
    for idx in top_indices:
        results.append({
            "chunk_id":chunks[idx]["chunk_id"],
            "arxiv_id": chunks[idx]["arxiv_id"],
            "title": chunks[idx]["title"],
            "text": chunks[idx]["text"],
            "bm25_score": scores[idx]
        })
    return results

def reciprocal_rank_fusion(chromadb_results,bm25_results,k=60):
    print("\nCHROMA")
    for r in chromadb_results:
        print(r["title"])

    print("\nBM25")
    for r in bm25_results:
        print(r["title"])
    scores={}
    for rank,chunk in enumerate(chromadb_results):
        chunk_id=chunk["chunk_id"]
        if chunk_id not in scores:
            scores[chunk_id]={"rrf_score":0,"chunk":chunk.copy()}
        else:
            scores[chunk_id]["chunk"].update(chunk.copy())

        scores[chunk_id]["rrf_score"]+=1/(k+rank+1)
    for rank, chunk in enumerate(bm25_results):
        chunk_id = chunk["chunk_id"]
        if chunk_id not in scores:
            scores[chunk_id] = {"rrf_score": 0, "chunk": chunk}
        else:
            scores[chunk_id]["chunk"].update(chunk)

        scores[chunk_id]["rrf_score"] += 1 / (k + rank + 1)
    sorted_chunks=sorted(scores.values(),key= lambda x: x["rrf_score"],reverse=True)
    results=[]
    for item in sorted_chunks:
        chunk=item["chunk"].copy()
        chunk["rrf_score"]=item["rrf_score"]
        results.append(chunk)
    return results

def hyde(query):
    prompt = f"""
Expand this query with related concepts,
methods and terminology likely to appear
in relevant papers.

Query:
{query}

Do not invent architectures,
datasets,
benchmarks,
or method names.

Output one paragraph.
"""
    response=GROQ_CLIENT.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=300
    )
    hypothetical_answer = response.choices[0].message.content
    return f"{query} {hypothetical_answer}"

def retrieve(collections,bm25,chunks,query,n_results=5,use_hyde=True):
    if use_hyde:
        expanded_query=hyde(query)
    else:
        expanded_query=query
    # print(f"HyDE Query: {expanded_query}")
    chroma_results=search_chromadb(collections,expanded_query,n_results=10)
    bm25_results = search_bm25(bm25,query,chunks,n_results=10)
    combined = reciprocal_rank_fusion(chroma_results, bm25_results)
    seen=set()
    deduped=[]
    for chunk in combined:
        if chunk["arxiv_id"] not in seen:
            seen.add(chunk["arxiv_id"])
            deduped.append(chunk)
    # print("before reranking")
    # for r in combined[:20]:
    #     print(r["title"])
    reranked=rerank(query,deduped[:20])
    # print("after reranking")
    # for r in reranked[:20]:
    #     print(r["title"])
    
    return reranked[:n_results]

def rerank(query,chunks):
    pairs=[[query,
            f"Title: {c['title']}\n\n{c['text']}"] for c in chunks]
    scores=RERANKER.predict(pairs)
    ranked=sorted(
        zip(scores,chunks),
        key=lambda x: x[0],
        reverse=True
    )
    results=[]
    for score,chunk in ranked:
        chunk=chunk.copy()
        chunk["rerank_score"]=float(score)
        results.append(chunk)
    return results

def build_index():
    # print(EMBEDDER.device)
    print("loading chunks")
    with open(CHUNKS_FILE) as f:
        chunks=json.load(f)
    print(f"loaded {len(chunks)} chunks")
    print("building chromadb index")
    collection=setup_chromadb(chunks)
    print("chromadb done")
    if os.path.exists(BM25_FILE):
        print("loading bm25 index")
        with open(BM25_FILE, "rb") as f:
            bm25 = pickle.load(f)
        print("bm25 loaded")
    else:
        print("building bm25 index")
        bm25 = setup_bm25(chunks)
        with open(BM25_FILE, "wb") as f:
            pickle.dump(bm25, f)
        print("bm25 saved")
    
    return collection,bm25,chunks

if __name__=="__main__":
    collection,bm25,chunks=build_index()
    test_query="how do LLM agents plan tasks"
    print(f"\nTest query: {test_query}")
    results = retrieve(collection, bm25, chunks, test_query)
    for i,r in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        print(f"Paper: {r['title']}")
        print(f"ArXiv: {r['arxiv_id']}")
        print(f"Text: {r['text'][:200]}...")
        print(f"Chroma Distance: {r.get('chroma_distance', 'N/A')}")
        print(f"BM25 Score: {r.get('bm25_score', 'N/A')}") 
        print(f"RRF Score: {r.get('rrf_score', 'N/A')}")
        print(f"Cross Encoder Score: {r.get('rerank_score', 'N/A')}")

        
import json
from groq import Groq
import json
from retriever import build_index,retrieve
GROQ_CLIENT=Groq(api_key="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")
MODEL="llama-3.3-70b-versatile"
collection, bm25, chunks_data,G = build_index()
def run_baseline(question):
    print(f"\n{'='*50}")
    print(f"Question: {question}")
    print(f"{'='*50}")
    print("\n[RETRIEVER] Single retrieval...")
    chunks = retrieve(collection, bm25, chunks_data, question, n_results=5)
    evidence = ""
    for i, chunk in enumerate(chunks):
        evidence += f"\n[{i+1}] From '{chunk['title']}' ({chunk['arxiv_id']}):\n{chunk['text'][:400]}\n"
    print("[SYNTHESIZER] Single LLM call...")
    prompt = f"""Answer this research question using only the provided evidence.
Cite sources inline using [arxiv_id] format.

Question: {question}

Evidence:
{evidence}

Answer:"""
    response = GROQ_CLIENT.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=800
    )
    answer = response.choices[0].message.content.strip()
    import re
    cited_ids = re.findall(r'\[(\d{4}\.\d{4,5}(?:v\d+)?)\]', answer)
    
    return {
        "question": question,
        "answer": answer,
        "cited_arxiv_ids": cited_ids,
        "tool_call_count": 1,
        "chunks_used": len(chunks)
    }
if __name__ == "__main__":
    test_question = "How do LLM agents plan and decompose complex tasks?"
    result = run_baseline(test_question)
    
    print("\n" + "="*50)
    print("BASELINE ANSWER:")
    print("="*50)
    print(result["answer"])
    print(f"\nCited papers: {result['cited_arxiv_ids']}")
    print(f"Tool calls: {result['tool_call_count']}")
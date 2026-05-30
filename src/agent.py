import json
import os
import time
from groq import Groq
from retriever import build_index,retrieve

GROQ_CLIENT = Groq(api_key="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")
MODEL = "llama-3.3-70b-versatile"

collection,bm25,chunks_data=build_index()

def plan(question):
    prompt = f"""You are a research planning assistant. Break down this research question into 3-4 specific sub-questions that together would fully answer it.

Return ONLY a JSON array of strings. No explanation, no markdown, just the JSON.

Example output:
["sub-question 1", "sub-question 2", "sub-question 3"]

Question: {question}"""
    response=GROQ_CLIENT.chat.completions.create(
        model=MODEL,
        messages=[{"role":"user","content":prompt}],
        response_format={"type":"json_object"},
        max_tokens=300
    )
    text=response.choices[0].message.content.strip()
    # print(text)
    try:
        data=json.loads(text)
        sub_questions = data["sub-questions"]
    except Exception as e:
        print(f"Planner error: {e}")
        sub_questions=[question]
    return sub_questions

def reflect(question,retrieved_chunks,round_num):
    evidence=""
    for i, chunk in enumerate(retrieved_chunks):
        evidence+=f"\n[{i+1}] from '{chunk['title']}' ({chunk['arxiv_id']}):\n{chunk['text'][:300]}\n"
    prompt = f"""You are a research quality checker. Given a question and retrieved evidence, decide if the evidence is sufficient to write a comprehensive answer.

Question: {question}

Retrieved Evidence:
{evidence}

Round: {round_num}/3

Respond ONLY with a JSON object in this exact format:
{{"sufficient": true/false, "reason": "one sentence explanation", "new_queries": ["query1", "query2"]}}

If sufficient is true, new_queries can be empty.
If sufficient is false, provide 2 new search queries to find missing evidence."""
    response = GROQ_CLIENT.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        response_format={"type":"json_object"}
    )    
    text = response.choices[0].message.content.strip()
    try:
        results=json.loads(text)
        print(results)
        print(type(results))
    except Exception as e:
        print(f"reflector error {e}")
        results={"sufficient": True, "reason": "parse error", "new_queries": []}
    return results

def synthesize(question,all_chunks):
    evidence=""
    for i,chunk in enumerate(all_chunks):
        evidence+=f"\n[{i+1}] From '{chunk['title']}' ({chunk['arxiv_id']}):\n{chunk['text'][:400]}\n"
    prompt = f"""You are a research synthesis assistant. Write a comprehensive answer to the question using ONLY the provided evidence.

Rules:
1. Every claim must be supported by evidence
2. Cite sources inline using the format [arxiv_id] e.g. [2401.12345]
3. Do not use any knowledge outside the provided evidence
4. Be specific and technical
5. If evidence is contradictory, acknowledge it

Question: {question}

Evidence:
{evidence}

Write the answer now:"""
    response=GROQ_CLIENT.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=800
    )
    return response.choices[0].message.content.strip()

def verify_citations(answer,all_chunks):
    chunk_lookup={}
    for chunk in all_chunks:
        arxiv_id=chunk["arxiv_id"]
        if arxiv_id not in chunk_lookup:
            chunk_lookup[arxiv_id]=chunk["text"]
    import re
    cited_ids = re.findall(r'\[(\d{4}\.\d{4,5}(?:v\d+)?)\]', answer)
    if not cited_ids:
        return answer,[]
    verified = []
    unverified = []
    
    for arxiv_id in cited_ids:
        if arxiv_id in chunk_lookup:
            verified.append(arxiv_id)
        else:
            unverified.append(arxiv_id)
    for uid in unverified:
        answer = answer.replace(f"[{uid}]", "")
    return answer, verified

def run_agent(question):
    print(f"\n{'='*50}")
    print(f"Question: {question}")
    print(f"\n{'='*50}")
    all_chunks = []
    tool_call_count = 0
    print("\n[PLANNER] breaking question into sub-questions")
    sub_questions = plan(question)
    print(f"sub-questions: {sub_questions}")
    print("\n[RETRIEVER] Retrieving evidence...")
    for sub_q in sub_questions:
        chunks = retrieve(collection, bm25, chunks_data, sub_q, n_results=5)
        all_chunks.extend(chunks)
        tool_call_count += 1
    seen_ids = set()
    unique_chunks = []
    for chunk in all_chunks:
        if chunk["chunk_id"] not in seen_ids:
            seen_ids.add(chunk["chunk_id"])
            unique_chunks.append(chunk)
    all_chunks = unique_chunks
    print(f"retrieved {len(all_chunks)} unique chunks")
    for round_num in range(1, 4):
        print(f"\n[REFLECTOR] round {round_num}/3")
        reflection = reflect(question, all_chunks, round_num)
        print(f"Sufficient: {reflection['sufficient']}")
        print(f"Reason: {reflection['reason']}")
        if reflection["sufficient"]:
            print("Evidence sufficient, stopping search.")
            break
        if round_num == 3:
            print("Max rounds reached, proceeding to synthesis.")
            break
        print(f"Searching for more evidence")
        for new_query in reflection.get("new_queries", []):
            new_chunks = retrieve(collection, bm25, chunks_data, new_query, n_results=3)
            all_chunks.extend(new_chunks)
            tool_call_count += 1
        seen_ids = set()
        unique_chunks = []
        for chunk in all_chunks:
            if chunk["chunk_id"] not in seen_ids:
                seen_ids.add(chunk["chunk_id"])
                unique_chunks.append(chunk)
        all_chunks = unique_chunks
    print(f"\n[SYNTHESIZER] writing answer from {len(all_chunks)} chunks")
    answer = synthesize(question, all_chunks)
    print("\n[VERIFIER] checking citations") 
    verified_answer, verified_ids = verify_citations(answer, all_chunks)
    print(f"verified citations: {verified_ids}")
    return {
        "question": question,
        "answer": verified_answer,
        "cited_arxiv_ids": verified_ids,
        "tool_call_count": tool_call_count,
        "chunks_used": len(all_chunks)
    }
if __name__ == "__main__":
    test_question = "How do LLM agents plan and decompose complex tasks?"
    result = run_agent(test_question)
    
    print("\n" + "="*50)
    print("FINAL ANSWER:")
    print("="*50)
    print(result["answer"])
    print(f"\nCited papers: {result['cited_arxiv_ids']}")
    print(f"Tool calls: {result['tool_call_count']}")


# reflect("ai agents workflows",retrieved_chunks=[],round_num=1) 
# if __name__=="__main__":
    # sub_questions=plan("how to handle agentic workflows efficiently")
    # for question in sub_questions:
    #     print(question)
# sub_questions = plan("how to handle agentic workflows efficiently")
# print(type(sub_questions))
# print(sub_questions)
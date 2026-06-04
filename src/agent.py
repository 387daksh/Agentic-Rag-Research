import json
import os
import re
import time
from groq import Groq
import google.generativeai as genai
from mistralai.client import Mistral
from retriever import build_index, retrieve

GROQ_CLIENT = Groq(api_key="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")
genai.configure(api_key="AQ.Ab8RN6JIPO2THkMXTYRMKrukMInFyOZKhTYj3MHEC94KYeDYBQ")
GEMINI_CLIENT = genai.GenerativeModel("gemma-4-31b-it")
MISTRAL_CLIENT = Mistral( api_key="aHeEzsRfNADJpzgOKhPFTs4z7mO3V8IJ")

GROQ_MODEL = "llama-3.1-8b-instant"
MISTRAL_MODEL = "ministral-14b-2512"
collection, bm25, chunks_data,G = build_index()

def llm_call(prompt, backend="mistral", max_tokens=800, json_mode=False):
    time.sleep(4.2)
    if backend == "groq":
        kwargs = {
            "model": GROQ_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        response = GROQ_CLIENT.chat.completions.create(**kwargs)
        return response.choices[0].message.content.strip()

    elif backend == "gemini":
        if json_mode:
            full_prompt = prompt + "\n\nCRITICAL: Respond with ONLY a valid JSON object. No markdown backticks, no explanation, no thinking out loud. Start your response with { and end with }."
        else:
            full_prompt = prompt + "\n\nCRITICAL: Write the final answer directly. No bullet points showing your reasoning process, no self-correction notes, no word count checks. Just the answer."
        
        response = GEMINI_CLIENT.generate_content(full_prompt)
        text = response.text.strip()
        if json_mode:
            text = text.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()
            # Find JSON object boundaries
            start = text.find("{")
            end = text.rfind("}") + 1
            if start != -1 and end > start:
                text = text[start:end]
        
        return text
    elif backend == "mistral":

        messages = [
            {
            "role": "user",
            "content": prompt
            }
        ]

        response = MISTRAL_CLIENT.chat.complete(
            model=MISTRAL_MODEL,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.1
        )

        return response.choices[0].message.content.strip()
    else:
        raise ValueError(f"Unknown backend: {backend}")

def plan(question, backend="mistral"):
    prompt = f"""
You are a research planning assistant. Break down this research question into 3-4 specific sub-questions that together would fully answer it.
Generate 3-4 focused aspects of the original question.
Requirements:
- Each aspect should answer a different part of the question.
- Avoid benchmark or evaluation-focused queries unless the original question asks about evaluation.
- Optimize for retrieving research papers directly relevant to answering the question.
- Keep each query under 12 words.
Return ONLY a JSON object with key "sub-questions" containing an array of strings.

Example output:
{{"sub-questions": ["sub-question 1", "sub-question 2", "sub-question 3"]}}

Question: {question}"""

    text = llm_call(prompt, backend=backend, max_tokens=300, json_mode=True)
    print(text)
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*","",text)
        text = re.sub(r"\s*```$","",text)
    try:
        data = json.loads(text)
        sub_questions = data["sub-questions"]
    except Exception as e:
        print(f"Planner error: {e}")
        sub_questions = [question]

    return sub_questions


def reflect(question, retrieved_chunks, round_num, backend="mistral"):
    evidence = ""
    for i, chunk in enumerate(retrieved_chunks):
        evidence += f"\n[{i+1}] from '{chunk['title']}' ({chunk['arxiv_id']}):\n{chunk['text'][:300]}\n"

    prompt = f"""You are a research quality checker. Given a question and retrieved evidence, decide if the evidence is sufficient to write a comprehensive answer.

Question: {question}

Retrieved Evidence:
{evidence}

Round: {round_num}/3

Respond ONLY with a JSON object in this exact format:
{{"sufficient": true, "reason": "one sentence explanation", "new_queries": []}}
or
{{"sufficient": false, "reason": "one sentence explanation", "new_queries": ["query1", "query2"]}}

If sufficient is true, new_queries must be empty.
If sufficient is false, provide 2 new search queries to find missing evidence."""

    text = llm_call(prompt, backend=backend, max_tokens=200, json_mode=True)
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*","",text)
        text = re.sub(r"\s*```$","",text)
    try:
        result = json.loads(text)
    except Exception as e:
        print(f"Reflector error: {e}")
        result = {"sufficient": True, "reason": "parse error", "new_queries": []}

    return result


def synthesize(question, all_chunks, question_type="survey", backend="mistral"):
    if question_type == "factoid":
        length_guide = "1 to 3 sentences. Be concise and direct."
    elif question_type == "comparative":
        length_guide = "100 to 300 words. Compare at least 2 papers directly."
    else:
        length_guide = "250 to 600 words. Be comprehensive and well structured."

    evidence = ""
    for i, chunk in enumerate(all_chunks):
        evidence += f'''
        Paper: {chunk['title']}
        ArXiv: {chunk['arxiv_id']}
        
        {chunk['text']}'''

    prompt = f"""You are a research synthesis assistant. Write a comprehensive answer to the question using ONLY the provided evidence.

Rules:
1. Every claim must be supported by evidence
2. Cite sources inline using the format [arxiv_id] e.g. [2401.12345v1]
3. Do not use any knowledge outside the provided evidence
4. Length: {length_guide}
5. If evidence is contradictory, acknowledge it
6. Write the answer directly. No reasoning steps, no self-correction, no word count checks.
7.Use ONLY citations of the form: [2504.19413], NEVER use [1], [2], [3]. Always cite using the actual arXiv ID.

Question: {question}

Evidence:
{evidence}

Return ONLY the final answer.

Do NOT output:
- notes
- reasoning
- drafts
- self-corrections
- word-count checks
- planning steps
- bullet-point analysis

The response must begin directly with the answer.

Final Answer:"""
    if (question_type=="survey"): max_tokens=1200
    else: max_tokens=800
    return llm_call(prompt, backend=backend, max_tokens=max_tokens)


def verify_citations(answer, all_chunks):
    chunk_lookup = {}
    for chunk in all_chunks:
        arxiv_id=chunk["arxiv_id"]
        if arxiv_id not in chunk_lookup:
            chunk_lookup[arxiv_id] = chunk["text"]

    cited_ids = re.findall(r'\[(\d{4}\.\d{4,5}(?:v\d+)?)\]', answer)
    if not cited_ids:
        return answer, []

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

def run_agent(question, question_type="survey",use_planner=True, use_reflector=True,use_reranker=True, use_hyde=True,use_verifier=True, backend="mistral",use_hippo=True):

    print(f"\n{'='*50}")
    print(f"Question: {question}")
    print(f"Backend: {backend}")
    print(f"{'='*50}")

    all_chunks = []
    tool_call_count = 0

    if use_planner:
        print("\n[PLANNER] breaking into sub questions")
        sub_questions = plan(question, backend=backend)
        print(f"Sub-questions: {sub_questions}")
    else:
        print("\n[PLANNER] Skipped")
        sub_questions = [question]

    print("\n[RETRIEVER] Retrieving evidence")
    for sub_q in sub_questions:
        chunks = retrieve(collection, bm25, chunks_data, sub_q,G=G,n_results=5, use_hyde=use_hyde, use_reranker=use_reranker,use_hippo=use_hippo)
        all_chunks.extend(chunks)
        tool_call_count += 1
    seen_ids = set()
    unique_chunks = []
    for chunk in all_chunks:
        if chunk["chunk_id"] not in seen_ids:
            seen_ids.add(chunk["chunk_id"])
            unique_chunks.append(chunk)
    all_chunks = unique_chunks
    print(f"Retrieved {len(all_chunks)} unique chunks")

    if use_reflector:
        for round_num in range(1, 4):
            print(f"\n[REFLECTOR] Round {round_num}/3...")
            reflection = reflect(question, all_chunks, round_num, backend=backend)
            print(f"Sufficient: {reflection['sufficient']}")
            print(f"Reason: {reflection['reason']}")

            if reflection["sufficient"]:
                print("Evidence sufficient, stopping.")
                break

            if round_num == 3:
                print("Max rounds reached.")
                break

            for new_query in reflection.get("new_queries", []):
                new_chunks = retrieve(collection, bm25, chunks_data, new_query,G=G,n_results=3, use_hyde=use_hyde, use_reranker=use_reranker,use_hippo=use_hippo)
                all_chunks.extend(new_chunks)
                tool_call_count += 1

            # Deduplicate again
            seen_ids = set()
            unique_chunks = []
            for chunk in all_chunks:
                if chunk["chunk_id"] not in seen_ids:
                    seen_ids.add(chunk["chunk_id"])
                    unique_chunks.append(chunk)
            all_chunks = unique_chunks
    else:
        print("\n[REFLECTOR] skipped")

    # Step 4: Synthesize
    print(f"\n[SYNTHESIZER] writing answer from {len(all_chunks)} chunks")
    answer = synthesize(question, all_chunks, question_type, backend=backend)

    # Step 5: Verify citations
    if use_verifier:
        print("\n[VERIFIER] checking citations")
        answer, verified_ids = verify_citations(answer, all_chunks)
        print(f"verified citations: {verified_ids}")
    else:
        print("\n[VERIFIER] skipped")
        verified_ids = re.findall(r'\[(\d{4}\.\d{4,5}(?:v\d+)?)\]', answer)

    return {
        "question": question,
        "answer": answer,
        "cited_arxiv_ids": verified_ids,
        "tool_call_count": tool_call_count,
        "chunks_used": len(all_chunks)
    }


if __name__ == "__main__":
    test_question = "How do LLM agents plan and decompose complex tasks?"
    result = run_agent(test_question, backend="mistral")
    print("\n" + "="*50)
    print("FINAL ANSWER:")
    print("="*50)
    print(result["answer"])
    print(f"\nCited papers: {result['cited_arxiv_ids']}")
    print(f"Tool calls: {result['tool_call_count']}")
import json
import os
import time
from agent import run_agent
from baseline import run_baseline

QUESTIONS_FILE = "eval/questions.jsonl"
PREDICTIONS_DIR = "predictions"

def load_questions():
    questions=[]
    with open(QUESTIONS_FILE) as f:
        for line in f:
            line=line.strip()
            if line:
                questions.append(json.loads(line))
    return questions
# print(load_questions())

def run_config(config_name,run_fn,questions):
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    output_file = os.path.join(PREDICTIONS_DIR, f"{config_name}.jsonl")
    
    completed_ids=set()
    results=[]
    if os.path.exists(output_file):
        with open(output_file) as f:
            for line in f:
                line=line.strip()
                if line:
                    r=json.loads(line)
                    results.append(r)
                    completed_ids.add(r["id"])
        print(f"RESUMING {config_name}: {len(completed_ids)} already done")
    for q in questions:
        if q["id"] in completed_ids:
            print(f"skipping {q['id']} (already done)")
            continue
        print(f"{config_name} RUNNING: {q['id']} - {q['question'][:60]}...")
        start_time=time.time()
        try:
            result=run_fn(q)
        except Exception as e:
            print(f"error on {q['id']}: {e}")
            result = {
                "question":q['question'],
                "answer": "Error occurred during generation.",
                "cited_arxiv_ids": [],
                "tool_call_count": 0,
                "chunks_used": 0
            }
        latency=time.time()-start_time
        cited=list(set(result.get("cited_arxiv_ids",[])))
        predictions={
            "id":q["id"],
            "answer":result["answer"],
            "cited_papers":cited
        }
        results.append(predictions)
        with open(output_file,"w") as f:
            for r in results:
                f.write(json.dumps(r)+"\n")
        print(f"Done in {latency:.1f}s | Citations: {len(cited)}")
    print(f"\nFinished {config_name}: {len(results)} predictions saved to {output_file}")
    return results
def main():
    questions = load_questions()
    print(f"Loaded {len(questions)} questions")
    configs = [
    # ("full_agent",            lambda q: run_agent(q["question"], q.get("type", "survey"))),
    # ("baseline",              lambda q: run_baseline(q["question"])),
    # ("ablation_no_planner",   lambda q: run_agent(q["question"], q.get("type", "survey"), use_planner=False)),
    # ("ablation_no_reflector", lambda q: run_agent(q["question"], q.get("type", "survey"), use_reflector=False)),
    # ("ablation_no_reranker",  lambda q: run_agent(q["question"], q.get("type", "survey"), use_reranker=False)),
    # ("ablation_no_hyde",      lambda q: run_agent(q["question"], q.get("type", "survey"), use_hyde=False)),
    # ("ablation_no_verifier",  lambda q: run_agent(q["question"], q.get("type", "survey"), use_verifier=False)),
    ("ablation_hippo", lambda q: run_agent(q["question"], q.get("type","survey"), use_hippo=True)),
]
    for config_name,run_fn in configs:
        print(f"RUNNING CONFIG: {config_name}")
        run_config(config_name, run_fn, questions)
if __name__ == "__main__":
    main()
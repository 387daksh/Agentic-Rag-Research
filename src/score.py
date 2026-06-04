import json
import os
from mistralai.client import Mistral
import time

client = Mistral(api_key="aHeEzsRfNADJpzgOKhPFTs4z7mO3V8IJ")
model = "mistral-small-latest"
predictions_dir = "predictions"
questions_file = "eval/questions.jsonl"


def load_question_types():
    qtypes = {}
    with open(questions_file) as f:
        for line in f:
            line = line.strip()
            if line:
                q = json.loads(line)
                qtypes[q["id"]] = {
                    "type": q.get("type", "survey"),
                    "question": q.get("question", "")
                }
    return qtypes


def judge(question, answer, qtype):
    time.sleep(1)
    prompt = f"""you are an expert judge evaluating a research answer.

question: {question}
type: {qtype}
answer: {answer}

score on two dimensions, each 1-5:

accuracy: does the answer correctly address the question with specific technical details?
1=completely wrong, 3=partially correct, 5=fully correct and specific

faithfulness: are all claims grounded in cited evidence?
1=many unsupported claims, 3=mostly grounded, 5=every claim has a citation

respond only with json:
{{"accuracy": X, "faithfulness": X, "reason": "one sentence"}}"""

    r = client.chat.complete(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150,
        response_format={"type": "json_object"}
    )
    try:
        return json.loads(r.choices[0].message.content)
    except:
        return {"accuracy": 0, "faithfulness": 0, "reason": "parse error"}


def score_config(config_name, qtypes):
    path = os.path.join(predictions_dir, f"{config_name}.jsonl")
    if not os.path.exists(path):
        print(f"missing: {path}")
        return None

    preds = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                preds.append(json.loads(line))

    results = []
    acc, faith, cites, lat, tools = 0, 0, 0, 0, 0

    for pred in preds:
        qid = pred["id"]
        qinfo = qtypes.get(qid, {"type": "survey", "question": qid})
        
        scores = judge(
            qinfo["question"],
            pred["answer"],
            qinfo["type"]
        )

        n_cites = len(pred["cited_papers"])
        latency = pred.get("metadata", {}).get("latency_seconds", 0)
        tool_calls = pred.get("metadata", {}).get("tool_call_count", 0)

        acc += scores["accuracy"]
        faith += scores["faithfulness"]
        cites += n_cites
        lat += latency
        tools += tool_calls

        results.append({
            "id": qid,
            "type": qinfo["type"],
            "accuracy": scores["accuracy"],
            "faithfulness": scores["faithfulness"],
            "n_citations": n_cites,
            "latency": latency,
            "tool_calls": tool_calls,
            "reason": scores["reason"]
        })
        print(f"  {qid} | acc={scores['accuracy']} faith={scores['faithfulness']} | {scores['reason'][:60]}")

    n = len(preds)
    summary = {
        "config": config_name,
        "n": n,
        "accuracy": round(acc / n, 2),
        "faithfulness": round(faith / n, 2),
        "citations": round(cites / n, 2),
        "latency": round(lat / n, 2),
        "tools": round(tools / n, 2),
    }
    return summary, results


def main():
    qtypes = load_question_types()
    print(f"loaded {len(qtypes)} questions")

    configs = [
        "full_agent",
        "baseline",
        "ablation_no_planner",
        "ablation_no_reflector",
        "ablation_no_reranker",
        "ablation_no_hyde",
        "ablation_no_verifier",
        "ablation_no_hybrid",
    ]

    print(f"\n{'config':<25} {'accuracy':>10} {'faithful':>10} {'citations':>10} {'latency':>10} {'tools':>8}")

    summaries = []
    for config in configs:
        print(f"\nscoring {config}...")
        result = score_config(config, qtypes)
        if result is None:
            continue
        summary, _ = result
        summaries.append(summary)
        print(f"\n{summary['config']:<25} {summary['accuracy']:>10} {summary['faithfulness']:>10} {summary['citations']:>10} {summary['latency']:>10} {summary['tools']:>8}")

    with open("scores.json", "w") as f:
        json.dump(summaries, f, indent=2)
    print("saved to scores.json")


if __name__ == "__main__":
    main()
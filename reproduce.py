import sys
import os
import argparse
import json
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
sys.path.insert(0, src_dir)


def check_files():
    if not os.path.exists("data/chunks.json") or not os.path.exists("eval/questions.jsonl"):
        print("error: missing chunks.json or questions.jsonl files. check if you are in the root folder.")
        sys.exit(1)

def run_indexing():
    print("setting up the databases and index")
    try:
        from retriever import build_index
        collection, bm25, chunks, G = build_index()
        print("index set up finished")
        print(f"chroma initialized and graph has {len(G.nodes)} nodes and {len(G.edges)} edges")
    except Exception as e:
        print(f"failed to build the index: {e}")
        sys.exit(1)


def run_evaluation():
    print("running the agent evaluation pipeline")
    groq = os.environ.get("GROQ_API_KEY")
    mistral = os.environ.get("MISTRAL_API_KEY")
    if not groq or not mistral:
        print("you need to set groq_api_key and mistral_api_key to run the evaluation")
        sys.exit(1)
    try:
        import evaluation
        evaluation.main()
        print("evaluation done")
    except Exception as e:
        print(f"evaluation failed: {e}")
        sys.exit(1)

def run_scoring():
    print("scoring the predictions using mistral judge")
    try:
        import score
        score.main()
        print("scoring finished")
    except Exception as e:
        print(f"scoring failed: {e}")
        if not os.path.exists("scores.json"):
            print("could not find scores.json file")
            sys.exit(1)


def print_scores():
    print("here are the results from scores.json:")
    with open("scores.json") as f:
        data = json.load(f)
    for r in data:
        print(f"{r.get('config')} - accuracy: {r.get('accuracy')}, faithfulness: {r.get('faithfulness')}, citations: {r.get('citations')}, latency: {r.get('latency')}, tools: {r.get('tools')}")
    print("reproduction completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rerun-eval", action="store_true")
    args = parser.parse_args()
    check_files()
    run_indexing()
    if args.rerun_eval:
        run_evaluation()
    else:
        print("skipping full evaluation run, using predictions to score")
    run_scoring()
    print_scores()

import json
from tqdm import tqdm

CHUNKS_FILE = "data/chunks.json"
METADATA_FILE = "data/metadata.json"
OUTPUT_FILE = "data/chunks_with_abstracts.json"
def main():
    print("loading metadata")
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    print("loading chunks")
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    missing = 0

    print("adding abstracts")

    for chunk in tqdm(chunks):

        arxiv_id = chunk["arxiv_id"]

        paper = metadata.get(arxiv_id)

        if paper:
            chunk["abstract"] = paper.get(
                "abstract",
                ""
            )
        else:
            chunk["abstract"] = ""
            missing += 1

    print(
        f"missing metadata for {missing} chunks"
    )

    with open(OUTPUT_FILE,"w",encoding="utf-8") as f:
        json.dump(chunks,f,ensure_ascii=False,indent=2)

    print(
        f"Saved {len(chunks)} chunks to "
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()
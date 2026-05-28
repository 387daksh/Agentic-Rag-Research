import arxiv
import json
from tqdm import tqdm
import os
import time
from urllib.request import urlretrieve
import requests

PAPERS_DIR="data/papers"
METADATA_FILE="data/metadata.json"

QUERIES=[
    "LLM Agents",
    "agentic rag",
    "tool use language model",
    "agent memory",
    "agent benchmarks",
    "computer use agents",
]
start_date = "202401010000"
end_date = "202604302359"
MAX_QUERY_RETRIES = 5
RETRY_BASE_SECONDS = 15
RETRY_MAX_SECONDS = 120
QUERY_PAUSE_SECONDS = 15

def search_paper():
    papers={}
    client=arxiv.Client(page_size=100, delay_seconds=6.0, num_retries=6)
    
    for query in QUERIES:
        print(f"\nSearching:{query}")
        formatted_query = f"all:\"{query}\" AND submittedDate:[{start_date} TO {end_date}]"
        search=arxiv.Search(
            query=formatted_query,
            max_results=150,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending
        )
        # for paper in tqdm(client.results(search)):
        #     print(paper.entry_id)
        
        attempt = 0
        while True:
            try:
                for paper in tqdm(client.results(search), desc=f"fetching {query}"):
                    year=paper.published.year
                    month=paper.published.month
                    print(year,month)
                    
                    too_old=year<2024
                    too_new=year>2026 or (year==2026 and month>4)
                    if too_old or too_new:
                        continue
                    arxiv_id=paper.entry_id.split("/")[-1]
                    if arxiv_id in papers:
                        continue
                    papers[arxiv_id]={
                        "arxiv_id":arxiv_id,
                        "title":paper.title,
                        "abstract":paper.summary,
                        "pdf_url":paper.pdf_url,
                        "published":str(paper.published.date()),
                    }
                break
            except arxiv.HTTPError as err:
                status = getattr(err, "status", None) or getattr(err, "status_code", None)
                if status in (429, 503) and attempt < MAX_QUERY_RETRIES:
                    sleep_seconds = min(RETRY_MAX_SECONDS, RETRY_BASE_SECONDS * (2 ** attempt))
                    print(f"Rate limited ({status}). Retrying in {sleep_seconds}s...")
                    time.sleep(sleep_seconds)
                    attempt += 1
                    continue
                raise
        time.sleep(QUERY_PAUSE_SECONDS)
    # print(papers)
    return papers


# papers=search_paper()
# print(len(papers))
def download_pdfs(papers):
    os.makedirs(PAPERS_DIR,exist_ok=True)
    for arxiv_id,paper in tqdm(papers.items()):
        pdf_path=os.path.join(PAPERS_DIR,f"{arxiv_id}.pdf")
        if os.path.exists(pdf_path):
            continue
        try:
           pdf_url = f"https://arxiv.org/pdf/{arxiv_id.split('v')[0]}"
           response = requests.get(pdf_url, timeout=30)
           if response.status_code == 200:
                with open(pdf_path, "wb") as f:
                    f.write(response.content)
           else:
                print(f"Failed {arxiv_id}: status {response.status_code}")
        except Exception as e:
            print(f"Failed to download {arxiv_id}: {e}")
            continue
def main():
    print("Step 1: Searching arXiv...")
    papers = search_paper()
    print(f"Found {len(papers)} unique papers")
    
    print("Step 2: Saving metadata...")
    with open(METADATA_FILE, "w") as f:
        json.dump(papers, f, indent=2)
    print(f"Saved metadata to {METADATA_FILE}")
    
    print("Step 3: Downloading PDFs...")
    download_pdfs(papers)
    print("Done.")

if __name__ == "__main__":
    main()


import fitz
import json
from tqdm import tqdm
import os
PAPERS_DIR = "data/papers"
METADATA_FILE = "data/metadata.json"
CHUNKS_FILE = "data/chunks.json"

CHUNK_SIZE=400
CHUNK_OVERLAP=50

def extract_text(pdf_path):
    try:
        doc=fitz.open(pdf_path)
        text=""
        for page in doc:
            text+= page.get_text()  
        doc.close()
        return text
    except Exception as e:
        print(f"failed parsign {pdf_path} : {e}")
        return ""
def chunk_text(text,arxiv_id,title):
    words=text.split()
    chunks=[]
    start=0
    chunk_index=0
    while(start<len(words)):
        end=start+CHUNK_SIZE
        chunk_words=words[start:end]
        chunk_text=" ". join(chunk_words)
        chunks.append({
            "chunk_id":f"{arxiv_id}_{chunk_index}",
            "arxiv_id":arxiv_id,
            "title":title,
            "text":chunk_text,
        })
        chunk_index+=1
        start+=(CHUNK_SIZE-CHUNK_OVERLAP)
    return chunks 

def main():
    with open(METADATA_FILE) as f:
        papers=json.load(f)
    all_chunks=[]
    for arxiv_id,paper in tqdm(papers.items()):
        pdf_path=os.path.join(PAPERS_DIR,f"{arxiv_id}.pdf")
        if not os.path.exists(pdf_path):
            continue
        text=extract_text(pdf_path)
        if not text.strip():
            continue
        chunks=chunk_text(text,arxiv_id,paper["title"])
        all_chunks.extend(chunks)
    with open(CHUNKS_FILE,"w") as c:
        json.dump(all_chunks, c, indent=2)
    print(f"total chunks {len(all_chunks)}")
    
if __name__=="__main__":
    main()

        
import json
import os
from tqdm import tqdm
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import PdfFormatOption

PAPERS_DIR = "data/papers"
METADATA_FILE = "data/metadata.json"
CHUNKS_FILE = "data/chunks.json"

MAX_CHUNK_WORDS = 400
OVERLAP_WORDS = 50

from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    AcceleratorOptions
)

pipeline_options = PdfPipelineOptions()

pipeline_options.accelerator_options = AcceleratorOptions(
    num_threads=8,
    device="cuda"
)
pipeline_options.do_ocr = False
pipeline_options.do_table_structure = True  

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)

SKIP_SECTION_PATTERNS = [
    "references", "bibliography", "acknowledgment",
    "acknowledgements", "appendix", "funding"
]

def is_skip_section(heading):
    if not heading:
        return False
    lower = heading.lower().strip()
    return any(lower.startswith(p) for p in SKIP_SECTION_PATTERNS)

def split_into_chunks(text, arxiv_id, title, section, chunk_index_start):
    words = text.split()
    chunks = []
    start = 0
    chunk_index = chunk_index_start

    while start < len(words):
        end = start + MAX_CHUNK_WORDS
        chunk_words = words[start:end]
        chunk_content = " ".join(chunk_words)
        if section:
            prefix = f"Paper: {title}\nSection: {section}\n\n"
        else:
            prefix = f"Paper: {title}\n\n"

        chunks.append({
            "chunk_id": f"{arxiv_id}_{chunk_index}",
            "arxiv_id": arxiv_id,
            "title": title,
            "section": section or "",
            "text": prefix + chunk_content,
        })

        chunk_index += 1
        start += (MAX_CHUNK_WORDS - OVERLAP_WORDS)

    return chunks, chunk_index


def extract_and_chunk(pdf_path, arxiv_id, title):
    try:
        result = converter.convert(pdf_path)
        doc = result.document
    except Exception as e:
        print(f"Docling failed on {pdf_path}: {e}")
        return []

    chunks = []
    chunk_index = 0
    current_section = None
    current_text = ""
    skip_current_section = False

    first_section_done = False

    for element, level in doc.iterate_items():
        try:
            from docling.datamodel.document import SectionHeaderItem, TextItem, TableItem, ListItem
        except ImportError:
            from docling.datamodel.base_models import SectionHeaderItem, TextItem, TableItem, ListItem
        
        if isinstance(element, SectionHeaderItem):
            heading = element.text.strip() if element.text else ""
            if current_text.strip() and not skip_current_section and first_section_done:
                new_chunks, chunk_index = split_into_chunks(
                    current_text.strip(), arxiv_id, title, current_section, chunk_index
                )
                chunks.extend(new_chunks)

            current_section = heading
            current_text = ""
            skip_current_section = is_skip_section(heading)
            first_section_done = True
            continue

        if skip_current_section:
            continue

        # Handle text
        if isinstance(element, TextItem):
            text = element.text.strip() if element.text else ""
            if not text:
                continue
            # Skip very short lines - likely captions or labels
            if len(text.split()) < 8:
                continue
            current_text += " " + text

        # Handle tables - convert to text summary
        elif isinstance(element, TableItem):
            try:
                table_text = element.export_to_markdown(doc)
                if table_text and len(table_text.split()) > 10:
                    current_text += f"\n[TABLE]\n{table_text}\n"
            except:
                pass

        # Handle lists
        elif isinstance(element, ListItem):
            text = element.text.strip() if element.text else ""
            if text and len(text.split()) >= 5:
                current_text += " " + text

    # Flush final section
    if current_text.strip() and not skip_current_section:
        new_chunks, chunk_index = split_into_chunks(
            current_text.strip(), arxiv_id, title, current_section, chunk_index
        )
        chunks.extend(new_chunks)

    return chunks


def main():
    with open(METADATA_FILE) as f:
        content = f.read()
        if not content.strip():
            print("metadata.json is empty!")
            exit()
        papers = json.loads(content)

    all_chunks = []
    failed = []

    for arxiv_id, paper in tqdm(papers.items()):
        pdf_path = os.path.join(PAPERS_DIR, f"{arxiv_id}.pdf")

        if not os.path.exists(pdf_path):
            continue

        chunks = extract_and_chunk(pdf_path, arxiv_id, paper["title"])

        if not chunks:
            failed.append(arxiv_id)
            continue

        all_chunks.extend(chunks)

    with open(CHUNKS_FILE, "w") as f:
        json.dump(all_chunks, f, indent=2)

    print(f"Total chunks: {len(all_chunks)}")
    print(f"Failed: {len(failed)} papers")
    if failed:
        print(failed[:10])


if __name__ == "__main__":
    main()

import fitz
doc = fitz.open('data/papers/2504.19413.pdf')
for i, page in enumerate(doc):
    print(f'--- Page {i+1} ---')
    print(page.get_text()[:300])
    print()
    if i > 2:
        break


import json
with open('data/metadata.json') as f:
    papers = json.load(f)
titles = [p['title'].lower() for p in papers.values()]
targets = ['mem0', 'tau-bench', 'osworld', 'swe-agent', 'appworld', 'ui-tars', 'openhands']
for t in targets:
    found = any(t in title for title in titles)
    print(f'{t}: {"FOUND" if found else "MISSING"}')


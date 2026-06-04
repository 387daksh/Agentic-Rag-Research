
import json

def load_qtypes():
    qtypes = {}
    with open('eval/questions.jsonl', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                q = json.loads(line)
                qtypes[q['id']] = q.get('type', 'survey')
    return qtypes

qtypes = load_qtypes()

for config in ['full_agent', 'baseline']:
    by_type = {'factoid': [], 'comparative': [], 'survey': []}
    with open(f'predictions/{config}.jsonl', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                qtype = qtypes.get(r['id'], 'survey')
                by_type[qtype].append(len(r['cited_papers']))
    print(f'\n{config} avg citations by type:')
    for t, counts in by_type.items():
        print(f'  {t}: {sum(counts)/len(counts):.1f}')

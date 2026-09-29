#!/usr/bin/env python3
"""Import already converted six-key JSONL into an isolated, audited train bundle.

No label inference or synthetic examples. Source files must use the exact benchmark
schema. The caller supplies provenance and license evidence in a source catalog.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'benchmarks' / 'data'
SCHEMA = {'id': str, 'task': str, 'context': str, 'candidates': list,
          'ground_truth': str, 'metadata': dict}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    with path.open(encoding='utf-8') as f:
        for number, line in enumerate(f, 1):
            if line.strip():
                yield number, json.loads(line)


def text_hash(value):
    return hashlib.sha256(' '.join(value.split()).casefold().encode()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--catalog', type=Path, required=True, help='JSON source catalog')
    p.add_argument('--output', type=Path, required=True, help='new output directory outside benchmarks/data')
    a = p.parse_args()
    if a.output.resolve() == DATA.resolve() or DATA.resolve() in a.output.resolve().parents:
        p.error('output must be outside frozen benchmarks/data')
    if a.output.exists():
        p.error('output already exists; choose a fresh path')
    catalog = json.loads(a.catalog.read_text(encoding='utf-8'))
    sources = catalog['sources']
    if not sources:
        p.error('catalog has no sources')
    excluded = set()
    manifest = json.loads((DATA / 'manifest.json').read_text(encoding='utf-8'))
    for entry in manifest['tasks'].values():
        for _, row in rows(DATA / entry['file']):
            excluded.add(text_hash(row['context']))
    for _, row in rows(DATA / 'calibration_clean_16.jsonl'):
        excluded.add(text_hash(row['context']))
    seen_ids, seen_contexts, prepared, entries = set(), set(), {}, {}
    for source in sources:
        name = source['name']
        path = Path(source['path']).resolve()
        if name in entries or not name.isidentifier():
            raise ValueError(f'invalid or repeated source name: {name}')
        if source['split'] != 'train':
            raise ValueError(f'{name}: only declared train split allowed')
        if not source.get('license') or not source.get('license_url') or not source.get('url'):
            raise ValueError(f'{name}: source, license and evidence URL required')
        if path == DATA or DATA in path.parents:
            raise ValueError(f'{name}: benchmark evaluation data cannot be training input')
        if digest(path) != source['sha256']:
            raise ValueError(f'{name}: input SHA256 mismatch')
        accepted = []
        for number, row in rows(path):
            if set(row) != set(SCHEMA) or any(not isinstance(row[k], t) for k, t in SCHEMA.items()):
                raise ValueError(f'{name}:{number}: schema mismatch')
            if not row['id'] or not row['task'] or not row['context'] or not row['candidates']:
                raise ValueError(f'{name}:{number}: empty required field')
            if not all(isinstance(x, str) and x for x in row['candidates']) or len(set(row['candidates'])) != len(row['candidates']):
                raise ValueError(f'{name}:{number}: invalid candidates')
            if row['ground_truth'] not in row['candidates']:
                raise ValueError(f'{name}:{number}: ground_truth absent from candidates')
            if row['task'] != name or row['id'] in seen_ids:
                raise ValueError(f'{name}:{number}: task or id collision')
            seen_ids.add(row['id'])
            key = text_hash(row['context'])
            if key in excluded or key in seen_contexts:
                continue
            seen_contexts.add(key)
            accepted.append(row)
        if not accepted:
            raise ValueError(f'{name}: zero accepted rows')
        prepared[name] = accepted
        entries[name] = {k: source[k] for k in ('url', 'license', 'license_url', 'split')}
        entries[name]['input_sha256'] = source['sha256']
        entries[name]['samples_count'] = len(accepted)
        entries[name]['file'] = name + '.jsonl'
        entries[name]['sample_ids'] = [r['id'] for r in accepted]
        entries[name]['candidates'] = sorted({c for r in accepted for c in r['candidates']})
        entries[name]['candidates_count'] = len(entries[name]['candidates'])
    a.output.mkdir(parents=True)
    combined = []
    for name, accepted in prepared.items():
        payload = ''.join(json.dumps(r, ensure_ascii=False, separators=(',', ':')) + '\n' for r in accepted)
        target = a.output / (name + '.jsonl')
        target.write_text(payload, encoding='utf-8')
        entries[name]['sha256'] = digest(target)
        combined.append(payload)
    aggregate = a.output / 'all_benchmarks.jsonl'
    aggregate.write_text(''.join(combined), encoding='utf-8')
    result = {'version': '1.1.0', 'schema_version': 1, 'purpose': 'distillation_train',
              'schema': {'id': 'str', 'task': 'str', 'context': 'str', 'candidates': 'List[str]',
                         'ground_truth': 'str', 'metadata': 'dict'},
              'benchmark_dimensions': len(entries), 'total_samples': sum(e['samples_count'] for e in entries.values()),
              'tasks': entries, 'all_benchmarks': {'file': aggregate.name, 'sha256': digest(aggregate),
                                                    'samples_count': sum(e['samples_count'] for e in entries.values())}}
    (a.output / 'manifest.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f"wrote {result['total_samples']} rows across {len(entries)} sources to {a.output}")


if __name__ == '__main__':
    main()

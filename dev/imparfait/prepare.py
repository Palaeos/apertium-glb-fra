#!/usr/bin/env python3
"""Build the weakly labelled imparfait dataset from Tatoeba fra–eng pairs.

Labels come from the English translation of each French sentence with exactly one imparfait verb:
  "used to" / "would" + verb  → hab   (Globasa le du-V)
  "was|were V-ing"            → prog  (Globasa le nun V)
  anything else               → stat  (Globasa le V)
Sentences whose translations disagree are dropped.

Usage (from the repo root, after `make`):
  python3 -I dev/imparfait/prepare.py dev/imparfait/data dev/imparfait/data/dataset.jsonl
"""
import bz2
import importlib.util
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    'imparfait', os.path.join(HERE, '..', '..', 'fra-glb-imparfait.py'))
imparfait = importlib.util.module_from_spec(spec)
spec.loader.exec_module(imparfait)

IMPARFAIT_SURFACE = re.compile(r"\b\w+(ais|ait|aient|ions|iez)\b", re.I)
HAB = re.compile(r"\bused to\b|\bwould (?!you\b|like\b|rather\b|have\b|be able\b)\w+", re.I)
PROG = re.compile(r"\b(was|were)\s+(?!going\b)\w+ing\b", re.I)


def read_sentences(path, wanted=None):
    out = {}
    with bz2.open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 3:
                continue
            sid = parts[0]
            if wanted is None or sid in wanted:
                out[sid] = parts[2]
    return out


def label(english, lus, i):
    """Weak label from the English, constrained by the French analysis (spot-checked:
    "would" mostly came from conditionals, "was X-ing" from -ing adjectives after être)."""
    verb = lus[i][1]
    if re.search(r"\bused to\b", english, re.I):
        return 'hab'
    if HAB.search(english):
        return 'hab'
    if PROG.search(english):
        if verb in ('être', 'avoir') and 'en train de' not in ' '.join(l for _, l, _ in lus):
            return 'stat'
        return 'prog'
    return 'stat'


def is_unreal_condition(lus):
    """si + imparfait and sentences with a conditionnel: handled by the rtx (eger + bare V)."""
    return any(l == 'si' for _, l, _ in lus) or any('cni' in t for _, _, t in lus)


def main(datadir, outpath):
    links = {}
    with bz2.open(os.path.join(datadir, 'fra-eng_links.tsv.bz2'), 'rt') as f:
        for line in f:
            a, b = line.split()
            links.setdefault(a, []).append(b)
    fra = read_sentences(os.path.join(datadir, 'fra_sentences.tsv.bz2'), set(links))
    fra = {k: v for k, v in fra.items() if IMPARFAIT_SURFACE.search(v)}
    eng_ids = {e for k in fra for e in links[k]}
    eng = read_sentences(os.path.join(datadir, 'eng_sentences.tsv.bz2'), eng_ids)
    print('candidate French sentences: %d' % len(fra), file=sys.stderr)

    ids = sorted(fra)
    # Tag with this pair's own French analyser + tagger. Numbered marker words separate the
    # sentences (the tagger does not preserve \0 separators); they come back as ^*QQSEPnQQ$.
    payload = ''.join('%s\nQQSEP%dQQ\n' % (fra[i].replace('\n', ' '), n) for n, i in enumerate(ids))
    stream = subprocess.run(['apertium', '-d', '.', 'fra-glb-tagger'], input=payload,
                            capture_output=True, text=True, check=True).stdout
    tagged = re.split(r'\^\*QQSEP\d+QQ\$', stream)
    assert len(tagged) >= len(ids), (len(tagged), len(ids))

    n = {'hab': 0, 'prog': 0, 'stat': 0}
    skipped = 0
    with open(outpath, 'w', encoding='utf-8') as out:
        for sid, stream in zip(ids, tagged):
            lus = [imparfait.parse_lu(m.group(1)) for m in imparfait.LU.finditer(stream)]
            idx = [i for i, (_, _, t) in enumerate(lus) if 'pii' in t]
            if len(idx) != 1:
                continue
            if is_unreal_condition(lus):
                continue
            labels = {label(eng[e], lus, idx[0]) for e in links[sid] if e in eng}
            if len(labels) != 1:
                skipped += 1
                continue
            y = labels.pop()
            n[y] += 1
            out.write(json.dumps({'id': sid, 'fr': fra[sid],
                                  'en': [eng[e] for e in links[sid] if e in eng],
                                  'y': y, 'x': imparfait.features(lus, idx[0])},
                                 ensure_ascii=False) + '\n')
    print('labelled: %s; dropped for conflicting translations: %d' % (n, skipped), file=sys.stderr)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])

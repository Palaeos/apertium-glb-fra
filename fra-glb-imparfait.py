#!/usr/bin/env python3
"""French imparfait sense classifier for the fra→glb pipeline.

Runs after apertium-tagger. For every verb in the imparfait (<pii>) it predicts whether the
form is habitual ("used to" → Globasa le du-V), progressive ("was V-ing" → le nun V) or a
state/background (plain le V), and appends <hab> or <prog> after <pii>. The rtx turns those
tags into the Globasa forms; untagged verbs keep plain le. [xwexi 08:56, 08:114]

The model is a logistic regression trained by dev/imparfait/train.py on Tatoeba fra–eng pairs,
weakly labelled from the English translation. Only the standard library is used here.

Usage: fra-glb-imparfait.py MODEL.json [-z]   (Apertium stream on stdin/stdout)
"""
import json
import math
import re
import sys

LU = re.compile(r'\^((?:\\.|[^$\\])*)\$')
STATIVE = {'être', 'avoir', 'savoir', 'connaître', 'vouloir', 'pouvoir', 'devoir', 'aimer',
           'croire', 'penser', 'sembler', 'paraître', 'falloir', 'exister', 'détester',
           'préférer', 'espérer', 'valoir', 'appartenir', 'contenir', 'coûter', 'peser'}
HAB_CUES = {'souvent', 'toujours', 'chaque', 'tout', 'habitude', 'parfois', 'quelquefois',
            'généralement', 'régulièrement', 'fréquemment', 'jamais', 'autrefois', 'jadis',
            'enfance', 'matin', 'soir', 'dimanche', 'lundi', 'samedi', 'été', 'hiver'}
PROG_CUES = {'quand', 'lorsque', 'pendant', 'tandis', 'alors', 'soudain', 'train',
             'moment', 'déjà', 'encore'}
TENSES = ('pri', 'pii', 'ifi', 'fti', 'cni', 'prs', 'imp', 'inf', 'pp', 'ger')


def parse_lu(body):
    """'lemma<t1><t2>…' (tagger output) or 'surface/lemma<t1>…' → (surface, lemma, [tags])."""
    parts = re.split(r'(?<!\\)/', body)
    if len(parts) >= 2:
        surface, analysis = parts[0], parts[1]
    else:
        surface, analysis = '', parts[0]
    analysis = analysis.split('+')[0]
    lemma = analysis.split('<', 1)[0]
    tags = re.findall(r'<([^>]+)>', analysis)
    return surface, lemma.lower(), tags


def features(lus, i):
    """Features for the imparfait verb at index i of a sentence given as (surface, lemma, tags)."""
    _, lemma, tags = lus[i]
    f = {'lem=' + lemma: 1.0, 'stative': 1.0 if lemma in STATIVE else 0.0, 'bias': 1.0}
    for d in range(-5, 6):
        j = i + d
        if d == 0 or j < 0 or j >= len(lus):
            continue
        lem = lus[j][1]
        f['w%+d=%s' % (d, lem)] = 1.0
        f['win=' + lem] = 1.0
        if lem in HAB_CUES:
            f['habcue'] = 1.0
        if lem in PROG_CUES:
            f['progcue'] = 1.0
    seen = set()
    for j, (_, lem, t) in enumerate(lus):
        if j == i:
            continue
        for tense in TENSES:
            if tense in t:
                seen.add(tense)
        if lem in HAB_CUES:
            f['s_habcue'] = 1.0
        if lem in PROG_CUES:
            f['s_progcue'] = 1.0
    for tense in seen:
        f['tense=' + tense] = 1.0
    if 'en train de' in ' '.join(l for _, l, _ in lus):
        f['entrain'] = 1.0
    return f


def predict(model, f):
    scores = {}
    for cls, w in model['weights'].items():
        s = model['intercept'][cls]
        for k, v in f.items():
            s += w.get(k, 0.0) * v
        scores[cls] = s
    m = max(scores.values())
    z = sum(math.exp(s - m) for s in scores.values())
    probs = {c: math.exp(s - m) / z for c, s in scores.items()}
    best = max(probs, key=probs.get)
    return best, probs[best]


def annotate(text, model):
    """Append <hab>/<prog> to confident imparfait verbs in an Apertium stream chunk."""
    matches = list(LU.finditer(text))
    out, last, sent = [], 0, []

    def flush(sentence):
        lus = [parse_lu(m.group(1)) for m in sentence]
        edits = {}
        for i, (_, _, tags) in enumerate(lus):
            if 'pii' not in tags:
                continue
            cls, p = predict(model, features(lus, i))
            if cls in ('hab', 'prog') and p >= model.get('threshold', 0.6):
                body = sentence[i].group(1)
                edits[sentence[i].start()] = body.replace('<pii>', '<pii><%s>' % cls, 1)
        return edits

    edits = {}
    for m in matches:
        sent.append(m)
        if '<sent>' in m.group(1):
            edits.update(flush(sent))
            sent = []
    if sent:
        edits.update(flush(sent))
    for m in matches:
        out.append(text[last:m.start()])
        out.append('^%s$' % edits.get(m.start(), m.group(1)))
        last = m.end()
    out.append(text[last:])
    return ''.join(out)


def main():
    args = [a for a in sys.argv[1:] if a != '-z']
    null_flush = '-z' in sys.argv[1:]
    if not args:
        sys.exit(__doc__)
    with open(args[0], encoding='utf-8') as f:
        model = json.load(f)
    data = sys.stdin.read()
    if null_flush:
        sys.stdout.write('\0'.join(annotate(chunk, model) for chunk in data.split('\0')))
    else:
        sys.stdout.write(annotate(data, model))
    sys.stdout.flush()


if __name__ == '__main__':
    main()

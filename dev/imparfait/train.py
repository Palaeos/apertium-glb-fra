#!/usr/bin/env python3
"""Train the imparfait sense classifier and export it for fra-glb-imparfait.py.

Usage (inside dev/imparfait/.venv, from the repo root):
  dev/imparfait/.venv/bin/python -I dev/imparfait/train.py dev/imparfait/data/dataset.jsonl fra-glb.imparfait.json

Splits 90/10 (stratified, fixed seed), tunes the confidence threshold on a validation slice of the
training part, then reports the held-out test set against the always-"stat" baseline. The model is
written only if it beats the baseline on macro-F1 without losing more than 0.5 points of accuracy;
otherwise the rtx default (plain le) stands.
"""
import json
import sys

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

SEED = 13


def apply_threshold(model, X, t):
    probs = model.predict_proba(X)
    classes = list(model.classes_)
    stat = classes.index('stat')
    out = []
    for p in probs:
        best = p.argmax()
        out.append(classes[best] if (best != stat and p[best] >= t) else 'stat')
    return out


def main(data, outpath):
    rows = [json.loads(line) for line in open(data, encoding='utf-8')]
    X = [r['x'] for r in rows]
    y = [r['y'] for r in rows]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.1, stratify=y, random_state=SEED)
    Xfit, Xval, yfit, yval = train_test_split(Xtr, ytr, test_size=0.15, stratify=ytr, random_state=SEED)

    vec = DictVectorizer()
    best = None
    for C in (0.1, 0.3, 1.0, 3.0):
        for cw in (None, 'balanced'):
            clf = LogisticRegression(C=C, class_weight=cw, max_iter=3000)
            clf.fit(vec.fit_transform(Xfit), yfit)
            for t in (0.4, 0.5, 0.6, 0.7, 0.8):
                pred = apply_threshold(clf, vec.transform(Xval), t)
                f = f1_score(yval, pred, average='macro')
                if best is None or f > best[0]:
                    best = (f, C, cw, t)
    f, C, cw, t = best
    print('selected C=%s class_weight=%s threshold=%s (val macro-F1 %.3f)' % (C, cw, t, f))

    vec = DictVectorizer()
    clf = LogisticRegression(C=C, class_weight=cw, max_iter=3000)
    clf.fit(vec.fit_transform(Xtr), ytr)
    pred = apply_threshold(clf, vec.transform(Xte), t)
    base = ['stat'] * len(yte)
    acc, acc0 = accuracy_score(yte, pred), accuracy_score(yte, base)
    f1, f10 = f1_score(yte, pred, average='macro'), f1_score(yte, base, average='macro')
    print('\nheld-out test (%d examples)' % len(yte))
    print(classification_report(yte, pred, digits=3, zero_division=0))
    print('baseline always-stat: accuracy %.3f, macro-F1 %.3f' % (acc0, f10))
    print('model:                accuracy %.3f, macro-F1 %.3f' % (acc, f1))

    if not (f1 > f10 and acc >= acc0 - 0.005):
        print('\nModel does not beat the baseline; not writing %s.' % outpath)
        return 1

    names = vec.get_feature_names_out()
    weights, intercept = {}, {}
    for k, cls in enumerate(clf.classes_):
        weights[cls] = {names[j]: round(float(w), 5) for j, w in enumerate(clf.coef_[k]) if abs(w) >= 1e-3}
        intercept[cls] = float(clf.intercept_[k])
    model = {'classes': list(clf.classes_), 'threshold': t, 'weights': weights, 'intercept': intercept,
             'source': 'Tatoeba fra-eng (CC BY 2.0 FR), weak labels from English; see dev/imparfait',
             'test': {'accuracy': round(acc, 4), 'macro_f1': round(f1, 4),
                      'baseline_accuracy': round(acc0, 4), 'baseline_macro_f1': round(f10, 4)}}
    with open(outpath, 'w', encoding='utf-8') as fo:
        json.dump(model, fo, ensure_ascii=False, separators=(',', ':'))
    print('\nwrote %s (%d features kept)' % (outpath, sum(len(w) for w in weights.values())))
    return 0


if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1], sys.argv[2]))

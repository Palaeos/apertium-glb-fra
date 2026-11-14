#!/usr/bin/env python3
"""Add the Globasa gender tag to bidix noun entries that lack one.

The Globasa generator only produces nouns tagged <n><ut|nt|m|f|GD>; a bare <n> fails (#myaw).
The gender comes from the noun's paradigm in the monolingual dictionary:
__n_ut → ut, __n_nt → nt, __n_m → m, __n_f → f, __n → GD.

The gender also drives te/to choice for anaphoric pronouns (nt → to, otherwise te).

Usage: dev/fix-noun-gender.py BIDIX GLB_MONODIX   (rewrites BIDIX in place, reports changes)
"""
import re
import sys

PARADIGM_GENDER = {'__n_ut': 'ut', '__n_nt': 'nt', '__n_m': 'm', '__n_f': 'f', '__n': 'GD'}


def load_genders(monodix):
    genders = {}
    pat = re.compile(r'<e lm="([^"]+)">\s*<i>[^<]*</i><par n="(__n(?:_ut|_nt|_m|_f)?)"/>')
    with open(monodix, encoding='utf-8') as f:
        for line in f:
            m = pat.search(line)
            if m:
                genders.setdefault(m.group(1), PARADIGM_GENDER[m.group(2)])
    return genders


def main(bidix, monodix):
    genders = load_genders(monodix)
    # Left side "lemma<s n="n"/></l>" (spacing inside the tag varies in the hand-written section)
    pat = re.compile(r'<l>([^<]+)<s n\s*=\s*"n"\s*/></l>')
    text = open(bidix, encoding='utf-8').read()
    fixed, missing = [], []

    def repl(m):
        lemma = m.group(1)
        g = genders.get(lemma)
        if g is None:
            missing.append(lemma)
            return m.group(0)
        fixed.append((lemma, g))
        return '<l>%s<s n="n"/><s n="%s"/></l>' % (lemma, g)

    text = pat.sub(repl, text)
    open(bidix, 'w', encoding='utf-8').write(text)
    for lemma, g in fixed:
        print('fixed   %s -> %s' % (lemma, g))
    for lemma in missing:
        print('MISSING %s (not a noun in the Globasa monodix)' % lemma)
    print('%d fixed, %d not found' % (len(fixed), len(missing)))


if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])

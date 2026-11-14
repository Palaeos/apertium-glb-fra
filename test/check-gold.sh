#!/bin/sh
# Translate each French line of test/fra-glb-gold.tsv and compare with the expected Globasa.
# Usage: test/check-gold.sh [-v] [MODE [GOLD.tsv]]   (-v also prints passing lines)
#   default: mode fra-glb against test/fra-glb-gold.tsv
#   test/check-gold.sh fra-glb_perf test/fra-glb_perf-gold.tsv   (passé composé → nun le)
cd "$(dirname "$0")/.." || exit 1
verbose=
[ "$1" = -v ] && { verbose=-v; shift; }
mode=${1:-fra-glb}
goldfile=${2:-test/$mode-gold.tsv}
pass=0; fail=0
tab=$(printf '\t')
while IFS="$tab" read -r src gold; do
    case "$src" in ''|'#'*) continue ;; esac
    out=$(printf '%s\n' "$src" | apertium -d . "$mode" | sed 's/[[:space:]]*$//')
    ok=0
    old_ifs=$IFS; IFS='|'
    for g in $gold; do
        g=$(printf '%s' "$g" | sed 's/^[[:space:]]*//; s/[[:space:]]*$//')
        [ "$out" = "$g" ] && ok=1
    done
    IFS=$old_ifs
    if [ $ok = 1 ]; then
        pass=$((pass+1)); [ "$verbose" = -v ] && printf 'PASS  %s\n      %s\n' "$src" "$out"
    else
        fail=$((fail+1)); printf 'FAIL  %s\n  got %s\n want %s\n' "$src" "$out" "$gold"
    fi
done < "$goldfile"
echo "passed $pass / $((pass+fail))"
[ $fail = 0 ]

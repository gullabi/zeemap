#!/usr/bin/env bash
# The check runner (TASKS.md, "The check runner"; VERIFICATION.md section 2).
#   tools/check.sh <tier> [--fast] [V<k>[<letter>] ...] [-- pytest arguments]
# With only a tier: every check of that tier and the tiers below. With items: only those tests (by name), and an item
# of a higher tier than <tier> is an error. --fast leaves out the tier-1 tests marked `slow` (end-to-end fits) and
# prints `SUMMARY tier 1 (fast): ...`; the other tiers are not affected. Prints one `SUMMARY tier <n>: ...` line per
# tier run; exits 1 if anything failed or errored, if a named item selected no test (conftest.py names each such
# item), or (tier 2 and above) if anything was skipped.
# Python: $ZEEMAP_PY, else .venv/bin/python, else python3.
set -u
cd "$(dirname "$0")/.."
tier=${1:-}; shift || true
[[ "$tier" =~ ^[123]$ ]] || { echo "usage: tools/check.sh <1|2|3> [--fast] [V<k>[<letter>] ...] [-- pytest args]" >&2; exit 2; }
items=(); extra=(); fast=0
while [ $# -gt 0 ]; do
  case "$1" in
    --) shift; extra=("$@"); break ;;
    --fast) fast=1 ;;
    V*) items+=("$1") ;;
    *) echo "check.sh: unexpected argument $1" >&2; exit 2 ;;
  esac
  shift
done
PY=${ZEEMAP_PY:-}
[ -z "$PY" ] && [ -x .venv/bin/python ] && PY=.venv/bin/python
[ -z "$PY" ] && PY=python3

tier_of_item() {  # V<k>[<letter>] -> tier
  local k; k=$(echo "$1" | sed -E 's/^V([0-9]+).*/\1/')
  if [ "$k" -ge 1 ] && [ "$k" -le 8 ] || [ "$k" -eq 14 ]; then echo 1
  elif [ "$k" -ge 9 ] && [ "$k" -le 13 ]; then echo 2
  elif [ "$k" -ge 15 ] && [ "$k" -le 19 ]; then echo 3
  else echo 0; fi
}

status=0
run_tier() {  # run_tier <n> [item ...]
  local n=$1; shift
  local out rc p f e s mark="tier$n" label="tier $n"
  if [ "$fast" -eq 1 ] && [ "$n" -eq 1 ]; then mark="tier1 and not slow"; label="tier 1 (fast)"; fi
  out=$("$PY" -m pytest -q -rN -m "$mark" "$@" "${extra[@]}" 2>&1); rc=$?
  p=$(echo "$out" | grep -oE '[0-9]+ passed' | tail -1 | cut -d' ' -f1); p=${p:-0}
  f=$(echo "$out" | grep -oE '[0-9]+ failed' | tail -1 | cut -d' ' -f1); f=${f:-0}
  e=$(echo "$out" | grep -oE '[0-9]+ errors?' | tail -1 | cut -d' ' -f1); e=${e:-0}
  s=$(echo "$out" | grep -oE '[0-9]+ skipped' | tail -1 | cut -d' ' -f1); s=${s:-0}
  if [ $rc -ne 0 ] && [ $rc -ne 5 ]; then echo "$out"; fi
  if [ $rc -eq 5 ] && [ $# -gt 0 ]; then echo "$out" | tail -3; echo "check.sh: tier $n: the items ${*:2} selected no test" >&2; status=1; fi
  echo "SUMMARY $label: $p passed, $f failed, $e errors, $s skipped"
  if [ "$f" -gt 0 ] || [ "$e" -gt 0 ]; then status=1; fi
  if [ "$n" -ge 2 ] && [ "$s" -gt 0 ]; then echo "check.sh: tier $n: $s skipped checks count as failures before a merge (GATES.md)" >&2; status=1; fi
  if [ $rc -ne 0 ] && [ $rc -ne 5 ] && [ $rc -ne 1 ]; then status=1; fi  # usage/internal errors
}

if [ ${#items[@]} -eq 0 ]; then
  for n in $(seq 1 "$tier"); do run_tier "$n"; done
else
  declare -A by_tier=()
  for it in "${items[@]}"; do
    t=$(tier_of_item "$it")
    if [ "$t" -eq 0 ]; then echo "check.sh: $it is not a check item" >&2; exit 2; fi
    if [ "$t" -gt "$tier" ]; then echo "check.sh: $it belongs to tier $t, above the requested tier $tier" >&2; exit 2; fi
    by_tier[$t]="${by_tier[$t]:-} $it"
  done
  for n in $(seq 1 "$tier"); do
    [ -n "${by_tier[$n]:-}" ] || continue
    # shellcheck disable=SC2086
    run_tier "$n" --items ${by_tier[$n]}
  done
fi
exit $status

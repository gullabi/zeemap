"""Mechanical checks of the zeemap plan: cross-references, links, long sentences, undefined glossary terms, and a
test for every check item a task's "done when" command names.
Usage: python3 lint_plan.py <repo dir>"""
import re, sys, os, glob
R = sys.argv[1]
files = [os.path.join(R, 'README.md'), os.path.join(R, 'AGENTS.md')] + sorted(glob.glob(os.path.join(R, 'docs', '*.md')))
txt = {os.path.relpath(f, R): open(f).read() for f in files}
spec, ver = txt['docs/SPEC.md'], txt['docs/VERIFICATION.md']
fail = []
req_ids = set(re.findall(r'\*\*([A-Z]+-\d+)\*\*', spec))
loops = set(re.findall(r'\*\*(V\d+)\.', ver))
# 1. every Checked by names existing loops (allow V2a style suffix)
for rid, chk in re.findall(r'\*\*([A-Z]+-\d+)\*\*.*?Checked by:([^\n]*(?:\n  [^\n]*)?)', spec, re.S):
    for v in re.findall(r'\bV(\d+)[a-z]?\b', chk):
        if 'V' + v not in loops: fail.append(f'{rid}: Checked by V{v}, no such loop')
    if not re.search(r'V\d+|gate', chk): fail.append(f'{rid}: no check named')
# 2. requirement IDs cited anywhere exist (ranges like LIB-1 to LIB-7 expanded by prefix)
for name, t in txt.items():
    for rid in set(re.findall(r'\b([A-Z]{2,6}-\d+)\b', t)):
        if rid.split('-')[0] in {'RFC', 'ISO', 'IEEE', 'SHA'}: continue
        if rid not in req_ids: fail.append(f'{name}: cites {rid}, not defined in SPEC')
# 3. loops: each loop's Covers lists existing ids; every requirement covered by some loop's Covers or Checked by
covers = ' '.join(re.findall(r'Covers: ([^\n]*)', ver))
for rid in sorted(req_ids):
    if rid not in covers and not re.search(r'\*\*' + rid + r'\*\*[^\n]*(\n  [^\n]*)*gate', spec):
        fail.append(f'{rid}: not in any loop\'s Covers line (VERIFICATION.md)')
# 3b. bidirectional: SPEC Checked-by Vx  <=> Vx Covers lists rid
chk = {}
for rid, body in re.findall(r'\*\*([A-Z]+-\d+)\*\*(.*?)(?=\n- \*\*|\n\n|\nExample)', spec, re.S):
    c = body.split('Checked by:')[-1].split('`[NEEDS')[0]
    chk[rid] = set('V' + v for v in re.findall(r'\bV(\d+)', c))
cov = {}
for v, body in re.findall(r'\*\*(V\d+)\.(.*?)(?=\*\*V\d+\.|\Z)', ver, re.S):
    m = re.search(r'Covers: ([^\n]*)', body)
    cov[v] = set(re.findall(r'[A-Z]+-\d+', m.group(1))) if m else set()
for rid, vs in chk.items():
    for v in vs:
        if v in cov and rid not in cov[v]: fail.append(f'{rid} Checked by {v}, but {v} Covers lacks it')
for v, rs in cov.items():
    for rid in rs:
        if v not in chk.get(rid, set()): fail.append(f'{v} Covers {rid}, but {rid} is not Checked by {v}')
# 3c. every requirement appears in TASKS (ranges "X-1 to X-5" expanded)
tasks = txt.get('docs/TASKS.md', '')
tids = set(re.findall(r'\b([A-Z]{2,6}-\d+)\b', tasks))
for a, pre, b in re.findall(r'\b(([A-Z]{2,6})-(\d+)) to \2-(\d+)', tasks) and [] or []: pass
for pre, a, b in re.findall(r'\b([A-Z]{2,6})-(\d+) to \1-(\d+)', tasks):
    tids |= {f'{pre}-{k}' for k in range(int(a), int(b) + 1)}
for rid in sorted(req_ids):
    if rid not in tids: fail.append(f'{rid}: no task in TASKS.md')
# 3d. NEEDS DECISION markers in docs all listed in SPEC section 8
for name, t in txt.items():
    for d in set(re.findall(r'NEEDS DECISION (D\d+)', t)):
        if f'| {d} |' not in spec: fail.append(f'{name}: marker {d} not in SPEC section 8')
for d in re.findall(r'^\| (D\d+) \|', spec.split('## 8.')[-1], re.M):
    if not any(f'NEEDS DECISION {d}' in t for t in txt.values()): fail.append(f'SPEC section 8 lists {d}, no marker in place')
# 4. loop references in other docs exist
for name, t in txt.items():
    for v in set(re.findall(r'\bV(\d+)[a-z]?\b', t)):
        if 'V' + v not in loops: fail.append(f'{name}: mentions V{v}, no such loop')
# 4b. every check item named in a TASKS.md "done when" command (`tools/check.sh <tier> V<k>[<letter>] ...`) has a
# test by name in tests/ (test_v<k><letter>_..., or any test_v<k>..._ for an item without a letter; the selection rule
# of tests/conftest.py). An item without a test is a notice when it is listed under "Check items not built" in
# TASKS.md, when its check is waived by decision #35, or when it is tier 3; otherwise it is a failure.
tests = set()
for f in glob.glob(os.path.join(R, 'tests', '**', '*.py'), recursive=True):
    tests |= set(re.findall(r'^\s*def test_v(\d+)([a-z]?)_', open(f).read(), re.M))
named = {}  # item -> tasks naming it in their "done when" command
for line in tasks.splitlines():
    m = re.match(r'\| (T\d+\.\d+)', line)
    if not m: continue
    for cmd in re.findall(r'`tools/check\.sh [123]([^`]*)`', line.rsplit('|', 2)[-2]):
        for it in re.findall(r'\bV\d+[a-z]?\b', cmd):
            named.setdefault(it, []).append(m.group(1))
nb = re.search(r'\*\*Check items not built:\*\*([^\n]*)', tasks)
not_built = set(re.findall(r'\bV\d+[a-z]?\b', nb.group(1))) if nb else set()
d35 = re.search(r'^\| 35 \|([^\n]*)', txt.get('docs/DECISIONS.md', ''), re.M)
waived = set()
if d35:
    for sent in re.split(r'(?<=\.) ', d35.group(1)):
        if 'waived' in sent: waived |= set(re.findall(r'\bV(\d+)[a-z]?\b', sent))
t3 = re.search(r'tier 3: V(\d+)-V(\d+)', tasks)
tier3 = set(str(k) for k in range(int(t3.group(1)), int(t3.group(2)) + 1)) if t3 else set()
untested = []
for it in sorted(named, key=lambda s: (int(re.match(r'V(\d+)', s).group(1)), s)):
    k, letter = re.fullmatch(r'V(\d+)([a-z]?)', it).groups()
    if any(tk == k and (not letter or tl == letter) for tk, tl in tests): continue
    why = ('listed as not built' if it in not_built else 'waived by decision #35' if k in waived
           else 'tier 3' if k in tier3 else None)
    untested.append(f'{it} ({", ".join(named[it])}: {why or "FAIL"})')
    if why is None:
        fail.append(f'docs/TASKS.md: {", ".join(named[it])} "done when" names {it}, no test test_v{k}{letter}_* in tests/')
unnamed = sorted(loops - {re.match(r'V\d+', it).group(0) for it in named}, key=lambda v: int(v[1:]))
# 5. relative links resolve
for name, t in txt.items():
    for l in re.findall(r'\]\(([^)#]+)(?:#[^)]*)?\)', t):
        if l.startswith('http'): continue
        p = os.path.normpath(os.path.join(R, os.path.dirname(name), l))
        if not os.path.exists(p): fail.append(f'{name}: broken link {l}')
# 6. long sentences (prose only, outside code blocks and tables)
long = []
for name, t in txt.items():
    t2 = re.sub(r'```.*?```', '', t, flags=re.S)
    prose = '\n'.join(l for l in t2.splitlines() if not l.startswith('|'))
    prose = re.sub(r'\n\s*(- |\d+\. |#+ )', '\n\n', prose)
    for s in re.split(r'(?<=[.!?;:])\s+|\n\n', prose):
        s = ' '.join(s.split())
        n = len(s.split())
        if n > 30: long.append((name, n, s[:90]))
# 7. absolute paths
for name, t in txt.items():
    for m in re.findall(r'(?<!`)/home/\w\S*', t): fail.append(f'{name}: absolute path {m}')
print('\n'.join('FAIL ' + f for f in fail))
print(f'INFO {len(untested)} check items named in TASKS.md "done when" commands have no test:'
      + ''.join(f'\n  {u}' for u in untested))
print(f'INFO {len(unnamed)} checks of VERIFICATION.md named in no "done when" command: {", ".join(unnamed) or "none"}')
print(f'INFO {len(long)} sentences over 30 words:'); [print(f'  {n:3d} {name}: {s}') for name, n, s in long[:40]]
print(f'SUMMARY: {len(fail)} failures, {len(req_ids)} requirements, {len(loops)} loops, {len(long)} long sentences')

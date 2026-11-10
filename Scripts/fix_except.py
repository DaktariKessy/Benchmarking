#!/usr/bin/env python3
"""Insert the missing `except ValueError` clause into bench_core.run_eval.

Finds the line `m = res["metrics"]` at 12 spaces, walks back over blank
lines, and inserts the clause there -- but only if what precedes it is not
already an except/finally. Writes a .bak first and refuses to save unless
the result parses.
"""
import ast, shutil, sys

path = sys.argv[1] if len(sys.argv) > 1 else "bench_core.py"
src = open(path).read().splitlines(keepends=True)

CLAUSE = [
    '            except ValueError as e:\n',
    '                skips["compare rejected"] += 1\n',
    '                if skips["compare rejected"] <= 3:\n',
    '                    print(f"[{n}] compare rejected: {e}")\n',
    '                continue\n',
    '\n',
]

hits = [i for i, l in enumerate(src) if l.rstrip("\n") == '            m = res["metrics"]']
if len(hits) != 1:
    sys.exit(f"found {len(hits)} lines matching `m = res[\"metrics\"]` at 12 "
             f"spaces; expected exactly 1 -- fix by hand")
i = hits[0]

j = i - 1
while j >= 0 and not src[j].strip():
    j -= 1
prev = src[j].strip()
if prev.startswith(("except", "finally")) or "compare rejected" in prev:
    sys.exit("an except/finally clause already looks present -- nothing done")

print(f"inserting before line {i+1}; the line above it is:\n    {prev[:70]}")

new = src[:j+1] + ["\n"] + CLAUSE + src[i:]
try:
    ast.parse("".join(new))
except SyntaxError as e:
    sys.exit(f"result would NOT parse ({e.lineno}: {e.msg}) -- file untouched")

shutil.copy2(path, path + ".bak")
open(path, "w").writelines(new)
print(f"patched {path}  (backup at {path}.bak)")
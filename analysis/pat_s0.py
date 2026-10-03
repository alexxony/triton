import gzip, json, os, sys
from collections import defaultdict
sys.path.insert(0, "/root/w"); os.chdir("/root/w")
import sweep
dump = json.load(gzip.open("dump_a_cfgs.json.gz", "rt"))
by = {c["key"]: c for c in dump["cfgs"]}
plan = [c for c in sweep.build_plan() if c["op"] in sweep.ARG_OPS and c["dtype"] != "int32"]
tested = defaultdict(int); viol = defaultdict(int); nocfg = 0
for c in plan:
    k = sweep.cfg_key(c)
    names = [sweep.base_name(n[0]) for n in sweep.patterns_for(c)]
    d = by.get(k)
    vp = set(d["viol_pats"]) if d else set()
    vb = set(sweep.base_name(x) for x in vp)
    for n in set(names):
        tested[n] += 1
        viol[n] += int(n in vb)
print("float arg configs:", len(plan), " with dump record:", sum(1 for c in plan if sweep.cfg_key(c) in by))
print("%-18s %8s %9s %7s" % ("pattern(base)", "tested", "violating", "rate"))
for n in sorted(tested, key=lambda x: -viol[x] / tested[x]):
    print("%-18s %8d %9d %7.3f" % (n, tested[n], viol[n], viol[n] / tested[n]))
# how many violating patterns per config
cnt = defaultdict(int)
for d in dump["cfgs"]:
    if d.get("kinds"): cnt[len(set(sweep.base_name(x) for x in d["viol_pats"]))] += 1
print("violating cfgs by number of violating pattern bases:", dict(sorted(cnt.items())))

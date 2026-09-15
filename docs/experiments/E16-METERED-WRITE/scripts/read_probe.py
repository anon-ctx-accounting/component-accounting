import json, os, random, time, urllib.request, urllib.error

key = os.environ["KARC_ANTHROPIC_KEY"]
MODEL = "claude-sonnet-4-6"

rng = random.Random(1313)
vocab = ["ledger","carry","residency","supersession","provenance","artifact","tier",
         "eviction","budget","token","prefix","transcript","invariant","closure"]
pfx = "\n".join(["# E16-READ synthetic cache-probe prefix"] +
                [f"{i:04d} " + " ".join(rng.choice(vocab) for _ in range(10)) for i in range(140)])

def call(suffix):
    body = {"model": MODEL, "max_tokens": 16,
            "system": [{"type": "text", "text": pfx,
                        "cache_control": {"type": "ephemeral", "ttl": "1h"}}],
            "messages": [{"role": "user", "content": suffix}]}
    req = urllib.request.Request("https://api.anthropic.com/v1/messages",
        data=json.dumps(body).encode(),
        headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                 "content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.load(r)

print(f"prefix chars={len(pfx)} est_tokens={len(pfx)//4}")
rows = []
for n in range(4):
    if n:
        time.sleep(4)
    d = call(f"Reply with one word: item{n}")
    u = d["usage"]
    cc = u.get("cache_creation", {})
    rows.append({"call": n, "stop_reason": d.get("stop_reason"),
                 "text": "".join(b.get("text","") for b in d.get("content",[]))[:40],
                 "geo": u.get("inference_geo"), "tier": u.get("service_tier"),
                 "input": u["input_tokens"], "create": u["cache_creation_input_tokens"],
                 "c1h": cc.get("ephemeral_1h_input_tokens"), "c5m": cc.get("ephemeral_5m_input_tokens"),
                 "read": u["cache_read_input_tokens"], "output": u["output_tokens"]})
    r = rows[-1]
    print(f"call {n}: stop={r['stop_reason']:>10s} out={r['output']:3d} "
          f"input={r['input']:3d} create={r['create']:5d} read={r['read']:5d} "
          f"geo={r['geo']} tier={r['tier']} text={r['text']!r}")

json.dump({"model": MODEL, "prefix_chars": len(pfx), "rows": rows},
          open("read-probe.json","w"), indent=2, sort_keys=True)

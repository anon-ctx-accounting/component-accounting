import json, os, random, hashlib, urllib.request, urllib.error

key = os.environ["KARC_ANTHROPIC_KEY"]
MODEL = "claude-sonnet-4-6"

def prefix(label, blocks=520, seed=1313):
    rng = random.Random(seed)
    vocab = ["ledger","carry","residency","supersession","provenance","artifact","tier",
             "eviction","budget","token","prefix","transcript","invariant","closure"]
    out = [f"# {label} synthetic cache-probe prefix"]
    for i in range(blocks):
        out.append(f"{i:04d} " + " ".join(rng.choice(vocab) for _ in range(10)))
    return "\n".join(out)

def call(pfx, ttl, suffix, beta):
    body = {"model": MODEL, "max_tokens": 8,
            "system": [{"type": "text", "text": pfx,
                        "cache_control": {"type": "ephemeral", "ttl": ttl}}],
            "messages": [{"role": "user", "content": suffix}]}
    h = {"x-api-key": key, "anthropic-version": "2023-06-01",
         "content-type": "application/json"}
    if beta:
        h["anthropic-beta"] = beta
    req = urllib.request.Request("https://api.anthropic.com/v1/messages",
                                 data=json.dumps(body).encode(), headers=h)
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            return json.load(r).get("usage")
    except urllib.error.HTTPError as e:
        return {"_http": e.code, "_body": e.read().decode()[:400]}

rows = []
for ttl in ("1h", "5m"):
    pfx = prefix(f"E16-TTL-{ttl}")
    tok_est = len(pfx) // 4
    for phase, suffix in (("cold", "Reply with one word: alpha"),
                          ("warm", "Reply with one word: beta")):
        u = call(pfx, ttl, suffix, None)
        if isinstance(u, dict) and u.get("_http") == 400 and "beta" in u.get("_body", ""):
            u = call(pfx, ttl, suffix, "extended-cache-ttl-2025-04-11")
            note = "beta-header-required"
        else:
            note = "no-beta-header"
        rows.append({"ttl": ttl, "phase": phase, "prefix_sha256": hashlib.sha256(pfx.encode()).hexdigest()[:16],
                     "prefix_chars": len(pfx), "prefix_tokens_est": tok_est, "note": note, "usage": u})
        print(f"ttl={ttl:3s} {phase:4s} [{note}] {json.dumps(u, sort_keys=True)}")

json.dump({"model": MODEL, "rows": rows}, open("ttl-probe.json", "w"), indent=2, sort_keys=True)

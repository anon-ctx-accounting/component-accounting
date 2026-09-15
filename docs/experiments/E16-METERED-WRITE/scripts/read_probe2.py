import json, os, time, urllib.request

key = os.environ["KARC_ANTHROPIC_KEY"]
MODEL = "claude-sonnet-4-6"

def build_prefix(label, paras=60):
    out = [f"Technical note {label}: accounting for persistent context in agent runtimes.", ""]
    for i in range(paras):
        out.append(
            f"Section {i+1}. In turn {i+1} the runtime resent the accumulated transcript prefix "
            f"to the provider, so the billed input for that turn contained {1000 + 37*i} tokens that "
            f"had already been sent in an earlier turn. The accounting contract separates this "
            f"quantity into an uncached component, a cache-creation component, and a cache-read "
            f"component, because each is billed at a different rate. When the resident working set "
            f"holds {3 + (i % 7)} artifacts, the retrieval arm reads {120 + 11*i} tokens from the "
            f"document store instead of carrying them in the prefix. The ledger row for this turn "
            f"therefore records both the carried tokens and the retrieved tokens, and the review "
            f"step checks that the two components sum to the gross figure reported by the runtime."
        )
        out.append("")
    return "\n".join(out)

pfx = build_prefix("E16-READ-2")

def call(suffix, ttl):
    body = {"model": MODEL, "max_tokens": 16,
            "system": [{"type": "text", "text": pfx,
                        "cache_control": {"type": "ephemeral", "ttl": ttl}}],
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
    d = call(f"In one short sentence, what does section {n+1} say about billed input?", "1h")
    u = d["usage"]; cc = u.get("cache_creation", {})
    r = {"call": n, "stop_reason": d.get("stop_reason"),
         "text": "".join(b.get("text","") for b in d.get("content",[]))[:60],
         "geo": u.get("inference_geo"), "input": u["input_tokens"],
         "create": u["cache_creation_input_tokens"],
         "c1h": cc.get("ephemeral_1h_input_tokens"), "c5m": cc.get("ephemeral_5m_input_tokens"),
         "read": u["cache_read_input_tokens"], "output": u["output_tokens"]}
    rows.append(r)
    print(f"call {n}: stop={str(r['stop_reason']):>10s} out={r['output']:3d} input={r['input']:3d} "
          f"create={r['create']:5d} (1h={r['c1h']} 5m={r['c5m']}) read={r['read']:5d} geo={r['geo']}")
    print(f"         text={r['text']!r}")

json.dump({"model": MODEL, "prefix_chars": len(pfx), "rows": rows},
          open("read-probe2.json","w"), indent=2, sort_keys=True)

import json, os, random, urllib.request, hashlib

key = os.environ["OPENAI_API_KEY"]

def build_prefix(label, target_words=2600, seed=1313):
    rng = random.Random(seed)
    vocab = ["ledger","carry","residency","supersession","provenance","artifact","tier",
             "eviction","budget","token","prefix","transcript","invariant","closure"]
    lines = [f"# {label} synthetic prefix block"]
    for i in range(target_words // 10):
        lines.append(f"{i:04d} " + " ".join(rng.choice(vocab) for _ in range(10)))
    return "\n".join(lines)

prefix = build_prefix("E16-EXPLORE")
print("prefix chars:", len(prefix), "sha256:", hashlib.sha256(prefix.encode()).hexdigest()[:16])

body = {
    "model": "gpt-5.6-luna",
    "input": [
        {"role": "developer", "content": prefix},
        {"role": "user", "content": "Reply with the single word: acknowledged."},
    ],
    "max_output_tokens": 16,
    "store": False,
    "prompt_cache_key": "karc-e16-explore",
}
req = urllib.request.Request(
    "https://api.openai.com/v1/responses",
    data=json.dumps(body).encode(),
    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
)
try:
    with urllib.request.urlopen(req, timeout=180) as r:
        d = json.load(r)
except urllib.error.HTTPError as e:
    print("HTTP", e.code); print(e.read().decode()[:1200]); raise SystemExit(1)

print("status:", d.get("status"), "| model:", d.get("model"))
print("USAGE VERBATIM:")
print(json.dumps(d.get("usage"), indent=2, sort_keys=True))

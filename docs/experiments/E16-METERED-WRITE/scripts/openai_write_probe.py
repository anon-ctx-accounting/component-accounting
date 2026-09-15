"""E16 OpenAI leg — 계량 route가 cache_write_tokens를 실어 보내는지 측정.

prefix는 일관된 기술 산문이다(E16 §4의 거부 함정 회피). 모델은 E15
subscription 세션과 동일한 gpt-5.6-luna로, route만 바꾼 대조를 만든다.
"""
import hashlib, json, os, subprocess, time, urllib.error, urllib.request

KEY = os.environ["OPENAI_API_KEY"]
MODEL = "gpt-5.6-luna"
URL = "https://api.openai.com/v1/responses"
EXPECTED_DETAIL_KEYS = {"cached_tokens", "cache_write_tokens", "text_tokens", "audio_tokens",
                        "image_tokens", "cache_creation_tokens"}


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


def call(prefix, suffix, cache_key, store):
    body = {"model": MODEL,
            "input": [{"role": "developer", "content": prefix},
                      {"role": "user", "content": suffix}],
            "max_output_tokens": 128, "store": store}
    if cache_key:
        body["prompt_cache_key"] = cache_key
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {KEY}",
                                          "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.load(r), None
    except urllib.error.HTTPError as e:
        return None, {"http": e.code, "body": e.read().decode()[:600]}


def run(label, cache_key, store, calls=4):
    prefix = build_prefix(label)
    rows = []
    for n in range(calls):
        if n:
            time.sleep(4)
        d, err = call(prefix, f"In one short sentence, what does section {n+1} say about billed input?",
                      cache_key, store)
        if err:
            rows.append({"call": n, "error": err})
            print(f"  call {n}: ERROR {err['http']} {err['body'][:160]}")
            continue
        u = d.get("usage") or {}
        det = u.get("input_tokens_details") or {}
        cached = int(det.get("cached_tokens") or 0)
        write = int(det.get("cache_write_tokens") or det.get("cache_creation_tokens") or 0)
        total_in = int(u.get("input_tokens") or 0)
        rows.append({
            "call": n, "status": d.get("status"), "model": d.get("model"),
            "input_tokens": total_in, "cached_tokens": cached, "cache_write_tokens": write,
            "fresh_normalized": max(0, total_in - cached - write),
            "output_tokens": u.get("output_tokens"),
            "input_tokens_details_verbatim": det,
            "write_key_present": "cache_write_tokens" in det,
            "unexpected_detail_keys": sorted(set(det) - EXPECTED_DETAIL_KEYS),
        })
        r = rows[-1]
        print(f"  call {n}: status={r['status']:<10s} input={total_in:6d} cached={cached:6d} "
              f"write={write:6d} out={r['output_tokens']} "
              f"write_key_present={r['write_key_present']} keys={sorted(det)}")
    return {"label": label, "prompt_cache_key": cache_key, "store": store,
            "prefix_sha256": hashlib.sha256(prefix.encode()).hexdigest()[:16],
            "prefix_chars": len(prefix), "rows": rows}


def main():
    out = {"probe": "openai-metered-write", "model": MODEL, "endpoint": URL,
           "code_git_hash": subprocess.run(["git", "rev-parse", "HEAD"],
                                           capture_output=True, text=True).stdout.strip(),
           "runs": []}
    print("run A: store=False, prompt_cache_key set")
    out["runs"].append(run("E16-OAI-A", "karc-e16-a", False))
    json.dump(out, open("docs/experiments/E16-METERED-WRITE/raw/openai-write.json", "w"),
              indent=2, sort_keys=True)
    print("\nwrote raw/openai-write.json")


if __name__ == "__main__":
    main()

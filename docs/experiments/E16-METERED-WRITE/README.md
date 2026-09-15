# E16-METERED-WRITE

This probe measures cache-write reporting, cache retention buckets, and write-to-read transitions through direct metered API requests. It is supporting telemetry evidence for the accounting contract, without a named main configuration or paired treatment arms. See [CONFIGURATIONS.md](../../../CONFIGURATIONS.md) for the mapping.

A execution date of 2026-08-28 is recorded; a UTC timestamp is not retained in these records. Python HTTP scripts called OpenAI gpt-5.6-luna through the Responses API and Anthropic claude-sonnet-4-6 through the Messages API. The report records 9 OpenAI calls and 12 Anthropic calls. These bounded prompt probes have no S01–S12 session window and use direct requests rather than a versioned CLI harness.

raw/openai-write.json records input inclusion and cold/warm write observations. raw/ttl-probe.json records Anthropic retention-bucket observations; read-probe.json and read-probe2.json retain cache-read and refusal evidence. raw/openai-blocked.json records the preliminary access probe. There is no analysis.json in this directory.

These probe records are a telemetry probe rather than a preregistered measurement. Refusal-bearing synthetic prompts are part of the recorded history. Run A was executed again during a module import, replacing its initial cold observation with a warm observation, as recorded in provenance_notes. E20 later adds the client-fixed route comparison. These probes do not grade the paired fixture's first task.

Inspect the retained probe evidence from the repository root:

```sh
python3 -m json.tool docs/experiments/E16-METERED-WRITE/raw/openai-write.json
```

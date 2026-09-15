/**
 * E17B model-free probe extension (verbatim derivation of E17, only the output env var differs).
 *
 * Dumps pi's OWN tool registry (pi.getActiveTools() / pi.getAllTools(),
 * ExtensionAPI, pi-coding-agent 0.84.3 dist/core/extensions/types.d.ts:978-980)
 * plus system-prompt size at session_start, so that "the bridged tools are
 * registered and active in pi" can be judged without spending a model call and
 * without trusting the bridge's self-report.
 *
 * R-9: names, counts, byte sizes and sha256 only. No prompt text, no tool text.
 */
import { writeFileSync } from "node:fs";
import { createHash } from "node:crypto";

const sha256 = (value: string) => createHash("sha256").update(value, "utf8").digest("hex");

export default async function e17bToolRegistryProbe(pi: any): Promise<void> {
	pi.on("session_start", async (_event: unknown, ctx: any) => {
		const out = process.env.KARC_E17B_PROBE_OUT;
		if (!out) return;
		const record: Record<string, unknown> = { probe: "e17b-tool-registry", ok: true };
		try {
			record.active_tools = pi.getActiveTools();
			record.all_tools = (pi.getAllTools() as any[]).map((tool) => ({
				name: tool.name,
				description_chars: (tool.description ?? "").length,
				description_sha256: sha256(tool.description ?? ""),
				parameters_sha256: sha256(JSON.stringify(tool.parameters ?? null)),
				source: tool.sourceInfo?.type ?? null,
			}));
			record.commands = (ctx.getCommands?.() ?? pi.getCommands?.() ?? []).map(
				(command: any) => command.name,
			);
		} catch (error) {
			record.ok = false;
			record.error = error instanceof Error ? error.message : String(error);
		}
		try {
			const systemPrompt: string = ctx.getSystemPrompt();
			record.system_prompt_chars = systemPrompt.length;
			record.system_prompt_sha256 = sha256(systemPrompt);
		} catch (error) {
			record.system_prompt_error = error instanceof Error ? error.message : String(error);
		}
		record.model_id = ctx.model?.id ?? null;
		record.session_id = ctx.sessionManager?.getSessionId?.() ?? null;
		writeFileSync(out, JSON.stringify(record, null, 2), "utf8");
	});
}

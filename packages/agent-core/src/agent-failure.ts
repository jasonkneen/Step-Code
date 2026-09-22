/**
 * Shared construction of the synthetic assistant message used to report an
 * agent run that failed outside the normal turn loop (a rejected promise, an
 * uncaught throw). Keeps the stateful `Agent` class and the standalone
 * `agentLoop`/`agentLoopContinue` functions reporting failures the same way:
 * a failure is its own terminal outcome, never a hang.
 */
import type { Model } from "@step-harness/providers";
import type { AgentMessage } from "./types.ts";

export const EMPTY_USAGE = {
	input: 0,
	output: 0,
	cacheRead: 0,
	cacheWrite: 0,
	totalTokens: 0,
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
};

/**
 * Build an assistant message representing a run failure, mirroring
 * `Agent.handleRunFailure`'s semantics.
 */
export function createFailureMessage(model: Model<any>, aborted: boolean, error: unknown): AgentMessage {
	return {
		role: "assistant",
		content: [{ type: "text", text: "" }],
		api: model.api,
		provider: model.provider,
		model: model.id,
		usage: EMPTY_USAGE,
		stopReason: aborted ? "aborted" : "error",
		errorMessage: error instanceof Error ? error.message : String(error),
		timestamp: Date.now(),
	} satisfies AgentMessage;
}

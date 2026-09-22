/**
 * Regression: a child whose final assistant message stopped with
 * stopReason "length" (output token limit; its answer is truncated) must
 * never be reported to the parent as "completed" — see
 * packages/coding-agent/src/features/step-subagent.ts `isFailed` and
 * packages/coding-agent/src/features/subagent/execute.ts `statusForResult`.
 */
import type { AgentToolResult } from "@step-harness/agent-core";
import { expect, test } from "vitest";
import type { ExtensionAPI, ExtensionContext, ToolDefinition } from "../src/core/extensions/types.ts";
import {
	createStepSubagentExtension,
	isFailed,
	parseJsonEvent,
	resultText,
	type StepSubagentDetails,
	type StepSubagentRunResult,
} from "../src/features/step-subagent.ts";
import { statusForResult } from "../src/features/subagent/execute.ts";

function baseResult(overrides: Partial<StepSubagentRunResult> = {}): StepSubagentRunResult {
	return {
		messages: [
			{
				role: "assistant",
				content: [{ type: "text", text: "partial answer cut off mid-sen" }],
				api: "anthropic-messages",
				provider: "step",
				model: "step-3.7-flash",
				usage: {
					input: 1,
					output: 2,
					cacheRead: 0,
					cacheWrite: 0,
					totalTokens: 3,
					cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
				},
				stopReason: "length",
				timestamp: Date.now(),
			},
		],
		stderr: "",
		exitCode: 0,
		usage: { input: 1, output: 2, cacheRead: 0, cacheWrite: 0, cost: 0, contextTokens: 3, turns: 1 },
		model: "step/step-3.7-flash",
		stopReason: "length",
		...overrides,
	};
}

test("isFailed treats a length stop as a failure", () => {
	expect(isFailed({ exitCode: 0, stopReason: "length" })).toBe(true);
});

test("isFailed still treats a normal stop as success", () => {
	expect(isFailed({ exitCode: 0, stopReason: "stop" })).toBe(false);
});

test("statusForResult reports a length stop as a non-completed status", () => {
	const status = statusForResult(baseResult());
	expect(status).not.toBe("completed");
	expect(status).toBe("failed");
});

test("statusForResult still reports a normal stop as completed", () => {
	const status = statusForResult(baseResult({ stopReason: "stop" }));
	expect(status).toBe("completed");
});

test("resultText for a length stop mentions truncation and includes the partial output", () => {
	const text = resultText(baseResult());
	expect(text.toLowerCase()).toContain("truncat");
	expect(text).toContain("partial answer cut off mid-sen");
});

test("resultText for a normal stop is unaffected", () => {
	const text = resultText(baseResult({ stopReason: "stop" }));
	expect(text).toBe("partial answer cut off mid-sen");
});

test("parseJsonEvent sets a truncation errorMessage when the child provides none", () => {
	const current: StepSubagentRunResult = {
		messages: [],
		stderr: "",
		exitCode: -1,
		usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, cost: 0, contextTokens: 0, turns: 0 },
	};
	const line = JSON.stringify({
		type: "message_end",
		message: {
			role: "assistant",
			content: [{ type: "text", text: "partial answer cut off mid-sen" }],
			api: "anthropic-messages",
			provider: "step",
			model: "step-3.7-flash",
			usage: {
				input: 1,
				output: 2,
				cacheRead: 0,
				cacheWrite: 0,
				totalTokens: 3,
				cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
			},
			stopReason: "length",
			timestamp: Date.now(),
		},
	});
	parseJsonEvent(line, current, undefined);
	expect(current.stopReason).toBe("length");
	expect(current.errorMessage).toContain("truncat");
});

test("parseJsonEvent keeps the provider's own errorMessage when one is present", () => {
	const current: StepSubagentRunResult = {
		messages: [],
		stderr: "",
		exitCode: -1,
		usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, cost: 0, contextTokens: 0, turns: 0 },
	};
	const line = JSON.stringify({
		type: "message_end",
		message: {
			role: "assistant",
			content: [{ type: "text", text: "" }],
			api: "anthropic-messages",
			provider: "step",
			model: "step-3.7-flash",
			usage: {
				input: 1,
				output: 0,
				cacheRead: 0,
				cacheWrite: 0,
				totalTokens: 1,
				cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
			},
			stopReason: "length",
			errorMessage: "provider-supplied detail",
			timestamp: Date.now(),
		},
	});
	parseJsonEvent(line, current, undefined);
	expect(current.errorMessage).toBe("provider-supplied detail");
});

interface SentMessage {
	customType: string;
	content: string;
	details?: { agentId?: string; event?: string; status?: string };
	deliverAs?: string;
}

function createApi(): { api: ExtensionAPI; tools: Map<string, ToolDefinition> } {
	const tools = new Map<string, ToolDefinition>();
	const api = {
		registerTool(tool: ToolDefinition) {
			tools.set(tool.name, tool);
		},
		registerCommand: () => {},
		registerFlag: () => {},
		registerShortcut: () => {},
		on: () => {},
		getActiveTools: () => [],
		setActiveTools: () => {},
		getFlag: () => false,
		appendEntry: () => {},
		sendMessage: (_message: SentMessage) => {},
		sendUserMessage: () => {},
	} as unknown as ExtensionAPI;
	return { api, tools };
}

function createContext(cwd: string): ExtensionContext {
	return {
		mode: "tui",
		hasUI: false,
		cwd,
		model: undefined,
		thinkingLevel: "high",
		isIdle: () => true,
		isProjectTrusted: () => true,
		ui: {
			confirm: async () => true,
			setWidget: () => {},
			notify: () => {},
		},
		sessionManager: { getEntries: () => [] },
	} as unknown as ExtensionContext;
}

test("a length-stopped subagent call is reported as failed, not completed, and excluded from the success count", async () => {
	const { api, tools } = createApi();
	createStepSubagentExtension({
		includeBuiltinAgents: true,
		agentDir: "/tmp/step-agent-test",
		runner: async () => baseResult(),
	})(api);
	const subagent = tools.get("subagent");
	expect(subagent).toBeDefined();
	const result = (await subagent!.execute(
		"call",
		{ tasks: [{ agent: "general", task: "inspect" }] } as never,
		undefined,
		undefined,
		createContext("/workspace"),
	)) as AgentToolResult<StepSubagentDetails>;
	const text = result.content.find((block) => block.type === "text");
	expect(text?.type).toBe("text");
	expect(text?.type === "text" ? text.text : "").toContain("0/1 succeeded");
	expect(text?.type === "text" ? text.text.toLowerCase() : "").toContain("truncat");
	expect(text?.type === "text" ? text.text : "").toContain("partial answer cut off mid-sen");
	expect(result.details?.results[0]?.status).toBe("failed");
});

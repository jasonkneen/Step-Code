/**
 * Invariant: a subagent child inherits the parent's model, provider, reasoning
 * effort and permission policy unless something explicitly overrides them, and
 * its permission policy is never more permissive than the parent's.
 *
 * Asserted through the args/env the child is actually spawned with, and then
 * round-tripped through the child's own flag parser and policy resolver so a
 * rename on either side fails here.
 */
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { parseArgs } from "../src/cli/args.ts";
import { createEventBus } from "../src/core/event-bus.ts";
import type { ExtensionAPI, ExtensionContext, ToolDefinition } from "../src/core/extensions/types.ts";
import {
	createStepSubagentExtension,
	emptyUsage,
	runStepSubagentProcess,
	type StepSubagentRunInput,
} from "../src/features/step-subagent.ts";
import type { StepAgentConfig } from "../src/features/step-subagent-agents.ts";
import { liveSubagentSessions } from "../src/features/subagent/lane-lifecycle.ts";
import {
	buildSubagentChildArgs,
	buildSubagentChildEnv,
	subagentPermissionKey,
} from "../src/features/subagent/rpc-adapter.ts";
import {
	answerStepPermissionStateRequests,
	decideStepToolCall,
	requestStepPermissionState,
	resolveInitialStepPermissionState,
	resolveStepChildPermissionPolicy,
	type StepPermissionState,
	stepPermissionStateForPreset,
} from "../src/step/permissions.ts";

const agent: StepAgentConfig = {
	name: "general",
	description: "test agent",
	systemPrompt: "",
	source: "builtin",
};

function runInput(overrides: Partial<StepSubagentRunInput> = {}): StepSubagentRunInput {
	return { agent, task: "do the thing", cwd: process.cwd(), ...overrides };
}

function flag(args: readonly string[], name: string): string | undefined {
	const index = args.indexOf(name);
	return index === -1 ? undefined : args[index + 1];
}

/**
 * Resolve the policy the child will actually run under: parse its argv the way
 * the Step launcher does and resolve against its env. A hostile inherited env
 * (a permissive preset + autopilot) proves the flags win over it.
 */
function childPolicy(parent: StepPermissionState, parentHasUI: boolean): StepPermissionState {
	const input = runInput({ permission: resolveStepChildPermissionPolicy(parent, parentHasUI) });
	vi.stubEnv("STEP_PERMISSION_PRESET", "bypass");
	vi.stubEnv("STEP_AUTOPILOT", "1");
	const env = buildSubagentChildEnv(input);
	const parsed = parseArgs(buildSubagentChildArgs(input, "child-session"));
	return resolveInitialStepPermissionState({
		approvalMode: parsed.approvalMode,
		nonInteractiveApproval: parsed.nonInteractiveApproval,
		toolOverrides: parsed.toolOverrides,
		env,
	});
}

afterEach(() => {
	vi.unstubAllEnvs();
});

describe("subagent child permission policy", () => {
	it("parent read-only -> child read-only", () => {
		const child = childPolicy(stepPermissionStateForPreset("read-only"), true);
		expect(child.mode).toBe("strict");
		expect(child.preset).toBe("read-only");
		expect(child.nonInteractiveApproval).toBe("deny");
		expect(child.autoResume).toBe(false);
	});

	it("parent ask -> child confirms, and denies unattended mutating calls", () => {
		const child = childPolicy(stepPermissionStateForPreset("ask"), true);
		expect(child.mode).toBe("confirm");
		expect(child.nonInteractiveApproval).toBe("deny");
		// The rpc child's confirm dialogs are auto-cancelled by the parent, so a
		// confirm decision is a block, never an unattended allow.
		expect(decideStepToolCall("write_file", { path: "x" }, child).action).toBe("confirm");
	});

	it("parent explicitly chose bypass -> child bypass", () => {
		const child = childPolicy(stepPermissionStateForPreset("bypass"), true);
		expect(child.mode).toBe("auto");
		expect(child.nonInteractiveApproval).toBe("allow");
		expect(child.preset).toBe("bypass");
		expect(child.autoResume).toBe(false);
	});

	it("parent autopilot -> child autopilot", () => {
		const child = childPolicy(stepPermissionStateForPreset("autopilot"), true);
		expect(child.preset).toBe("autopilot");
		expect(child.autoResume).toBe(true);
	});

	it("a defaulted interactive parent never hands the child an unattended allow", () => {
		const parent = resolveInitialStepPermissionState({ env: {} });
		expect(parent.defaulted).toBe(true);
		const child = childPolicy(parent, true);
		expect(child.mode).toBe("auto");
		expect(child.nonInteractiveApproval).toBe("deny");
		expect(child.defaulted).toBeUndefined();
	});

	it("a headless parent passes on its downgraded effective policy", () => {
		const defaulted = resolveInitialStepPermissionState({ env: {} });
		expect(childPolicy(defaulted, false).mode).toBe("confirm");
		const refused = resolveInitialStepPermissionState({
			approvalMode: "auto",
			nonInteractiveApproval: "deny",
			env: {},
		});
		expect(childPolicy(refused, false).mode).toBe("confirm");
		// An explicit unattended allow stays an allow.
		expect(childPolicy(stepPermissionStateForPreset("bypass"), false).mode).toBe("auto");
	});

	it("tool overrides reach the child", () => {
		const parent: StepPermissionState = {
			...stepPermissionStateForPreset("bypass"),
			toolOverrides: { run_command: "deny", write_file: "confirm" },
		};
		expect(childPolicy(parent, true).toolOverrides).toEqual({ run_command: "deny", write_file: "confirm" });
	});

	it("no parent policy -> no flags (embedders without the Step extension)", () => {
		const args = buildSubagentChildArgs(runInput(), "child-session");
		expect(args).not.toContain("--approval-mode");
		expect(args).not.toContain("--non-interactive-approval");
	});

	it("the subagent tool reads the parent's live state over the extension event bus", () => {
		const events = createEventBus();
		expect(requestStepPermissionState(events)).toBeUndefined();
		let state = stepPermissionStateForPreset("bypass");
		answerStepPermissionStateRequests(events, () => state);
		expect(requestStepPermissionState(events)?.preset).toBe("bypass");
		state = stepPermissionStateForPreset("read-only");
		expect(requestStepPermissionState(events)?.preset).toBe("read-only");
	});
});

describe("subagent child model and thinking", () => {
	it("inherits the parent's qualified model and thinking level", () => {
		const args = buildSubagentChildArgs(
			runInput({ model: "stepfun/step-3", thinkingLevel: "high" }),
			"child-session",
		);
		expect(flag(args, "--model")).toBe("stepfun/step-3");
		expect(flag(args, "--thinking")).toBe("high");
	});

	it("agent file sets a model but no thinking -> child gets the parent's thinking", () => {
		const args = buildSubagentChildArgs(
			runInput({
				agent: { ...agent, model: "openai/gpt-5" },
				model: "stepfun/step-3",
				thinkingLevel: "high",
			}),
			"child-session",
		);
		expect(flag(args, "--model")).toBe("openai/gpt-5");
		expect(flag(args, "--thinking")).toBe("high");
	});

	it("agent declares thinking via its model suffix -> that wins", () => {
		const args = buildSubagentChildArgs(
			runInput({
				agent: { ...agent, model: "openai/gpt-5:low" },
				model: "stepfun/step-3",
				thinkingLevel: "high",
			}),
			"child-session",
		);
		expect(flag(args, "--model")).toBe("openai/gpt-5:low");
		// `--thinking` would override the suffix in the child's resolver.
		expect(args).not.toContain("--thinking");
		expect(parseArgs(args).thinking).toBeUndefined();
	});

	it("a non-thinking colon suffix is part of the model id, not a declaration", () => {
		const args = buildSubagentChildArgs(
			runInput({ agent: { ...agent, model: "ollama/llama3:8b" }, thinkingLevel: "medium" }),
			"child-session",
		);
		expect(flag(args, "--thinking")).toBe("medium");
	});
});

describe("subagent tool wiring", () => {
	it("hands the runner the parent's live policy, qualified model and thinking level", async () => {
		const events = createEventBus();
		let parentState = stepPermissionStateForPreset("ask");
		answerStepPermissionStateRequests(events, () => parentState);
		const tools = new Map<string, ToolDefinition>();
		const api = {
			events,
			registerTool: (tool: ToolDefinition) => tools.set(tool.name, tool),
			registerCommand: () => {},
			registerFlag: () => {},
			registerShortcut: () => {},
			on: () => {},
			getFlag: () => false,
		} as unknown as ExtensionAPI;
		const inputs: StepSubagentRunInput[] = [];
		createStepSubagentExtension({
			agentDir: "/tmp/step-agent-inherit-test",
			runner: async (input) => {
				inputs.push(input);
				return { messages: [], stderr: "", exitCode: 0, usage: emptyUsage() };
			},
		})(api);
		const ctx = {
			mode: "tui",
			hasUI: true,
			cwd: process.cwd(),
			model: { provider: "stepfun", id: "step-3" },
			thinkingLevel: "high",
			isProjectTrusted: () => true,
			ui: { setWidget: () => {}, setStatus: () => {}, notify: () => {} },
		} as unknown as ExtensionContext;
		const execute = (): Promise<unknown> =>
			tools.get("subagent")!.execute("call", { agent: "general", task: "go" }, undefined, undefined, ctx);

		await execute();
		expect(inputs[0]?.model).toBe("stepfun/step-3");
		expect(inputs[0]?.thinkingLevel).toBe("high");
		expect(inputs[0]?.permission).toEqual({
			approvalMode: "confirm",
			nonInteractiveApproval: "deny",
			autoResume: false,
		});

		// A preset switched mid-session reaches the next child.
		parentState = stepPermissionStateForPreset("read-only");
		await execute();
		expect(inputs[1]?.permission?.approvalMode).toBe("strict");
	});
});

describe("keep-alive subagent lanes", () => {
	it("replace an idle child whose spawn-time policy no longer matches the parent's", async () => {
		const dir = await mkdtemp(path.join(tmpdir(), "step-subagent-inherit-"));
		// Stand-in for `step --mode rpc`: `currentStepInvocation` reuses argv[1].
		const script = path.join(dir, "fake-child.mjs");
		await writeFile(
			script,
			`let buffer = "";
process.stdin.on("data", (chunk) => {
	buffer += chunk.toString();
	const lines = buffer.split("\\n");
	buffer = lines.pop() ?? "";
	for (const line of lines) {
		if (!line.trim()) continue;
		const command = JSON.parse(line);
		if (command.type !== "prompt") continue;
		process.stdout.write(JSON.stringify({ type: "response", id: command.id, command: "prompt", success: true }) + "\\n");
		process.stdout.write(JSON.stringify({ type: "agent_settled" }) + "\\n");
	}
});
process.stdin.on("end", () => process.exit(0));
`,
			"utf8",
		);
		const originalArgv1 = process.argv[1];
		process.argv[1] = script;
		const sessionId = `inherit-keepalive-${Date.now()}`;
		try {
			const bypass = resolveStepChildPermissionPolicy(stepPermissionStateForPreset("bypass"), true);
			const readOnly = resolveStepChildPermissionPolicy(stepPermissionStateForPreset("read-only"), true);
			const respawns: string[] = [];
			const turn = (permission: typeof bypass, label: string) =>
				runStepSubagentProcess(
					runInput({ sessionId, keepAlive: true, permission, onChildRespawn: () => respawns.push(label) }),
				);

			await turn(bypass, "first");
			const firstChild = liveSubagentSessions.get(sessionId);
			expect(firstChild?.permissionKey).toBe(subagentPermissionKey(bypass));
			await turn(bypass, "same policy");
			expect(liveSubagentSessions.get(sessionId)).toBe(firstChild);
			expect(respawns).toEqual([]);

			await turn(readOnly, "tightened");
			expect(respawns).toEqual(["tightened"]);
			const replaced = liveSubagentSessions.get(sessionId);
			expect(replaced).not.toBe(firstChild);
			expect(replaced?.permissionKey).toBe(subagentPermissionKey(readOnly));
			replaced?.stop();
		} finally {
			process.argv[1] = originalArgv1;
			liveSubagentSessions.clear();
			await rm(dir, { recursive: true, force: true });
		}
	}, 20_000);
});

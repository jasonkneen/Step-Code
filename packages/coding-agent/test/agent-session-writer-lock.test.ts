import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { hostname, tmpdir } from "node:os";
import { join } from "node:path";
import { Agent } from "@step-harness/agent-core";
import type { AssistantMessage } from "@step-harness/providers/compat";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AgentSession } from "../src/core/agent-session.ts";
import { AuthStorage } from "../src/core/auth-storage.ts";
import { SessionManager } from "../src/core/session-manager.ts";
import { SettingsManager } from "../src/core/settings-manager.ts";
import { createModelRegistry, getModelRuntime } from "./model-runtime-test-utils.ts";
import { createTestResourceLoader, stepModel } from "./utilities.ts";

function assistantMessage(text: string): AssistantMessage {
	return {
		role: "assistant",
		content: [{ type: "text", text }],
		api: "anthropic-messages",
		provider: "anthropic",
		model: "mock",
		usage: {
			input: 0,
			output: 0,
			cacheRead: 0,
			cacheWrite: 0,
			totalTokens: 0,
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
		},
		stopReason: "stop",
		timestamp: Date.now(),
	};
}

describe("AgentSession with a session file locked by another process", () => {
	let tempDir: string;
	let session: AgentSession | undefined;

	beforeEach(() => {
		tempDir = mkdtempSync(join(tmpdir(), "agent-session-writer-lock-"));
	});

	afterEach(async () => {
		session?.dispose();
		session = undefined;
		await new Promise((resolve) => setTimeout(resolve, 0));
		rmSync(tempDir, { recursive: true, force: true });
	});

	it("rejects prompt() with the lock error before calling the model", async () => {
		const seed = SessionManager.create(tempDir, tempDir);
		seed.appendMessage({ role: "user", content: "hi", timestamp: Date.now() });
		seed.appendMessage(assistantMessage("hello"));
		const file = seed.getSessionFile()!;
		seed.dispose();
		writeFileSync(
			`${file}.lock`,
			JSON.stringify({ pid: process.ppid, hostname: hostname(), startedAt: new Date().toISOString() }),
		);
		const before = readFileSync(file, "utf8");

		let modelCalls = 0;
		const agent = new Agent({
			getApiKey: () => "test-key",
			initialState: { model: stepModel(), systemPrompt: "Test", tools: [] },
			streamFn: () => {
				modelCalls++;
				throw new Error("model must not be called");
			},
		});
		const authStorage = AuthStorage.create(join(tempDir, "auth.json"));
		await authStorage.modify("step", async () => ({ type: "api_key", key: "test-key" }));
		session = new AgentSession({
			agent,
			sessionManager: SessionManager.open(file, tempDir),
			settingsManager: SettingsManager.create(tempDir, tempDir),
			cwd: tempDir,
			modelRuntime: getModelRuntime(await createModelRegistry(authStorage, tempDir)),
			resourceLoader: createTestResourceLoader(),
		});

		await expect(session.prompt("resume here")).rejects.toThrow(
			`is already open in another process (pid ${process.ppid} on host ${hostname()})`,
		);
		expect(modelCalls).toBe(0);
		expect(readFileSync(file, "utf8")).toBe(before);
	});
});

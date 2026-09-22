import { mkdirSync, readdirSync, readFileSync, rmSync, writeFileSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

// _rewriteFile() (session-manager.ts) truncates the live session file and then
// writes entries one at a time. If a write fails partway through (crash,
// exception, ENOSPC) the file used to end up truncated or empty, silently
// losing the session. It must instead write a sibling temp file and rename it
// over the destination, so a failure mid-write leaves the original untouched.
const controls = vi.hoisted(() => ({
	failAfterCalls: null as number | null,
	callCount: 0,
}));

vi.mock("fs", async (importOriginal) => {
	const actual = await importOriginal<typeof import("fs")>();
	return {
		...actual,
		writeFileSync: (...args: Parameters<typeof actual.writeFileSync>) => {
			controls.callCount++;
			if (controls.failAfterCalls !== null && controls.callCount > controls.failAfterCalls) {
				throw new Error("simulated ENOSPC mid-rewrite");
			}
			return actual.writeFileSync(...args);
		},
	};
});

const { SessionManager } = await import("../../src/core/session-manager.ts");

describe("SessionManager._rewriteFile atomicity", () => {
	let tempDir: string;

	beforeEach(() => {
		tempDir = join(tmpdir(), `session-rewrite-test-${Date.now()}-${Math.random().toString(36).slice(2)}`);
		mkdirSync(tempDir, { recursive: true });
		controls.failAfterCalls = null;
		controls.callCount = 0;
	});

	afterEach(() => {
		vi.restoreAllMocks();
		rmSync(tempDir, { recursive: true, force: true });
	});

	function writeV1SessionFile(file: string): string {
		// A v1-style session (no per-entry id/parentId) forces migrateToCurrentVersion()
		// to report a change, which triggers _rewriteFile() on open.
		const content =
			`${JSON.stringify({ type: "session", id: "sess-1", timestamp: "2025-01-01T00:00:00Z", cwd: "/tmp" })}\n` +
			`${JSON.stringify({ type: "message", timestamp: "2025-01-01T00:00:01Z", message: { role: "user", content: "hi", timestamp: 1 } })}\n` +
			`${JSON.stringify({
				type: "message",
				timestamp: "2025-01-01T00:00:02Z",
				message: {
					role: "assistant",
					content: [{ type: "text", text: "hello" }],
					api: "test",
					provider: "test",
					model: "test",
					usage: { input: 1, output: 1, cacheRead: 0, cacheWrite: 0 },
					stopReason: "stop",
					timestamp: 2,
				},
			})}\n`;
		writeFileSync(file, content);
		return content;
	}

	it("leaves the original file byte-identical and no temp file behind when a write fails mid-rewrite", () => {
		const file = join(tempDir, "session.jsonl");
		const originalContent = writeV1SessionFile(file);

		// Let the first entry write (the header) to the temp file succeed, then fail on
		// the second so the rewrite is caught mid-way through populating the sibling
		// temp file, after content has actually been written to it.
		controls.callCount = 0;
		controls.failAfterCalls = 1;

		expect(() => SessionManager.open(file, tempDir)).toThrow("simulated ENOSPC mid-rewrite");

		// The original file must be untouched.
		expect(readFileSync(file, "utf8")).toBe(originalContent);

		// No leftover temp file in the session directory.
		const leftovers = readdirSync(tempDir).filter((name) => name !== "session.jsonl");
		expect(leftovers).toEqual([]);
	});

	it("still produces the same migrated content on the happy path", () => {
		const file = join(tempDir, "session.jsonl");
		writeV1SessionFile(file);

		const sm = SessionManager.open(file, tempDir);

		const lines = readFileSync(file, "utf8").trim().split("\n");
		expect(lines).toHaveLength(3);
		const header = JSON.parse(lines[0]);
		expect(header.version).toBe(3);
		expect(sm.getSessionId()).toBe("sess-1");

		// No leftover temp files.
		const leftovers = readdirSync(tempDir).filter((name) => name !== "session.jsonl");
		expect(leftovers).toEqual([]);
	});
});

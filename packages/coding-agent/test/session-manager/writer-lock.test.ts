import { spawnSync } from "child_process";
import { existsSync, mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from "fs";
import { hostname, tmpdir } from "os";
import { join } from "path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { SessionManager } from "../../src/core/session-manager.ts";

function userMessage(text: string) {
	return { role: "user" as const, content: text, timestamp: Date.now() };
}

function assistantMessage(text: string) {
	return {
		role: "assistant" as const,
		content: [{ type: "text" as const, text }],
		api: "anthropic-messages" as const,
		provider: "anthropic",
		model: "test",
		usage: {
			input: 0,
			output: 0,
			cacheRead: 0,
			cacheWrite: 0,
			totalTokens: 0,
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
		},
		stopReason: "stop" as const,
		timestamp: Date.now(),
	};
}

function writeForeignLock(sessionFile: string, pid: number, host = hostname()): void {
	writeFileSync(`${sessionFile}.lock`, JSON.stringify({ pid, hostname: host, startedAt: new Date().toISOString() }));
}

/** A pid that has certainly exited: the child of a completed spawnSync. */
function deadPid(): number {
	const result = spawnSync(process.execPath, ["-e", ""]);
	return result.pid;
}

describe("SessionManager writer lock", () => {
	let dir: string;

	beforeEach(() => {
		dir = mkdtempSync(join(tmpdir(), "session-writer-lock-"));
	});

	afterEach(() => {
		rmSync(dir, { recursive: true, force: true });
	});

	/** Create a flushed session file on disk and release its lock. */
	function createFlushedSession(): string {
		const manager = SessionManager.create(dir, dir);
		manager.appendMessage(userMessage("hi"));
		manager.appendMessage(assistantMessage("hello"));
		const file = manager.getSessionFile()!;
		expect(existsSync(file)).toBe(true);
		manager.dispose();
		return file;
	}

	it("holds a lock file while a persisting manager owns the session file", () => {
		const manager = SessionManager.create(dir, dir);
		manager.appendMessage(userMessage("hi"));
		manager.appendMessage(assistantMessage("hello"));
		const lockPath = `${manager.getSessionFile()}.lock`;
		expect(existsSync(lockPath)).toBe(true);
		const lock = JSON.parse(readFileSync(lockPath, "utf8"));
		expect(lock.pid).toBe(process.pid);
		expect(lock.hostname).toBe(hostname());
		expect(typeof lock.startedAt).toBe("string");
		manager.dispose();
		expect(existsSync(lockPath)).toBe(false);
	});

	it("refuses to append when another live process holds the lock", () => {
		const file = createFlushedSession();
		writeForeignLock(file, process.ppid);
		const before = readFileSync(file, "utf8");

		const second = SessionManager.open(file, dir);
		expect(() => second.appendMessage(userMessage("again"))).toThrow(
			`Session ${file} is already open in another process (pid ${process.ppid} on host ${hostname()})`,
		);
		expect(readFileSync(file, "utf8")).toBe(before);
		// The foreign lock is left intact.
		expect(JSON.parse(readFileSync(`${file}.lock`, "utf8")).pid).toBe(process.ppid);
	});

	it("refuses to write when the lock belongs to another host", () => {
		const file = createFlushedSession();
		writeForeignLock(file, process.pid, "some-other-host.invalid");
		const second = SessionManager.open(file, dir);
		expect(() => second.appendMessage(userMessage("again"))).toThrow(/on host some-other-host\.invalid/);
	});

	it("reclaims a stale lock left by a dead process on this host", () => {
		const file = createFlushedSession();
		const stalePid = deadPid();
		writeForeignLock(file, stalePid);

		const manager = SessionManager.open(file, dir);
		manager.appendMessage(userMessage("resumed"));
		expect(JSON.parse(readFileSync(`${file}.lock`, "utf8")).pid).toBe(process.pid);
		expect(readFileSync(file, "utf8")).toContain("resumed");
		manager.dispose();
	});

	it("releases the lock on session switch so the file can be reopened", () => {
		const file = createFlushedSession();
		const first = SessionManager.open(file, dir);
		first.appendMessage(userMessage("one"));
		expect(existsSync(`${file}.lock`)).toBe(true);

		first.newSession();
		expect(existsSync(`${file}.lock`)).toBe(false);

		// Pretend the other process now owns it, then verify release on setSessionFile too.
		const second = SessionManager.open(file, dir);
		second.appendMessage(userMessage("two"));
		expect(existsSync(`${file}.lock`)).toBe(true);
		const other = createFlushedSession();
		second.setSessionFile(other);
		expect(existsSync(`${file}.lock`)).toBe(false);
		second.dispose();
	});

	it("releases the lock when branching into a new session file", () => {
		const file = createFlushedSession();
		const manager = SessionManager.open(file, dir);
		manager.appendMessage(userMessage("more"));
		expect(existsSync(`${file}.lock`)).toBe(true);
		const leaf = manager.getLeafId()!;
		const branched = manager.createBranchedSession(leaf)!;
		expect(existsSync(`${file}.lock`)).toBe(false);
		expect(existsSync(`${branched}.lock`)).toBe(true);
		manager.dispose();
		expect(existsSync(`${branched}.lock`)).toBe(false);
	});

	it("shares the lock between managers in the same process", () => {
		const file = createFlushedSession();
		const active = SessionManager.open(file, dir);
		active.appendMessage(userMessage("active"));
		// A transient manager (e.g. renaming the active session) writes and disposes.
		const transient = SessionManager.open(file, dir);
		transient.appendSessionInfo("renamed");
		transient.dispose();
		// The active manager still owns the file.
		expect(existsSync(`${file}.lock`)).toBe(true);
		active.appendMessage(userMessage("still active"));
		active.dispose();
		expect(existsSync(`${file}.lock`)).toBe(false);
	});

	it("does not lock sessions that are only opened for reading", () => {
		const file = createFlushedSession();
		const reader = SessionManager.open(file, dir);
		reader.getEntries();
		reader.buildSessionContext();
		expect(existsSync(`${file}.lock`)).toBe(false);
	});

	it("creates no lock files for in-memory sessions", () => {
		const manager = SessionManager.inMemory(dir);
		manager.appendMessage(userMessage("hi"));
		manager.appendMessage(assistantMessage("hello"));
		expect(readdirSync(dir).filter((name) => name.endsWith(".lock"))).toEqual([]);
	});

	it("ignores lock files when listing sessions", async () => {
		const file = createFlushedSession();
		writeForeignLock(file, process.ppid);
		const sessions = await SessionManager.list(dir, dir);
		expect(sessions.map((session) => session.path)).toEqual([file]);
		const all = await SessionManager.listAll(dir);
		expect(all.map((session) => session.path)).toEqual([file]);
	});
});

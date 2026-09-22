import { closeSync, openSync, readFileSync, unlinkSync, writeSync } from "fs";
import { hostname } from "os";

/**
 * Advisory single-writer lock for a session JSONL file.
 *
 * A persisting SessionManager holds `<sessionFile>.lock` for as long as it owns
 * the file, so two processes resuming the same session cannot interleave their
 * appends. The lock records the owner's pid and host; a lock whose pid is no
 * longer alive on this host is stale and reclaimed. Locks from another host
 * cannot be verified and are always treated as held.
 */
interface SessionLockOwner {
	pid: number;
	hostname: string;
	startedAt: string;
}

/** Locks held by this process, reference-counted across SessionManager instances. */
const heldLocks = new Map<string, number>();
let exitHandlerInstalled = false;

function releaseAllOnExit(): void {
	for (const lockPath of heldLocks.keys()) {
		unlinkIfOurs(lockPath);
	}
	heldLocks.clear();
}

/** Remove a lock file unless it has since been taken over by another process. */
function unlinkIfOurs(lockPath: string): void {
	const owner = readOwner(lockPath);
	if (owner && (owner.pid !== process.pid || owner.hostname !== hostname())) return;
	try {
		unlinkSync(lockPath);
	} catch {
		// Already gone.
	}
}

function readOwner(lockPath: string): SessionLockOwner | undefined {
	try {
		const parsed = JSON.parse(readFileSync(lockPath, "utf8")) as Partial<SessionLockOwner>;
		if (typeof parsed.pid !== "number" || typeof parsed.hostname !== "string") return undefined;
		return { pid: parsed.pid, hostname: parsed.hostname, startedAt: String(parsed.startedAt ?? "") };
	} catch {
		return undefined;
	}
}

function isProcessAlive(pid: number): boolean {
	try {
		process.kill(pid, 0);
		return true;
	} catch (error) {
		// EPERM means the process exists but belongs to another user.
		return (error as NodeJS.ErrnoException).code !== "ESRCH";
	}
}

/** True when the recorded owner is this process or a dead process on this host. */
function isReclaimable(owner: SessionLockOwner): boolean {
	if (owner.hostname !== hostname()) return false;
	return owner.pid === process.pid || !isProcessAlive(owner.pid);
}

function tryCreate(lockPath: string): boolean {
	let fd: number;
	try {
		fd = openSync(lockPath, "wx", 0o600);
	} catch (error) {
		if ((error as NodeJS.ErrnoException).code === "EEXIST") return false;
		throw error;
	}
	try {
		const owner: SessionLockOwner = { pid: process.pid, hostname: hostname(), startedAt: new Date().toISOString() };
		writeSync(fd, JSON.stringify(owner));
	} finally {
		closeSync(fd);
	}
	return true;
}

/**
 * Acquire the writer lock for a session file. Returns the lock path.
 * Throws when a live process (or any process on another host) owns it.
 */
export function acquireSessionWriterLock(sessionFile: string): string {
	const lockPath = `${sessionFile}.lock`;
	const held = heldLocks.get(lockPath);
	if (held !== undefined) {
		// Another manager in this process already owns the file (e.g. a transient
		// manager renaming the active session); share the lock instead of racing it.
		heldLocks.set(lockPath, held + 1);
		return lockPath;
	}
	if (!tryCreate(lockPath)) {
		const owner = readOwner(lockPath);
		if (!owner) {
			throw new Error(
				`Session ${sessionFile} is locked by an unreadable lock file. If no other process is using this session, delete ${lockPath} and retry.`,
			);
		}
		if (!isReclaimable(owner)) {
			throw new Error(
				`Session ${sessionFile} is already open in another process (pid ${owner.pid} on host ${owner.hostname}). Close it or use --fork to continue in a new session. If that process is gone, delete ${lockPath}.`,
			);
		}
		try {
			unlinkSync(lockPath);
		} catch {
			// Another process may have reclaimed it first; the create below decides.
		}
		if (!tryCreate(lockPath)) {
			const current = readOwner(lockPath);
			throw new Error(
				`Session ${sessionFile} is already open in another process${current ? ` (pid ${current.pid} on host ${current.hostname})` : ""}. Close it or use --fork to continue in a new session.`,
			);
		}
	}
	heldLocks.set(lockPath, 1);
	if (!exitHandlerInstalled) {
		exitHandlerInstalled = true;
		process.once("exit", releaseAllOnExit);
	}
	return lockPath;
}

/** Release a lock previously returned by acquireSessionWriterLock, if we still own it. */
export function releaseSessionWriterLock(lockPath: string): void {
	const held = heldLocks.get(lockPath);
	if (held === undefined) return;
	if (held > 1) {
		heldLocks.set(lockPath, held - 1);
		return;
	}
	heldLocks.delete(lockPath);
	unlinkIfOurs(lockPath);
}

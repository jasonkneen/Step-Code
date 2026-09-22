import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { createProvider } from "../src/models.ts";
import type { Api, AssistantMessage, Context, Model, StreamOptions } from "../src/types.ts";
import { AssistantMessageEventStream } from "../src/utils/event-stream.ts";
import { isRetryableAssistantError } from "../src/utils/retry.ts";

const model: Model<Api> = {
	id: "idle-model",
	name: "idle-model",
	api: "idle-api",
	provider: "idle-provider",
	baseUrl: "https://example.test/v1",
	reasoning: false,
	input: ["text"],
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
	contextWindow: 10000,
	maxTokens: 1000,
};

const context: Context = { messages: [{ role: "user", content: "hi", timestamp: 0 }] };

function message(stopReason: AssistantMessage["stopReason"], text = ""): AssistantMessage {
	return {
		role: "assistant",
		content: text ? [{ type: "text", text }] : [],
		api: model.api,
		provider: model.provider,
		model: model.id,
		usage: {
			input: 0,
			output: 0,
			cacheRead: 0,
			cacheWrite: 0,
			totalTokens: 0,
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
		},
		stopReason,
		timestamp: Date.now(),
	};
}

/**
 * A wire API that behaves like an adapter whose upstream only sends keepalive
 * pings: pings are filtered below the event layer, so after `start` (and any
 * scripted deltas) nothing is emitted until the request signal aborts.
 */
function scriptedProvider(script: (stream: AssistantMessageEventStream, signal: AbortSignal | undefined) => void) {
	const seen: { signal?: AbortSignal; streamIdleTimeoutMs?: number }[] = [];
	const provider = createProvider({
		id: model.provider,
		auth: { apiKey: { name: "key", resolve: async () => ({ auth: { apiKey: "k" } }) } } as never,
		models: [model],
		api: {
			stream: (_model, _context, options?: StreamOptions) => {
				seen.push({ signal: options?.signal, streamIdleTimeoutMs: options?.streamIdleTimeoutMs });
				const stream = new AssistantMessageEventStream();
				const signal = options?.signal;
				signal?.addEventListener("abort", () => {
					const aborted = message("aborted");
					stream.push({ type: "error", reason: "aborted", error: aborted });
					stream.end(aborted);
				});
				script(stream, signal);
				return stream;
			},
			streamSimple: () => {
				throw new Error("unused");
			},
		},
	});
	return { provider, seen };
}

describe("stream content-idle watchdog", () => {
	beforeEach(() => {
		vi.useFakeTimers();
		vi.setSystemTime(new Date("2026-01-01T00:00:00.000Z"));
	});
	afterEach(() => {
		vi.useRealTimers();
	});

	it("fails with stopReason error naming the phase when only keepalives follow start", async () => {
		const { provider } = scriptedProvider((stream) => {
			stream.push({ type: "start", partial: message("stop") });
		});
		const stream = provider.stream(model, context, { streamIdleTimeoutMs: 1000 });
		const events: string[] = [];
		const consume = (async () => {
			for await (const event of stream) events.push(event.type);
		})();

		await vi.advanceTimersByTimeAsync(999);
		expect(events).toEqual(["start"]);
		await vi.advanceTimersByTimeAsync(1);
		await consume;

		const result = await stream.result();
		expect(events).toEqual(["start", "error"]);
		expect(result.stopReason).toBe("error");
		expect(result.errorMessage).toBe(
			"Stream idle timeout: no content for 1000ms (phase: waiting for first token, last content at 2026-01-01T00:00:00.000Z)",
		);
		expect(isRetryableAssistantError(result)).toBe(true);
	});

	it("names the last streamed event and keeps partial content when a stream stalls mid-generation", async () => {
		const { provider } = scriptedProvider((stream) => {
			const partial = message("stop", "hel");
			stream.push({ type: "start", partial: message("stop") });
			setTimeout(() => stream.push({ type: "text_start", contentIndex: 0, partial }), 500);
			setTimeout(() => stream.push({ type: "text_delta", contentIndex: 0, delta: "hel", partial }), 900);
		});
		const stream = provider.stream(model, context, { streamIdleTimeoutMs: 1000 });

		await vi.advanceTimersByTimeAsync(1899);
		let settled = false;
		void stream.result().then(() => {
			settled = true;
		});
		await vi.advanceTimersByTimeAsync(0);
		expect(settled).toBe(false);
		await vi.advanceTimersByTimeAsync(1);

		const result = await stream.result();
		expect(result.stopReason).toBe("error");
		expect(result.content).toEqual([{ type: "text", text: "hel" }]);
		expect(result.errorMessage).toBe(
			"Stream idle timeout: no content for 1000ms (phase: streaming text_delta, last content at 2026-01-01T00:00:00.900Z)",
		);
	});

	it("aborts the underlying request when the watchdog fires", async () => {
		const { provider, seen } = scriptedProvider(() => {});
		const stream = provider.stream(model, context, { streamIdleTimeoutMs: 1000 });
		await vi.advanceTimersByTimeAsync(1000);

		const result = await stream.result();
		expect(result.stopReason).toBe("error");
		expect(result.errorMessage).toBe(
			"Stream idle timeout: no content for 1000ms (phase: waiting for first token, request started at 2026-01-01T00:00:00.000Z)",
		);
		expect(seen[0].signal?.aborted).toBe(true);
		// The option is consumed by the outermost guard so nested dispatch layers do not double-guard.
		expect(seen[0].streamIdleTimeoutMs).toBeUndefined();
	});

	it("never trips while content arrives more often than the timeout", async () => {
		const { provider } = scriptedProvider((stream) => {
			const partial = message("stop", "x");
			stream.push({ type: "start", partial: message("stop") });
			stream.push({ type: "text_start", contentIndex: 0, partial });
			for (let i = 1; i <= 10; i++) {
				setTimeout(() => stream.push({ type: "text_delta", contentIndex: 0, delta: "x", partial }), i * 800);
			}
			setTimeout(() => {
				const done = message("stop", "x");
				stream.push({ type: "done", reason: "stop", message: done });
				stream.end(done);
			}, 8500);
		});
		const stream = provider.stream(model, context, { streamIdleTimeoutMs: 1000 });

		await vi.advanceTimersByTimeAsync(8500);
		const result = await stream.result();
		expect(result.stopReason).toBe("stop");
		expect(vi.getTimerCount()).toBe(0);
	});

	it("still reports a caller abort as aborted", async () => {
		const { provider } = scriptedProvider((stream) => {
			stream.push({ type: "start", partial: message("stop") });
		});
		const controller = new AbortController();
		const stream = provider.stream(model, context, { streamIdleTimeoutMs: 1000, signal: controller.signal });

		await vi.advanceTimersByTimeAsync(500);
		controller.abort();
		await vi.advanceTimersByTimeAsync(5000);

		const result = await stream.result();
		expect(result.stopReason).toBe("aborted");
		expect(vi.getTimerCount()).toBe(0);
	});

	it.each([0, undefined])("installs no watchdog when streamIdleTimeoutMs is %s", async (streamIdleTimeoutMs) => {
		const { provider, seen } = scriptedProvider((stream) => {
			stream.push({ type: "start", partial: message("stop") });
		});
		const stream = provider.stream(model, context, { streamIdleTimeoutMs });
		let settled = false;
		void stream.result().then(() => {
			settled = true;
		});

		await vi.advanceTimersByTimeAsync(10 * 60_000);
		expect(settled).toBe(false);
		expect(vi.getTimerCount()).toBe(0);
		expect(seen[0].signal).toBeUndefined();
	});
});

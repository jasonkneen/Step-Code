import type { Api, AssistantMessage, AssistantMessageEvent, Model, StreamOptions } from "../types.ts";
import { AssistantMessageEventStream } from "./event-stream.ts";

function isTerminal(event: AssistantMessageEvent): event is Extract<AssistantMessageEvent, { type: "done" | "error" }> {
	return event.type === "done" || event.type === "error";
}

function createIdleErrorMessage(
	model: Model<Api>,
	partial: AssistantMessage | undefined,
	errorMessage: string,
): AssistantMessage {
	return {
		...(partial ?? {
			role: "assistant",
			content: [],
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
			timestamp: Date.now(),
		}),
		stopReason: "error",
		errorMessage,
	};
}

/**
 * Guards a provider stream with a content-idle watchdog when
 * `options.streamIdleTimeoutMs` is > 0. The timer starts when the request is
 * issued and resets on every emitted stream event. Adapters only emit content
 * events (SSE pings/comment keepalives are filtered below this layer), so a
 * stalled generation that keeps pinging still times out, unlike transport body
 * timeouts that reset on any byte.
 *
 * On expiry the request is aborted and the stream terminates with
 * `stopReason: "error"` (not "aborted": the caller did not abort) and a message
 * naming the phase and last activity. The message matches
 * `isRetryableAssistantError`. Caller aborts are forwarded unchanged.
 *
 * The option is consumed here, so nested dispatch layers do not re-guard.
 */
export function withStreamIdleTimeout<TOptions extends StreamOptions>(
	model: Model<Api>,
	options: TOptions | undefined,
	start: (options: TOptions | undefined) => AssistantMessageEventStream,
): AssistantMessageEventStream {
	const timeoutMs = options?.streamIdleTimeoutMs;
	if (!options || timeoutMs === undefined || !Number.isFinite(timeoutMs) || timeoutMs <= 0) {
		return start(options);
	}

	const { streamIdleTimeoutMs: _consumed, ...rest } = options;
	const callerSignal = options.signal;
	const controller = new AbortController();
	const outer = new AssistantMessageEventStream();
	const requestStartedAt = Date.now();
	let timer: ReturnType<typeof setTimeout> | undefined;
	let finished = false;
	let lastPartial: AssistantMessage | undefined;
	let lastEventType: AssistantMessageEvent["type"] | undefined;
	let lastEventAt = requestStartedAt;

	const onCallerAbort = () => {
		clearTimeout(timer);
		controller.abort(callerSignal?.reason);
	};
	const finish = () => {
		finished = true;
		clearTimeout(timer);
		callerSignal?.removeEventListener("abort", onCallerAbort);
	};
	const onIdle = () => {
		if (finished || callerSignal?.aborted) return;
		const phase =
			lastEventType === undefined || lastEventType === "start"
				? "waiting for first token"
				: `streaming ${lastEventType}`;
		const activity =
			lastEventType === undefined
				? `request started at ${new Date(requestStartedAt).toISOString()}`
				: `last content at ${new Date(lastEventAt).toISOString()}`;
		const errorMessage = `Stream idle timeout: no content for ${timeoutMs}ms (phase: ${phase}, ${activity})`;
		const error = createIdleErrorMessage(model, lastPartial, errorMessage);
		finish();
		controller.abort(new Error(errorMessage));
		outer.push({ type: "error", reason: "error", error });
		outer.end(error);
	};
	const arm = () => {
		clearTimeout(timer);
		if (finished || callerSignal?.aborted) return;
		timer = setTimeout(onIdle, timeoutMs);
	};

	if (callerSignal?.aborted) controller.abort(callerSignal.reason);
	else callerSignal?.addEventListener("abort", onCallerAbort, { once: true });

	arm();
	let inner: AssistantMessageEventStream;
	try {
		inner = start({ ...rest, signal: controller.signal } as TOptions);
	} catch (error) {
		finish();
		throw error;
	}

	void (async () => {
		let terminal: AssistantMessage | undefined;
		try {
			for await (const event of inner) {
				if (finished) return;
				if (isTerminal(event)) {
					terminal = event.type === "done" ? event.message : event.error;
					finish();
				} else {
					lastPartial = event.partial;
					lastEventType = event.type;
					lastEventAt = Date.now();
					arm();
				}
				outer.push(event);
			}
		} catch (error) {
			if (finished) return;
			finish();
			const message = createIdleErrorMessage(
				model,
				lastPartial,
				error instanceof Error ? error.message : String(error),
			);
			outer.push({ type: "error", reason: "error", error: message });
			outer.end(message);
			return;
		}
		if (terminal) {
			outer.end(terminal);
			return;
		}
		if (finished) return;
		finish();
		outer.end(await inner.result());
	})();

	return outer;
}

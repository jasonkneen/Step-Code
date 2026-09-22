import type Anthropic from "@anthropic-ai/sdk";
import { describe, expect, it, vi } from "vitest";
import { stream as streamAnthropic } from "../src/api/anthropic-messages.ts";
import { stream as streamOpenAICompletions } from "../src/api/openai-completions.ts";
import { stream as streamOpenAIResponses } from "../src/api/openai-responses.ts";
import type { AssistantMessage, Context, Model, ToolCall } from "../src/types.ts";
import { anthropicModel } from "./helpers/step-fixtures.ts";

// Invariant: final tool-call arguments must be a complete JSON object. The
// lenient streaming parser salvages `{"path":"a.ts","cont` into
// `{ path: "a.ts" }`, which validates for a tool whose remaining fields are
// optional and would execute with silently truncated input. Every adapter must
// flag such calls with `argumentsError` regardless of the stop reason.

const TRUNCATED = '{"path":"a.ts","cont';
const COMPLETE = '{"path":"a.ts","content":"hi"}';

const mockState = vi.hoisted(() => ({
	chunks: [] as unknown[],
	responseEvents: [] as unknown[],
	requests: [] as unknown[],
}));

function makeStreamResult(items: unknown[]) {
	const stream = {
		async *[Symbol.asyncIterator]() {
			for (const item of items) yield item;
		},
	};
	const result = Promise.resolve(stream) as Promise<typeof stream> & {
		withResponse: () => Promise<{ data: typeof stream; response: { status: number; headers: Headers } }>;
	};
	result.withResponse = async () => ({
		data: stream,
		response: { status: 200, headers: new Headers() },
	});
	return result;
}

vi.mock("openai", () => {
	class FakeOpenAI {
		chat = {
			completions: {
				create: (params: unknown) => {
					mockState.requests.push(params);
					return makeStreamResult(mockState.chunks);
				},
			},
		};
		responses = {
			create: (params: unknown) => {
				mockState.requests.push(params);
				return makeStreamResult(mockState.responseEvents);
			},
		};
	}
	return { default: FakeOpenAI };
});

const userContext: Context = {
	messages: [{ role: "user", content: "write it", timestamp: 1 }],
};

function onlyToolCall(message: AssistantMessage): ToolCall {
	const toolCalls = message.content.filter((block): block is ToolCall => block.type === "toolCall");
	expect(toolCalls).toHaveLength(1);
	return toolCalls[0];
}

function anthropicSse(argumentChunks: string[]): string {
	return [
		{
			type: "message_start",
			message: {
				id: "m",
				usage: { input_tokens: 1, output_tokens: 0, cache_read_input_tokens: 0, cache_creation_input_tokens: 0 },
			},
		},
		{
			type: "content_block_start",
			index: 0,
			content_block: { type: "tool_use", id: "toolu_1", name: "write", input: {} },
		},
		...argumentChunks.map((chunk) => ({
			type: "content_block_delta",
			index: 0,
			delta: { type: "input_json_delta", partial_json: chunk },
		})),
		{ type: "content_block_stop", index: 0 },
		{
			type: "message_delta",
			delta: { stop_reason: "tool_use" },
			usage: { input_tokens: 1, output_tokens: 1, cache_read_input_tokens: 0, cache_creation_input_tokens: 0 },
		},
		{ type: "message_stop" },
	]
		.map((data) => `event: ${data.type}\ndata: ${JSON.stringify(data)}\n`)
		.join("\n");
}

async function runAnthropic(argumentChunks: string[], context: Context = userContext) {
	const requests: unknown[] = [];
	const client = {
		messages: {
			create: (params: unknown) => {
				requests.push(params);
				const response = new Response(anthropicSse(argumentChunks), {
					status: 200,
					headers: { "content-type": "text/event-stream" },
				});
				return { asResponse: async () => response };
			},
		},
	} as unknown as Anthropic;
	const message = await streamAnthropic(anthropicModel(), context, { client }).result();
	return { message, requests };
}

const completionsModel: Model<"openai-completions"> = {
	id: "test-model",
	name: "Test",
	api: "openai-completions",
	provider: "openrouter",
	baseUrl: "https://openrouter.ai/api/v1",
	reasoning: false,
	input: ["text"],
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
	contextWindow: 100_000,
	maxTokens: 4096,
};

async function runCompletions(argumentChunks: string[], context: Context = userContext) {
	mockState.requests = [];
	mockState.chunks = [
		{
			id: "c",
			model: "test-model",
			choices: [
				{
					index: 0,
					delta: {
						tool_calls: [
							{ index: 0, id: "call_1", type: "function", function: { name: "write", arguments: "" } },
						],
					},
					finish_reason: null,
				},
			],
		},
		...argumentChunks.map((chunk) => ({
			id: "c",
			model: "test-model",
			choices: [
				{ index: 0, delta: { tool_calls: [{ index: 0, function: { arguments: chunk } }] }, finish_reason: null },
			],
		})),
		{ id: "c", model: "test-model", choices: [{ index: 0, delta: {}, finish_reason: "tool_calls" }] },
	];
	const message = await streamOpenAICompletions(completionsModel, context, { apiKey: "test" }).result();
	return { message, requests: mockState.requests };
}

const responsesModel: Model<"openai-responses"> = {
	id: "gpt-5-mini",
	name: "GPT-5 Mini",
	api: "openai-responses",
	provider: "openai",
	baseUrl: "https://api.openai.com/v1",
	reasoning: false,
	input: ["text"],
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
	contextWindow: 400_000,
	maxTokens: 128_000,
};

async function runResponses(argumentChunks: string[], context: Context = userContext) {
	mockState.requests = [];
	const item = { type: "function_call", id: "fc_1", call_id: "call_1", name: "write", arguments: "" };
	mockState.responseEvents = [
		{ type: "response.created", sequence_number: 0, response: { id: "r" } },
		{ type: "response.output_item.added", sequence_number: 1, output_index: 0, item },
		...argumentChunks.map((chunk, i) => ({
			type: "response.function_call_arguments.delta",
			sequence_number: 2 + i,
			output_index: 0,
			item_id: "fc_1",
			delta: chunk,
		})),
		{
			type: "response.output_item.done",
			sequence_number: 100,
			output_index: 0,
			item: { ...item, arguments: argumentChunks.join("") },
		},
		{
			type: "response.completed",
			sequence_number: 101,
			response: {
				id: "r",
				status: "completed",
				usage: { input_tokens: 1, output_tokens: 1, total_tokens: 2, input_tokens_details: { cached_tokens: 0 } },
			},
		},
	];
	const message = await streamOpenAIResponses(responsesModel, context, { apiKey: "test" }).result();
	return { message, requests: mockState.requests };
}

const adapters = [
	{ name: "anthropic-messages", run: runAnthropic },
	{ name: "openai-completions", run: runCompletions },
	{ name: "openai-responses", run: runResponses },
];

describe.each(adapters)("$name: strict final tool-call arguments", ({ run }) => {
	it("flags truncated argument JSON on a normal tool-use stop", async () => {
		const { message } = await run(['{"path":', '"a.ts","cont']);
		expect(message.stopReason).toBe("toolUse");
		const toolCall = onlyToolCall(message);
		expect(toolCall.argumentsError).toEqual(expect.any(String));
		// Salvaged arguments are kept for display only.
		expect(toolCall.arguments).toEqual({ path: "a.ts" });
	});

	it("does not flag complete argument JSON", async () => {
		const { message } = await run([COMPLETE.slice(0, 10), COMPLETE.slice(10)]);
		const toolCall = onlyToolCall(message);
		expect(toolCall.argumentsError).toBeUndefined();
		expect("argumentsError" in toolCall).toBe(false);
		expect(toolCall.arguments).toEqual({ path: "a.ts", content: "hi" });
	});

	it("treats empty argument text as a valid no-arg call", async () => {
		const { message } = await run([]);
		const toolCall = onlyToolCall(message);
		expect(toolCall.argumentsError).toBeUndefined();
		expect(toolCall.arguments).toEqual({});
	});

	it("flags argument JSON that is not an object", async () => {
		const { message } = await run(["[1,2]"]);
		expect(onlyToolCall(message).argumentsError).toEqual(expect.any(String));
	});

	it("never sends argumentsError back on the wire when replaying the call", async () => {
		const { message } = await run([TRUNCATED]);
		expect(onlyToolCall(message).argumentsError).toBeDefined();
		const replayContext: Context = {
			messages: [
				...userContext.messages,
				message,
				{
					role: "toolResult",
					toolCallId: onlyToolCall(message).id,
					toolName: "write",
					content: [{ type: "text", text: "not executed" }],
					isError: true,
					timestamp: 2,
				},
			],
		};
		const { requests } = await run([COMPLETE], replayContext);
		expect(requests.length).toBeGreaterThan(0);
		expect(JSON.stringify(requests[requests.length - 1])).not.toContain("argumentsError");
	});
});

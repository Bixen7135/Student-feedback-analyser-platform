import assert from "node:assert/strict";
import test from "node:test";

import { DELETE, GET } from "../../app/api/[...path]/route.ts";

test("GET proxy forwards path and query string to backend", async () => {
  const calls: Array<{ input: string; init?: RequestInit & { duplex?: "half" } }> = [];
  const originalFetch = globalThis.fetch;

  globalThis.fetch = (async (input, init) => {
    calls.push({
      input: String(input),
      init: init as RequestInit & { duplex?: "half" },
    });
    return Response.json({ ok: true }, { status: 200 });
  }) as typeof fetch;

  try {
    const req = new Request("http://localhost:3000/api/models?run_id=run_123&page=2");
    const res = await GET(req, {
      params: Promise.resolve({ path: ["models"] }),
    });

    assert.equal(res.status, 200);
    assert.equal(calls.length, 1);
    assert.equal(
      calls[0]?.input,
      "http://localhost:8000/api/models?run_id=run_123&page=2",
    );
    assert.equal(calls[0]?.init?.method, "GET");
    assert.equal(calls[0]?.init?.body, undefined);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("DELETE proxy forwards request body and returns 502 on upstream failure", async () => {
  const calls: Array<{ input: string; init?: RequestInit & { duplex?: "half" } }> = [];
  const originalFetch = globalThis.fetch;

  globalThis.fetch = (async (input, init) => {
    calls.push({
      input: String(input),
      init: init as RequestInit & { duplex?: "half" },
    });
    throw new Error("backend down");
  }) as typeof fetch;

  try {
    const req = new Request("http://localhost:3000/api/runs/run_123", {
      method: "DELETE",
      body: JSON.stringify({ force: true }),
      headers: { "Content-Type": "application/json" },
    });
    const res = await DELETE(req, {
      params: Promise.resolve({ path: ["runs", "run_123"] }),
    });
    const body = await res.json();

    assert.equal(calls.length, 1);
    assert.equal(calls[0]?.input, "http://localhost:8000/api/runs/run_123");
    assert.equal(calls[0]?.init?.method, "DELETE");
    assert.equal(calls[0]?.init?.duplex, "half");
    assert.ok(calls[0]?.init?.body instanceof ReadableStream);
    assert.equal(res.status, 502);
    assert.equal(body.detail, "backend down");
  } finally {
    globalThis.fetch = originalFetch;
  }
});

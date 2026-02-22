/**
 * Route Handler for /api/runs/[runId]
 *
 * Next.js 16 rewrites() do not forward DELETE requests, so we use an explicit
 * Route Handler for GET and DELETE on this path. All other sub-paths
 * (/stages/..., /metrics/..., /artifacts/...) still go through next.config.ts rewrites.
 */

const BACKEND_BASE = process.env.BACKEND_URL ?? "http://localhost:8000";

export async function GET(
  _req: Request,
  { params }: { params: Promise<{ runId: string }> },
) {
  const { runId } = await params;
  const res = await fetch(`${BACKEND_BASE}/api/runs/${runId}`);
  const data = await res.json().catch(() => ({}));
  return Response.json(data, { status: res.status });
}

export async function DELETE(
  _req: Request,
  { params }: { params: Promise<{ runId: string }> },
) {
  const { runId } = await params;
  const res = await fetch(`${BACKEND_BASE}/api/runs/${runId}`, { method: "DELETE" });
  if (res.status === 204) return new Response(null, { status: 204 });
  const data = await res.json().catch(() => ({}));
  return Response.json(data, { status: res.status });
}

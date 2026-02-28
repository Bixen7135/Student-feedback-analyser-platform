/**
 * Catch-all API proxy for /api/*.
 *
 * This app routes all /api/* traffic through Next route handlers, which proxy
 * requests to the FastAPI backend for all verbs.
 */

const BACKEND_BASE = process.env.BACKEND_URL ?? "http://localhost:8000";

type RouteContext = {
  params: Promise<{ path: string[] }>;
};

async function forward(req: Request, { params }: RouteContext): Promise<Response> {
  const { path } = await params;
  const incomingUrl = new URL(req.url);
  const targetPath = path.map((segment) => encodeURIComponent(segment)).join("/");
  const targetUrl = `${BACKEND_BASE}/api/${targetPath}${incomingUrl.search}`;

  const headers = new Headers(req.headers);
  headers.delete("host");
  headers.delete("content-length");

  const method = req.method.toUpperCase();
  const init: RequestInit & { duplex?: "half" } = {
    method,
    headers,
    cache: "no-store",
    redirect: "manual",
  };

  if (method !== "GET" && method !== "HEAD" && req.body) {
    init.body = req.body;
    init.duplex = "half";
  }

  try {
    return await fetch(targetUrl, init);
  } catch (error) {
    const detail =
      error instanceof Error ? error.message : "Request could not be forwarded.";
    return Response.json({ detail }, { status: 502 });
  }
}

export async function GET(req: Request, context: RouteContext) {
  return forward(req, context);
}

export async function POST(req: Request, context: RouteContext) {
  return forward(req, context);
}

export async function PUT(req: Request, context: RouteContext) {
  return forward(req, context);
}

export async function PATCH(req: Request, context: RouteContext) {
  return forward(req, context);
}

export async function DELETE(req: Request, context: RouteContext) {
  return forward(req, context);
}

export async function HEAD(req: Request, context: RouteContext) {
  return forward(req, context);
}

export async function OPTIONS(req: Request, context: RouteContext) {
  return forward(req, context);
}

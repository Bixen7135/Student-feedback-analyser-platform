import type { NextRequest } from "next/server";
import { NextResponse } from "next/server";

const DEVTOOLS_SOURCEMAPS = new Set([
  "installHook.js.map",
  "react_devtools_backend_compact.js.map",
]);

function buildEmptySourceMap(file: string) {
  return {
    version: 3,
    file,
    sources: [],
    names: [],
    mappings: "",
  };
}

export function middleware(req: NextRequest) {
  const file = req.nextUrl.pathname.split("/").pop() ?? "";
  if (!DEVTOOLS_SOURCEMAPS.has(file)) {
    return NextResponse.next();
  }

  const headers = new Headers({
    "content-type": "application/json; charset=utf-8",
    "cache-control": "no-store",
  });

  if (req.method === "HEAD") {
    return new NextResponse(null, { status: 200, headers });
  }

  return new NextResponse(JSON.stringify(buildEmptySourceMap(file)), {
    status: 200,
    headers,
  });
}

export const config = {
  matcher: "/:path*",
};

"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";

type ApiStatus = "connecting" | "online" | "offline";

interface NavItem {
  href: string;
  label: string;
  icon: React.ReactNode;
}

const navItems: NavItem[] = [
  {
    href: "/",
    label: "Dashboard",
    icon: (
      <svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden="true">
        <rect x="1" y="1" width="5" height="5" rx="1" stroke="currentColor" strokeWidth="1.35" />
        <rect x="8" y="1" width="5" height="5" rx="1" stroke="currentColor" strokeWidth="1.35" />
        <rect x="1" y="8" width="5" height="5" rx="1" stroke="currentColor" strokeWidth="1.35" />
        <rect x="8" y="8" width="5" height="5" rx="1" stroke="currentColor" strokeWidth="1.35" />
      </svg>
    ),
  },
  {
    href: "/datasets",
    label: "Datasets",
    icon: (
      <svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden="true">
        <ellipse cx="7" cy="4" rx="5" ry="2" stroke="currentColor" strokeWidth="1.35" />
        <path d="M2 4V7C2 8.1 4.24 9 7 9C9.76 9 12 8.1 12 7V4" stroke="currentColor" strokeWidth="1.35" />
        <path d="M2 7V10C2 11.1 4.24 12 7 12C9.76 12 12 11.1 12 10V7" stroke="currentColor" strokeWidth="1.35" />
      </svg>
    ),
  },
  {
    href: "/models",
    label: "Models",
    icon: (
      <svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden="true">
        <path d="M7 1L12.5 4V10L7 13L1.5 10V4L7 1Z" stroke="currentColor" strokeWidth="1.35" strokeLinejoin="round" />
        <path d="M7 7L12.5 4M7 7L1.5 4M7 7V13" stroke="currentColor" strokeWidth="1.35" />
      </svg>
    ),
  },
  {
    href: "/runs",
    label: "Run History",
    icon: (
      <svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden="true">
        <circle cx="7" cy="7" r="5.5" stroke="currentColor" strokeWidth="1.35" />
        <path d="M7 4V7.5L9 9" stroke="currentColor" strokeWidth="1.35" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    ),
  },
  {
    href: "/runs/new",
    label: "New Run",
    icon: (
      <svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden="true">
        <circle cx="7" cy="7" r="5.5" stroke="currentColor" strokeWidth="1.35" />
        <path d="M7 4.5V9.5M4.5 7H9.5" stroke="currentColor" strokeWidth="1.35" strokeLinecap="round" />
      </svg>
    ),
  },
  {
    href: "/training",
    label: "Training",
    icon: (
      <svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden="true">
        <path d="M2 11L5 7L7.5 9.5L10 5.5L12 8" stroke="currentColor" strokeWidth="1.35" strokeLinecap="round" strokeLinejoin="round" />
        <circle cx="12" cy="4" r="1.5" stroke="currentColor" strokeWidth="1.35" />
      </svg>
    ),
  },
  {
    href: "/analyses",
    label: "Analyses",
    icon: (
      <svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden="true">
        <path d="M2 12V5M5 12V3M8 12V7M11 12V1" stroke="currentColor" strokeWidth="1.35" strokeLinecap="round" />
      </svg>
    ),
  },
];

const STATUS_CONFIG: Record<ApiStatus, { label: string; color: string; pulse: boolean }> = {
  connecting: { label: "connecting...", color: "var(--warning, #f59e0b)", pulse: true },
  online: { label: "api | port 8000", color: "var(--success)", pulse: false },
  offline: { label: "api | offline", color: "var(--error, #ef4444)", pulse: false },
};

export function Sidebar() {
  const pathname = usePathname();
  const [apiStatus, setApiStatus] = useState<ApiStatus>("connecting");

  useEffect(() => {
    let cancelled = false;

    async function check() {
      try {
        const res = await fetch("http://localhost:8000/health", {
          signal: AbortSignal.timeout(3000),
        });
        if (!cancelled) setApiStatus(res.ok ? "online" : "offline");
      } catch {
        if (!cancelled) setApiStatus("offline");
      }
    }

    check();
    const id = setInterval(check, 15_000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  return (
    <aside
      className="flex-shrink-0 flex flex-col sticky top-0"
      style={{
        width: "216px",
        height: "100vh",
        background: "var(--bg-surface)",
        borderRight: "1px solid var(--border-dim)",
      }}
    >
      <div
        className="flex items-center gap-3 px-5"
        style={{
          padding: "18px 20px",
          borderBottom: "1px solid var(--border-dim)",
        }}
      >
        <div
          className="flex items-center justify-center rounded flex-shrink-0"
          style={{
            width: "28px",
            height: "28px",
            background: "var(--gold-faint)",
            border: "1px solid var(--gold-muted)",
          }}
        >
          <svg width="12" height="12" viewBox="0 0 12 12" fill="none" aria-hidden="true">
            <path
              d="M1.5 10.5L5 2.5L6 5.5M6 5.5L9 10.5M6 5.5L10 2"
              stroke="var(--gold)"
              strokeWidth="1.4"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        </div>
        <div>
          <div
            style={{
              fontFamily: "var(--font-syne)",
              fontWeight: 700,
              fontSize: "12px",
              letterSpacing: "0.09em",
              textTransform: "uppercase",
              color: "var(--text-primary)",
              lineHeight: 1.2,
            }}
          >
            Feedback Lab
          </div>
          <div
            style={{
              fontFamily: "var(--font-jetbrains)",
              fontSize: "10px",
              color: "var(--text-tertiary)",
              marginTop: "2px",
            }}
          >
            v0.1.0
          </div>
        </div>
      </div>

      <nav className="flex-1 flex flex-col" style={{ padding: "16px 12px 8px" }}>
        <div
          style={{
            fontFamily: "var(--font-syne)",
            fontSize: "9.5px",
            fontWeight: 700,
            letterSpacing: "0.18em",
            textTransform: "uppercase",
            color: "var(--text-tertiary)",
            padding: "0 8px 10px",
          }}
        >
          Navigation
        </div>
        <div className="flex flex-col gap-0.5">
          {navItems.map((item) => {
            const isActive =
              item.href === "/"
                ? pathname === "/"
                : item.href === "/runs"
                ? pathname === "/runs" ||
                  (pathname.startsWith("/runs/") &&
                    !pathname.startsWith("/runs/new"))
                : item.href === "/datasets"
                ? pathname === "/datasets" || pathname.startsWith("/datasets/")
                : pathname === item.href || pathname.startsWith(item.href + "/");
            return (
              <Link
                key={item.href}
                href={item.href}
                className="flex items-center gap-2.5 rounded transition-all duration-100"
                style={{
                  padding: "7px 10px",
                  background: isActive ? "var(--gold-faint)" : "transparent",
                  color: isActive ? "var(--gold)" : "var(--text-secondary)",
                  borderLeft: `2px solid ${isActive ? "var(--gold)" : "transparent"}`,
                  marginLeft: "-1px",
                  fontSize: "13px",
                  fontWeight: isActive ? 500 : 400,
                }}
              >
                <span style={{ opacity: isActive ? 1 : 0.6, flexShrink: 0 }}>
                  {item.icon}
                </span>
                {item.label}
              </Link>
            );
          })}
        </div>
      </nav>

      <div
        style={{
          padding: "14px 20px",
          borderTop: "1px solid var(--border-dim)",
        }}
      >
        <div className="flex items-center gap-2" style={{ marginBottom: "5px" }}>
          <span
            className="rounded-full flex-shrink-0"
            style={{
              width: "6px",
              height: "6px",
              background: STATUS_CONFIG[apiStatus].color,
              opacity: STATUS_CONFIG[apiStatus].pulse ? 0.7 : 1,
              transition: "background 0.3s",
            }}
          />
          <span
            style={{
              fontFamily: "var(--font-jetbrains)",
              fontSize: "10px",
              color: "var(--text-tertiary)",
            }}
          >
            {STATUS_CONFIG[apiStatus].label}
          </span>
        </div>
        <div
          style={{
            fontFamily: "var(--font-jetbrains)",
            fontSize: "10px",
            color: "var(--text-tertiary)",
          }}
        >
          RU | KZ | multilingual
        </div>
      </div>
    </aside>
  );
}

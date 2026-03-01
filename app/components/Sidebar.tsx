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
    label: "Run Pipeline",
    icon: (
      <svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden="true">
        <circle cx="7" cy="7" r="5.5" stroke="currentColor" strokeWidth="1.35" />
        <path d="M7 4.5V9.5M4.5 7H9.5" stroke="currentColor" strokeWidth="1.35" strokeLinecap="round" />
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

const MOBILE_NAV_MEDIA_QUERY = "(max-width: 56rem)";

export function Sidebar() {
  const pathname = usePathname();
  const [apiStatus, setApiStatus] = useState<ApiStatus>("connecting");
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const [isMobileViewport, setIsMobileViewport] = useState(false);

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

  useEffect(() => {
    const mediaQuery = window.matchMedia(MOBILE_NAV_MEDIA_QUERY);
    const syncViewport = (event?: MediaQueryListEvent) => {
      const nextIsMobileViewport = event?.matches ?? mediaQuery.matches;
      setIsMobileViewport(nextIsMobileViewport);

      if (!nextIsMobileViewport) {
        setIsMobileMenuOpen(false);
      }
    };

    syncViewport();

    if ("addEventListener" in mediaQuery) {
      mediaQuery.addEventListener("change", syncViewport);
    } else {
      mediaQuery.addListener(syncViewport);
    }

    return () => {
      if ("removeEventListener" in mediaQuery) {
        mediaQuery.removeEventListener("change", syncViewport);
      } else {
        mediaQuery.removeListener(syncViewport);
      }
    };
  }, []);

  useEffect(() => {
    if (!isMobileViewport || !isMobileMenuOpen) {
      return;
    }

    const previousBodyOverflow = document.body.style.overflow;
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setIsMobileMenuOpen(false);
      }
    };

    document.body.style.overflow = "hidden";
    window.addEventListener("keydown", handleKeyDown);

    return () => {
      document.body.style.overflow = previousBodyOverflow;
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [isMobileMenuOpen, isMobileViewport]);

  const disableNavTabbing = isMobileViewport && !isMobileMenuOpen;

  return (
    <aside className={`app-sidebar${isMobileMenuOpen ? " is-mobile-open" : ""}`}>
      <div className="app-sidebar__mobile-bar">
        <div className="app-sidebar__brand">
          <div className="app-sidebar__brand-mark">
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
          <div className="app-sidebar__brand-meta">
            <div className="app-sidebar__brand-title">Feedback Lab</div>
            <div className="app-sidebar__brand-version">v0.1.0</div>
          </div>
        </div>
        <button
          type="button"
          className="app-sidebar__mobile-toggle"
          aria-label={isMobileMenuOpen ? "Close navigation menu" : "Open navigation menu"}
          aria-expanded={isMobileMenuOpen}
          aria-controls="app-sidebar-drawer"
          onClick={() => setIsMobileMenuOpen((open) => !open)}
        >
          <span className="app-sidebar__mobile-toggle-line" />
          <span className="app-sidebar__mobile-toggle-line" />
          <span className="app-sidebar__mobile-toggle-line" />
        </button>
      </div>

      {isMobileMenuOpen ? (
        <button
          type="button"
          className="app-sidebar__mobile-overlay"
          aria-label="Close navigation menu"
          onClick={() => setIsMobileMenuOpen(false)}
        />
      ) : null}

      <div
        id="app-sidebar-drawer"
        className="app-sidebar__drawer"
        aria-hidden={isMobileViewport ? !isMobileMenuOpen : undefined}
      >
        <nav className="app-sidebar__nav">
          <div className="app-sidebar__label">Navigation</div>
          <div className="app-sidebar__list">
            {navItems.map((item) => {
              const isActive =
                item.href === "/"
                  ? pathname === "/"
                  : item.href === "/runs"
                  ? pathname === "/runs" ||
                    pathname.startsWith("/models") ||
                    (pathname.startsWith("/runs/") &&
                      !pathname.startsWith("/runs/new"))
                  : item.href === "/datasets"
                  ? pathname === "/datasets" || pathname.startsWith("/datasets/")
                  : item.href === "/runs/new"
                  ? pathname === "/runs/new" || pathname.startsWith("/training")
                  : pathname === item.href || pathname.startsWith(item.href + "/");
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={`app-sidebar__link${isActive ? " is-active" : ""}`}
                  onClick={() => setIsMobileMenuOpen(false)}
                  tabIndex={disableNavTabbing ? -1 : undefined}
                >
                  <span className="app-sidebar__icon">{item.icon}</span>
                  <span className="app-sidebar__text">{item.label}</span>
                </Link>
              );
            })}
          </div>
        </nav>

        <div className="app-sidebar__footer">
          <div className="app-sidebar__status-row">
            <span
              className={`app-sidebar__status-dot${STATUS_CONFIG[apiStatus].pulse ? " animate-pulse-dot" : ""}`}
              style={{
                background: STATUS_CONFIG[apiStatus].color,
                opacity: STATUS_CONFIG[apiStatus].pulse ? 0.7 : 1,
              }}
            />
            <span className="app-sidebar__status-copy">{STATUS_CONFIG[apiStatus].label}</span>
          </div>
          <div className="app-sidebar__locale">RU | KZ | multilingual</div>
        </div>
      </div>
    </aside>
  );
}

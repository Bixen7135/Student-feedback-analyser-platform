import Link from "next/link";

export function Navbar() {
  return (
    <nav className="app-navbar">
      <Link href="/" className="app-navbar__brand">
        SFAP
      </Link>
      <Link href="/runs" className="app-navbar__link">
        Run History
      </Link>
      <Link href="/runs/new" className="app-navbar__link">
        Run Pipeline
      </Link>
      <span className="app-navbar__meta">Student Feedback Analysis Platform v0.1</span>
    </nav>
  );
}

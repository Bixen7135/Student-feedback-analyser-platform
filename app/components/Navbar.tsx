import Link from "next/link";

export function Navbar() {
  return (
    <nav className="bg-slate-800 text-white px-6 py-3 flex items-center gap-6">
      <Link href="/" className="font-bold text-lg tracking-tight hover:text-slate-300">
        SFAP
      </Link>
      <Link href="/runs" className="text-sm hover:text-slate-300">
        Run History
      </Link>
      <Link href="/runs/new" className="text-sm hover:text-slate-300">
        New Run
      </Link>
      <span className="ml-auto text-xs text-slate-400">
        Student Feedback Analysis Platform v0.1
      </span>
    </nav>
  );
}

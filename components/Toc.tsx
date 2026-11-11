"use client";

import { useEffect, useState } from "react";
import type { TocEntry } from "@/lib/markdown";

// "On this page": the list of sections, with the current one marked.
export function Toc({ entries }: { entries: TocEntry[] }) {
  const [active, setActive] = useState<string>();

  useEffect(() => {
    const headings = entries
      .map((entry) => document.getElementById(entry.id))
      .filter((el): el is HTMLElement => el !== null);
    const update = () => {
      const passed = headings.filter((el) => el.getBoundingClientRect().top <= 120);
      setActive((passed.at(-1) ?? headings[0])?.id);
    };
    update();
    window.addEventListener("scroll", update, { passive: true });
    return () => window.removeEventListener("scroll", update);
  }, [entries]);

  return (
    <nav className="toc" aria-label="On this page">
      <p className="eyebrow">On this page</p>
      <ul>
        {entries.map((entry) => (
          <li key={entry.id} className={`toc__item toc__item--${entry.depth}`}>
            <a href={`#${entry.id}`} aria-current={active === entry.id ? "location" : undefined}>
              {entry.text}
            </a>
          </li>
        ))}
      </ul>
    </nav>
  );
}

"use client";

import { useRouter } from "next/navigation";
import { useEffect, useMemo, useRef, useState } from "react";
import type { SearchItem } from "@/lib/site";

interface Result {
  item: SearchItem;
  phrase?: string; // the phrase of the lesson that matches, if the title does not
  rank: number;
}

function search(items: SearchItem[], query: string): Result[] {
  const words = query.toLowerCase().split(/\s+/).filter(Boolean);
  if (words.length === 0) return [];
  const matches = (text: string) => words.every((word) => text.toLowerCase().includes(word));
  const results: Result[] = [];
  for (const item of items) {
    if (matches(item.title)) results.push({ item, rank: 0 });
    for (const phrase of item.phrases.filter(matches).slice(0, 2)) results.push({ item, phrase, rank: 1 });
  }
  return results.sort((a, b) => a.rank - b.rank).slice(0, 8);
}

// Search in lesson titles, section titles, defined terms and step titles.
// Open with the button, "/" or Ctrl/Cmd + K.
export function Search() {
  const dialog = useRef<HTMLDialogElement>(null);
  const [items, setItems] = useState<SearchItem[]>();
  const [query, setQuery] = useState("");
  const [failed, setFailed] = useState(false);
  const router = useRouter();

  const open = () => {
    dialog.current?.showModal();
    if (!items) {
      setFailed(false);
      fetch("/search-index.json")
        .then((response) => {
          if (!response.ok) throw new Error(`search index: HTTP ${response.status}`);
          return response.json();
        })
        .then(setItems)
        .catch((error) => {
          console.error(error);
          setFailed(true);
        });
    }
  };

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const typing = event.target instanceof HTMLElement && /^(INPUT|TEXTAREA|SELECT)$/.test(event.target.tagName);
      const shortcut = (event.key === "k" && (event.metaKey || event.ctrlKey)) || (event.key === "/" && !typing);
      if (shortcut && !dialog.current?.open) {
        event.preventDefault();
        open();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  const results = useMemo(() => search(items ?? [], query), [items, query]);

  const go = (href: string) => {
    dialog.current?.close();
    router.push(href);
  };

  return (
    <>
      <button type="button" className="search-button" onClick={open}>
        <span>Search</span>
        <kbd>/</kbd>
      </button>
      <dialog ref={dialog} className="search" onClose={() => setQuery("")}>
        <form
          method="dialog"
          onSubmit={(event) => {
            event.preventDefault();
            if (results[0]) go(results[0].item.href);
          }}
        >
          <input
            type="search"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Search the lessons: attention, LoRA, RAG…"
            aria-label="Search the lessons"
            autoFocus
          />
        </form>
        <ul>
          {results.map(({ item, phrase }) => (
            <li key={item.href + (phrase ?? "")}>
              <button type="button" onClick={() => go(item.href)}>
                <strong>{phrase ?? item.title}</strong>
                <span>{phrase ? `${item.context} · ${item.title}` : item.context}</span>
              </button>
            </li>
          ))}
          {failed && <li className="search__empty">The search index did not load. Close the search and try again.</li>}
          {items && query && results.length === 0 && <li className="search__empty">No lesson matches “{query}”.</li>}
        </ul>
      </dialog>
    </>
  );
}

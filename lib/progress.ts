// Which lessons the reader marked as done. Kept in the browser only (localStorage).
"use client";

import { useSyncExternalStore } from "react";

const KEY = "llm-course-explained:done";
const EVENT = "llm-course-explained:progress";
const EMPTY: string[] = [];

let snapshot: string[] = EMPTY;
let snapshotRaw: string | null = null;

function read(): string[] {
  const raw = window.localStorage.getItem(KEY);
  if (raw !== snapshotRaw) {
    snapshotRaw = raw;
    snapshot = raw ? (JSON.parse(raw) as string[]) : EMPTY;
  }
  return snapshot;
}

function subscribe(onChange: () => void) {
  window.addEventListener(EVENT, onChange);
  window.addEventListener("storage", onChange);
  return () => {
    window.removeEventListener(EVENT, onChange);
    window.removeEventListener("storage", onChange);
  };
}

// The hrefs of the lessons that are done. Empty on the server and during hydration.
export function useDone(): string[] {
  return useSyncExternalStore(subscribe, read, () => EMPTY);
}

export function setDone(href: string, done: boolean) {
  const current = read().filter((item) => item !== href);
  window.localStorage.setItem(KEY, JSON.stringify(done ? [...current, href] : current));
  window.dispatchEvent(new Event(EVENT));
}

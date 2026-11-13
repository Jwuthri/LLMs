"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { setDone, useDone } from "@/lib/progress";

// A small check mark next to a lesson that the reader finished.
export function DoneMark({ href }: { href: string }) {
  const done = useDone().includes(href);
  return (
    <span className={"done-mark" + (done ? " is-done" : "")} aria-label={done ? "Done" : undefined}>
      {done ? "✓" : ""}
    </span>
  );
}

// "3 of 8 done" with a thin bar.
export function TrackProgress({ hrefs }: { hrefs: string[] }) {
  const done = useDone().filter((href) => hrefs.includes(href)).length;
  return (
    <div className="track-progress" aria-label={`${done} of ${hrefs.length} lessons done`}>
      <div className="track-progress__bar">
        <div className="track-progress__fill" style={{ width: `${(done / hrefs.length) * 100}%` }} />
      </div>
      <span>
        {done} / {hrefs.length}
      </span>
    </div>
  );
}

// The button at the end of a lesson.
export function CompleteButton({ href, nextHref }: { href: string; nextHref?: string }) {
  const done = useDone().includes(href);
  return (
    <div className="complete">
      <button
        type="button"
        className={"button" + (done ? " button--quiet" : "")}
        aria-pressed={done}
        onClick={() => setDone(href, !done)}
      >
        {done ? "✓ Done. Mark as not done" : "Mark this lesson as done"}
      </button>
      {done && nextHref && (
        <Link className="button" href={nextHref}>
          Next lesson →
        </Link>
      )}
    </div>
  );
}

// A thin line at the top of the window that shows how far the reader is in the page.
export function ReadingBar() {
  const [ratio, setRatio] = useState(0);
  useEffect(() => {
    const update = () => {
      const max = document.documentElement.scrollHeight - window.innerHeight;
      setRatio(max > 0 ? Math.min(1, window.scrollY / max) : 0);
    };
    update();
    window.addEventListener("scroll", update, { passive: true });
    window.addEventListener("resize", update);
    return () => {
      window.removeEventListener("scroll", update);
      window.removeEventListener("resize", update);
    };
  }, []);
  return <div className="reading-bar" style={{ transform: `scaleX(${ratio})` }} aria-hidden="true" />;
}

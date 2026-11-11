import Link from "next/link";
import { Search } from "./Search";
import { TRACKS } from "@/lib/site";

export function Header() {
  return (
    <header className="site-header">
      <div className="site-header__inner">
        <Link href="/" className="wordmark">
          <span className="wordmark__dot" aria-hidden="true" />
          LLM Course<span className="wordmark__soft">, Explained</span>
        </Link>
        <nav className="site-nav" aria-label="Tracks">
          <Link href="/toy-example/">Toy example</Link>
          {TRACKS.map((track) => (
            <Link key={track.id} href={`/${track.id}/`}>
              {track.title.replace(/^(The )?LLM /, "")}
            </Link>
          ))}
        </nav>
        <Search />
      </div>
    </header>
  );
}

export function Footer() {
  return (
    <footer className="site-footer">
      <p>
        The list of topics and the reference links come from the{" "}
        <a href="https://github.com/mlabonne/llm-course" target="_blank" rel="noopener">
          LLM Course
        </a>{" "}
        by Maxime Labonne (Apache-2.0). The explanations, examples, figures and widgets are new.
      </p>
    </footer>
  );
}

import type { Metadata } from "next";
import Link from "next/link";
import { Article } from "@/components/Article";
import { renderPage } from "@/lib/markdown";
import { lessonsOf } from "@/lib/site";

export const metadata: Metadata = {
  title: "AI safety glossary",
  description: "The main words of AI safety: one sentence for each word, and the lesson that explains it.",
};

export default async function GlossaryPage() {
  const page = await renderPage("glossary.md");
  const first = lessonsOf("safety")[0];
  return (
    <Article
      page={page}
      header={
        <header className="lesson-header">
          <p className="eyebrow">AI Safety</p>
          <h1>AI safety glossary</h1>
        </header>
      }
      footer={
        <footer className="lesson-footer">
          <nav className="pager" aria-label="Next lesson">
            <span />
            <Link href={first.href} className="pager__link pager__link--next">
              <span className="eyebrow">First lesson of the track →</span>
              <span>{first.title}</span>
            </Link>
          </nav>
        </footer>
      }
    />
  );
}

import type { Metadata } from "next";
import Link from "next/link";
import { Article } from "@/components/Article";
import { renderPage } from "@/lib/markdown";
import { allLessons } from "@/lib/site";

export const metadata: Metadata = {
  title: "The toy example",
  description: "The one small model that all lessons use: three words, numbers that you can calculate by hand.",
};

export default async function ToyExamplePage() {
  const page = await renderPage("toy-example.md");
  const first = allLessons()[0];
  return (
    <Article
      page={page}
      header={
        <header className="lesson-header">
          <p className="eyebrow">Start here</p>
          <h1>The toy example</h1>
        </header>
      }
      footer={
        <footer className="lesson-footer">
          <nav className="pager" aria-label="Next lesson">
            <span />
            <Link href={first.href} className="pager__link pager__link--next">
              <span className="eyebrow">First lesson →</span>
              <span>{first.title}</span>
            </Link>
          </nav>
        </footer>
      }
    />
  );
}

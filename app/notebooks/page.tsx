import type { Metadata } from "next";
import Link from "next/link";
import { Article } from "@/components/Article";
import { renderPage } from "@/lib/markdown";

export const metadata: Metadata = {
  title: "Notebook guide",
  description: "The notebooks of the LLM Course: what each one does, which lesson to read first, and what to look at.",
};

export default async function NotebooksPage() {
  const page = await renderPage("notebooks.md");
  return (
    <Article
      page={page}
      header={
        <header className="lesson-header">
          <p className="eyebrow">Practice</p>
          <h1>Notebook guide</h1>
        </header>
      }
      footer={
        <footer className="lesson-footer">
          <nav className="pager" aria-label="Lessons">
            <span />
            <Link href="/#tracks" className="pager__link pager__link--next">
              <span className="eyebrow">Back to →</span>
              <span>The lessons</span>
            </Link>
          </nav>
        </footer>
      }
    />
  );
}

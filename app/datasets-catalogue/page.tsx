import type { Metadata } from "next";
import Link from "next/link";
import { Article } from "@/components/Article";
import { renderPage } from "@/lib/markdown";
import { lessonsOf } from "@/lib/site";

export const metadata: Metadata = {
  title: "Dataset catalogue",
  description: "Public datasets and tools for LLM post-training, by family, with size, date and licence note.",
};

export default async function DatasetCataloguePage() {
  const page = await renderPage("datasets-catalogue.md");
  const first = lessonsOf("datasets")[0];
  return (
    <Article
      page={page}
      header={
        <header className="lesson-header">
          <p className="eyebrow">LLM Datasets</p>
          <h1>Dataset catalogue</h1>
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

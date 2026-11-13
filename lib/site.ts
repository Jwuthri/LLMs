// Site structure: the tracks and their lessons, read from content/ at build time.
import fs from "node:fs";
import path from "node:path";

export const CONTENT_DIR = path.join(process.cwd(), "content");

export type TrackId = "fundamentals" | "scientist" | "engineer" | "safety" | "datasets";

export interface Track {
  id: TrackId;
  title: string;
  tagline: string;
}

export const TRACKS: Track[] = [
  {
    id: "fundamentals",
    title: "LLM Fundamentals",
    tagline: "The base: mathematics, Python, neural networks and language processing.",
  },
  {
    id: "scientist",
    title: "The LLM Scientist",
    tagline: "How to build a model and make it better.",
  },
  {
    id: "engineer",
    title: "The LLM Engineer",
    tagline: "How to build applications with a model and put them in service.",
  },
  {
    id: "safety",
    title: "AI Safety",
    tagline: "How to test a model, protect it, and check what it does inside.",
  },
  {
    id: "datasets",
    title: "LLM Datasets",
    tagline: "How to measure, choose, mix and build the data that a model learns from.",
  },
];

export interface LessonMeta {
  track: TrackId;
  slug: string; // "fine-tuning"
  file: string; // "scientist/04-fine-tuning.md"
  number: number; // 4
  title: string; // "Supervised fine-tuning"
  summary: string; // first sentence of the core idea
  parts: string[]; // titles of the numbered sub-parts
  minutes: number; // reading time
  href: string; // "/scientist/fine-tuning/"
}

function readLessonMeta(track: TrackId, fileName: string): LessonMeta {
  const file = `${track}/${fileName}`;
  const text = fs.readFileSync(path.join(CONTENT_DIR, file), "utf8");
  const heading = text.match(/^# (\d+)\. (.+)$/m);
  if (!heading) throw new Error(`${file}: the first heading must be "# N. Title"`);
  const core = text.match(/^## Core idea\n+([\s\S]*?)\n\n/m);
  if (!core) throw new Error(`${file}: missing "## Core idea"`);
  const firstSentence = core[1].replace(/\n/g, " ").replace(/\*\*/g, "").match(/^.*?[.!?](?=\s|$)/);
  const slug = fileName.replace(/^\d+-/, "").replace(/\.md$/, "");
  const prose = text.replace(/```[\s\S]*?```/g, "");
  return {
    track,
    slug,
    file,
    number: Number(heading[1]),
    title: heading[2],
    summary: firstSentence ? firstSentence[0] : core[1],
    parts: [...text.matchAll(/^## \d+\. (.+)$/gm)].map((m) => m[1]),
    minutes: Math.max(1, Math.round(prose.split(/\s+/).length / 200)),
    href: `/${track}/${slug}/`,
  };
}

let cache: LessonMeta[] | undefined;

// The build reads the lessons once. The dev server reads them again on each call, so that a new or
// changed lesson shows without a restart.
export function allLessons(): LessonMeta[] {
  if (cache && process.env.NODE_ENV === "production") return cache;
  cache = TRACKS.flatMap((track) =>
    fs
      .readdirSync(path.join(CONTENT_DIR, track.id))
      .filter((name) => name.endsWith(".md"))
      .sort()
      .map((name) => readLessonMeta(track.id, name))
  );
  return cache;
}

export function lessonsOf(track: TrackId): LessonMeta[] {
  return allLessons().filter((lesson) => lesson.track === track);
}

export function findLesson(track: string, slug: string): LessonMeta | undefined {
  return allLessons().find((lesson) => lesson.track === track && lesson.slug === slug);
}

export function trackOf(id: string): Track | undefined {
  return TRACKS.find((track) => track.id === id);
}

// Route of a content file, for links between pages: "scientist/04-fine-tuning.md" -> "/scientist/fine-tuning/".
export function routeOfFile(file: string): string {
  if (file === "toy-example.md") return "/toy-example/";
  if (file === "glossary.md") return "/glossary/";
  if (file === "datasets-catalogue.md") return "/datasets-catalogue/";
  if (file === "notebooks.md") return "/notebooks/";
  const lesson = allLessons().find((item) => item.file === file);
  if (!lesson) throw new Error(`Link to a page that does not exist: ${file}`);
  return lesson.href;
}

export interface SearchItem {
  href: string;
  title: string;
  context: string; // track name
  phrases: string[]; // section titles, defined terms and step titles
}

// The phrases in bold are the defined terms and the step titles of a page.
function phrasesOf(file: string): string[] {
  const text = fs.readFileSync(path.join(CONTENT_DIR, file), "utf8").replace(/```[\s\S]*?```/g, "");
  const sections = [...text.matchAll(/^## (?:\d+\. )?(.+)$/gm)].map((m) => m[1]);
  const bold = [...text.matchAll(/\*\*([^*\n]+)\*\*/g)]
    .map((m) => m[1].replace(/[.:]$/, "").replace(/[$`\\]/g, ""))
    .filter((phrase) => phrase !== "Why" && !phrase.startsWith('"'));
  return [...new Set([...sections, ...bold])];
}

export function searchIndex(): SearchItem[] {
  return [
    { href: "/toy-example/", title: "The toy example", context: "Start here", phrases: phrasesOf("toy-example.md") },
    { href: "/glossary/", title: "AI safety glossary", context: "AI Safety", phrases: phrasesOf("glossary.md") },
    {
      href: "/datasets-catalogue/",
      title: "Dataset catalogue",
      context: "LLM Datasets",
      phrases: phrasesOf("datasets-catalogue.md"),
    },
    { href: "/notebooks/", title: "Notebook guide", context: "Practice", phrases: phrasesOf("notebooks.md") },
    ...allLessons().map((lesson) => ({
      href: lesson.href,
      title: lesson.title,
      context: trackOf(lesson.track)!.title,
      phrases: phrasesOf(lesson.file),
    })),
  ];
}

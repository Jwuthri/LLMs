import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import { Article } from "@/components/Article";
import { CompleteButton, DoneMark } from "@/components/Progress";
import { renderPage } from "@/lib/markdown";
import { allLessons, findLesson, lessonsOf, trackOf } from "@/lib/site";

export const dynamicParams = false;

export function generateStaticParams() {
  return allLessons().map((lesson) => ({ track: lesson.track, slug: lesson.slug }));
}

type Props = { params: Promise<{ track: string; slug: string }> };

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { track, slug } = await params;
  const lesson = findLesson(track, slug);
  return { title: lesson?.title, description: lesson?.summary };
}

export default async function LessonPage({ params }: Props) {
  const { track: trackId, slug } = await params;
  const lesson = findLesson(trackId, slug);
  const track = trackOf(trackId);
  if (!lesson || !track) notFound();

  const page = await renderPage(lesson.file);
  const lessons = lessonsOf(track.id);
  const all = allLessons();
  const index = all.indexOf(lesson);
  const previous = all[index - 1];
  const next = all[index + 1];

  return (
    <Article
      page={page}
      sidebar={
        <nav className="lesson-nav" aria-label={track.title}>
          <Link href={`/${track.id}/`} className="eyebrow">
            {track.title}
          </Link>
          <ol>
            {lessons.map((item) => (
              <li key={item.href}>
                <Link href={item.href} aria-current={item === lesson ? "page" : undefined}>
                  <span className="lesson-nav__number">{item.number}</span>
                  <span className="lesson-nav__title">{item.title}</span>
                  <DoneMark href={item.href} />
                </Link>
              </li>
            ))}
          </ol>
        </nav>
      }
      header={
        <header className="lesson-header">
          <p className="eyebrow">
            <Link href={`/${track.id}/`}>{track.title}</Link> · Lesson {lesson.number} of {lessons.length}
          </p>
          <h1>{lesson.title}</h1>
          <p className="lesson-header__meta">
            {lesson.minutes} min read · {lesson.parts.length} parts
          </p>
        </header>
      }
      footer={
        <footer className="lesson-footer">
          <CompleteButton href={lesson.href} nextHref={next?.href} />
          <nav className="pager" aria-label="Previous and next lesson">
            {previous ? (
              <Link href={previous.href} className="pager__link">
                <span className="eyebrow">← Previous</span>
                <span>{previous.title}</span>
              </Link>
            ) : (
              <span />
            )}
            {next && (
              <Link href={next.href} className="pager__link pager__link--next">
                <span className="eyebrow">Next →</span>
                <span>{next.title}</span>
              </Link>
            )}
          </nav>
        </footer>
      }
    />
  );
}

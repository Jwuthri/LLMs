import type { Metadata } from "next";
import { notFound } from "next/navigation";
import { LessonCard } from "@/components/LessonCard";
import { TrackProgress } from "@/components/Progress";
import { TRACKS, lessonsOf, trackOf } from "@/lib/site";

export const dynamicParams = false;

export function generateStaticParams() {
  return TRACKS.map((track) => ({ track: track.id }));
}

type Props = { params: Promise<{ track: string }> };

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const track = trackOf((await params).track);
  return { title: track?.title, description: track?.tagline };
}

export default async function TrackPage({ params }: Props) {
  const track = trackOf((await params).track);
  if (!track) notFound();
  const lessons = lessonsOf(track.id);
  return (
    <main className="home">
      <section className="track track--page">
        <header className="track__header">
          <div>
            <p className="eyebrow">Track · {lessons.length} lessons</p>
            <h1>{track.title}</h1>
            <p>{track.tagline}</p>
          </div>
          <TrackProgress hrefs={lessons.map((lesson) => lesson.href)} />
        </header>
        <div className="lesson-grid">
          {lessons.map((lesson) => (
            <LessonCard key={lesson.href} lesson={lesson} />
          ))}
        </div>
      </section>
    </main>
  );
}

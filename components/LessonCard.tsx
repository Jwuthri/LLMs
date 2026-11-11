import Link from "next/link";
import { DoneMark } from "./Progress";
import type { LessonMeta } from "@/lib/site";

export function LessonCard({ lesson }: { lesson: LessonMeta }) {
  return (
    <Link href={lesson.href} className="lesson-card">
      <span className="lesson-card__number">{String(lesson.number).padStart(2, "0")}</span>
      <span className="lesson-card__body">
        <span className="lesson-card__title">
          {lesson.title}
          <DoneMark href={lesson.href} />
        </span>
        <span className="lesson-card__summary">{lesson.summary}</span>
        <span className="lesson-card__meta">
          {lesson.minutes} min · {lesson.parts.length} parts
        </span>
      </span>
    </Link>
  );
}

import Link from "next/link";
import { LessonCard } from "@/components/LessonCard";
import { TrackProgress } from "@/components/Progress";
import { TRACKS, lessonsOf } from "@/lib/site";

const METHOD = [
  { title: "Core idea", text: "Two or three sentences, with no special terms." },
  { title: "Setup, steps, key point", text: "Each step shows the calculation, then why you do it." },
  { title: "From toy to real", text: "The same thing at the size of a real model." },
  { title: "Compare and check", text: "A comparison table, then common misconceptions." },
];

// The next-token probabilities of the toy model after "the cat".
const TOY = [
  { token: "the", p: 0.099 },
  { token: "cat", p: 0.242 },
  { token: "sat", p: 0.659 },
];

export default function Home() {
  return (
    <main className="home">
      <section className="hero">
        <div className="hero__text">
          <p className="eyebrow">Twenty lessons · three tracks</p>
          <h1>
            Understand large language models, <em>one small number at a time.</em>
          </h1>
          <p className="hero__lead">
            Each lesson explains one section of the LLM Course with the same tiny model: three words, numbers that
            you can calculate by hand, and the reason for each step.
          </p>
          <div className="hero__actions">
            <Link className="button" href="/toy-example/">
              Start with the toy example
            </Link>
            <a className="button button--quiet" href="#tracks">
              See the lessons
            </a>
          </div>
        </div>
        <figure className="hero__card" aria-label="The toy model gives a probability to each next word after “the cat”.">
          <p className="eyebrow">The toy model</p>
          <p className="hero__prompt">
            the cat <span className="hero__blank">?</span>
          </p>
          <div className="hero__bars">
            {TOY.map((row) => (
              <div key={row.token} className="hero__row">
                <code>{row.token}</code>
                <div className="hero__track">
                  <div className="hero__fill" style={{ width: `${row.p * 100}%` }} />
                </div>
                <span>{row.p.toFixed(3)}</span>
              </div>
            ))}
          </div>
          <figcaption>Scores [0.1, 1.0, 2.0] become probabilities. All lessons use these numbers.</figcaption>
        </figure>
      </section>

      <section className="method" aria-labelledby="method-title">
        <h2 id="method-title">How each lesson works</h2>
        <ol className="method__list">
          {METHOD.map((item, index) => (
            <li key={item.title}>
              <span className="method__number">{String(index + 1).padStart(2, "0")}</span>
              <h3>{item.title}</h3>
              <p>{item.text}</p>
            </li>
          ))}
        </ol>
      </section>

      <div id="tracks">
        {TRACKS.map((track) => {
          const lessons = lessonsOf(track.id);
          return (
            <section key={track.id} className="track" aria-labelledby={`track-${track.id}`}>
              <header className="track__header">
                <div>
                  <h2 id={`track-${track.id}`}>
                    <Link href={`/${track.id}/`}>{track.title}</Link>
                  </h2>
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
          );
        })}
      </div>
    </main>
  );
}

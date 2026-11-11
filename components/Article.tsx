import { Toc } from "./Toc";
import { Widgets } from "./Widgets";
import { ReadingBar } from "./Progress";
import type { RenderedPage } from "@/lib/markdown";

// The reading layout: optional side navigation, the text, and "On this page".
export function Article(props: {
  page: RenderedPage;
  header: React.ReactNode;
  sidebar?: React.ReactNode;
  footer?: React.ReactNode;
}) {
  return (
    <>
      <ReadingBar />
      <div className={"reading" + (props.sidebar ? " reading--with-sidebar" : "")}>
        {props.sidebar && <aside className="reading__sidebar">{props.sidebar}</aside>}
        <main className="reading__main">
          {props.header}
          <article className="prose" dangerouslySetInnerHTML={{ __html: props.page.html }} />
          {props.footer}
        </main>
        <aside className="reading__toc">
          <Toc entries={props.page.toc} />
        </aside>
      </div>
      <Widgets names={props.page.widgets} />
    </>
  );
}

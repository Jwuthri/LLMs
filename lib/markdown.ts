// Markdown -> HTML for one content page. Runs at build time only.
import fs from "node:fs";
import path from "node:path";
import { unified } from "unified";
import remarkParse from "remark-parse";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import remarkRehype from "remark-rehype";
import rehypeRaw from "rehype-raw";
import rehypeKatex from "rehype-katex";
import rehypeSlug from "rehype-slug";
import rehypeHighlight from "rehype-highlight";
import rehypeStringify from "rehype-stringify";
import type { Root as MdRoot, Paragraph } from "mdast";
import type { Element, Root as HtmlRoot, ElementContent } from "hast";
import { CONTENT_DIR, routeOfFile } from "./site";

const FIGURES_DIR = path.join(process.cwd(), "public", "figures");

export interface TocEntry {
  id: string;
  text: string;
  depth: 2 | 3;
}

export interface RenderedPage {
  html: string;
  toc: TocEntry[];
  widgets: string[];
}

// <figure class="diagram" data-src="page/name.svg"> gets the SVG file inline, so that the page CSS styles it.
function inlineDiagrams(markdown: string): string {
  return markdown.replace(/<figure class="diagram" data-src="([^"]+)">/g, (_, src: string) => {
    const svg = fs.readFileSync(path.join(FIGURES_DIR, src), "utf8").trim();
    if (/\n\s*\n/.test(svg)) throw new Error(`${src}: an inline SVG must not have blank lines`);
    return `<figure class="diagram">\n${svg}`;
  });
}

// The pages write a display formula as `$$...$$` on one line. remark-math reads that form as inline math,
// so put the two fences on their own lines.
function fenceDisplayMath(markdown: string): string {
  return markdown.replace(
    /^([ \t]*)\$\$(.+)\$\$[ \t]*$/gm,
    (_, indent: string, formula: string) => `${indent}$$\n${indent}${formula.trim()}\n${indent}$$`
  );
}

function visit<T extends { type: string; children?: unknown[] }>(node: T, test: (n: any, parent: any) => void, parent?: any) {
  test(node, parent);
  for (const child of (node.children ?? []) as any[]) visit(child, test, node);
}

function textOf(node: any): string {
  return node.type === "text" ? node.value : (node.children ?? []).map(textOf).join("");
}

// A paragraph that starts with "**Why:**" is the reason of a step.
function remarkWhy() {
  return (tree: MdRoot) => {
    visit(tree as any, (node) => {
      if (node.type !== "paragraph") return;
      const first = (node as Paragraph).children[0];
      if (first?.type === "strong" && textOf(first) === "Why:") {
        node.data = { hProperties: { className: ["why"] } };
      }
    });
  };
}

function hasClass(node: Element, name: string): boolean {
  const classes = node.properties?.className;
  return Array.isArray(classes) && classes.includes(name);
}

function rehypePage(options: { file: string; toc: TocEntry[]; widgets: string[] }) {
  const dir = path.posix.dirname(options.file);
  return (tree: HtmlRoot) => {
    visit(tree as any, (node: Element, parent: Element) => {
      if (node.type !== "element") return;

      if (node.tagName === "a" && typeof node.properties.href === "string") {
        const href = node.properties.href;
        if (/^https?:/.test(href)) {
          node.properties.target = "_blank";
          node.properties.rel = ["noopener"];
        } else if (!href.startsWith("#") && !href.startsWith("/")) {
          const [target, anchor] = href.split("#");
          const file = path.posix.normalize(path.posix.join(dir, target));
          if (!fs.existsSync(path.join(CONTENT_DIR, file))) throw new Error(`${options.file}: broken link ${href}`);
          node.properties.href = routeOfFile(file) + (anchor ? `#${anchor}` : "");
        }
      }

      if ((node.tagName === "h2" || node.tagName === "h3") && typeof node.properties.id === "string") {
        const text = textOf(node);
        if (node.tagName === "h2") {
          options.toc.push({ id: node.properties.id, text, depth: 2 });
        } else if (["Setup", "Steps", "From toy to real"].includes(text)) {
          node.properties.className = ["stage"]; // styled as a small label, not listed in the table of contents
        } else {
          options.toc.push({ id: node.properties.id, text, depth: 3 });
        }
      }

      if (node.tagName === "div" && typeof node.properties.dataWidget === "string") {
        options.widgets.push(node.properties.dataWidget);
      }

      // A wide table scrolls inside its own box.
      if (node.tagName === "table" && parent && !(parent.type === "element" && hasClass(parent, "table-wrap"))) {
        const index = parent.children.indexOf(node as ElementContent);
        parent.children[index] = {
          type: "element",
          tagName: "div",
          properties: { className: ["table-wrap"] },
          children: [node],
        };
      }
    });
  };
}

export async function renderPage(file: string): Promise<RenderedPage> {
  const source = fs.readFileSync(path.join(CONTENT_DIR, file), "utf8");
  const body = fenceDisplayMath(inlineDiagrams(source.replace(/^# .+\n/, "")));
  const toc: TocEntry[] = [];
  const widgets: string[] = [];
  const result = await unified()
    .use(remarkParse)
    .use(remarkGfm)
    .use(remarkMath)
    .use(remarkWhy)
    .use(remarkRehype, { allowDangerousHtml: true })
    .use(rehypeRaw)
    .use(rehypeKatex)
    .use(rehypeSlug)
    .use(rehypeHighlight, { detect: false })
    .use(rehypePage, { file, toc, widgets })
    .use(rehypeStringify)
    .process(body);
  // A formula that KaTeX cannot read must stop the build, not show red text on the page.
  if (result.messages.length > 0) {
    throw new Error(`${file}: ${result.messages.map((message) => String(message)).join("; ")}`);
  }
  return { html: String(result), toc, widgets: [...new Set(widgets)] };
}

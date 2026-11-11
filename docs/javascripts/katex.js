// Render KaTeX formulas on first load and after each instant navigation.
// pymdownx.arithmatex rewrites `$...$` to `\\(...\\)` and `$$...$$` to `\\[...\\]`,
// so a literal dollar sign in the text or in a diagram is never read as a formula.
document$.subscribe(({ body }) => {
  renderMathInElement(body, {
    delimiters: [
      { left: "\\(", right: "\\)", display: false },
      { left: "\\[", right: "\\]", display: true },
    ],
  });
});

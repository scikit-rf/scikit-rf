// Open links that leave the site in a new tab.
document.addEventListener("DOMContentLoaded", () => {
  for (const a of document.querySelectorAll("a[href]")) {
    if (/^https?:$/.test(a.protocol) && a.host !== location.host) {
      a.target = "_blank";
      a.rel = "noopener";
    }
  }
});

// In the left sidebar, drop a page's own headings (the right sidebar lists those),
// but keep headings that group other pages, such as the sections of the examples and API pages.
document.addEventListener("DOMContentLoaded", () => {
  for (const ul of document.querySelectorAll(".sy-lside .globaltoc li > ul")) {
    const links = [...ul.querySelectorAll("a[href]")];
    if (links.every((a) => a.getAttribute("href").startsWith("#"))) {
      ul.parentElement.classList.remove("_expand");
      ul.remove();
    }
  }
});

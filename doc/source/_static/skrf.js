// Open links that leave the site in a new tab.
document.addEventListener("DOMContentLoaded", () => {
  for (const a of document.querySelectorAll("a[href]")) {
    if (/^https?:$/.test(a.protocol) && a.host !== location.host) {
      a.target = "_blank";
      a.rel = "noopener";
    }
  }
});

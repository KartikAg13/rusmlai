// rusmlai site: Rust highlighting, copy buttons, and the notebook cell on the landing page.
// Every page reads fine without this file; it only adds to what is already there.

const RUST =
  /(\/\/[^\n]*)|("(?:\\.|[^"\\])*")|\b(fn|let|mut|use|pub|struct|impl|for|in|if|else|match|return|mod|as|where|const|static|enum|trait|type|move|ref|true|false|self|Self|crate|Some|None|Ok|Err)\b|\b([A-Z][A-Za-z0-9_]*)\b|\b(\d[\d_]*(?:\.\d+)?)\b|\b([a-z_][a-z0-9_]*!)/g;
const KINDS = ["c", "s", "k", "t", "n", "m"];
const escape = (text) => text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

for (const code of document.querySelectorAll("pre code.rust")) {
  const src = code.textContent;
  let html = "";
  let last = 0;
  for (const match of src.matchAll(RUST)) {
    const kind = KINDS[match.slice(1).findIndex((group) => group !== undefined)];
    html += escape(src.slice(last, match.index)) + `<span class="${kind}">${escape(match[0])}</span>`;
    last = match.index + match[0].length;
  }
  code.innerHTML = html + escape(src.slice(last));
}

async function copy(text, button, label) {
  try {
    await navigator.clipboard.writeText(text);
    label.textContent = "Copied";
  } catch {
    label.textContent = "Select and copy";
  }
  button.dataset.done = "";
  setTimeout(() => {
    label.textContent = "Copy";
    delete button.dataset.done;
  }, 1600);
}

for (const pre of document.querySelectorAll("pre:not([data-nocopy])")) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "copy";
  button.textContent = "Copy";
  button.addEventListener("click", () => copy(pre.querySelector("code").textContent, button, button));
  pre.append(button);
}

for (const button of document.querySelectorAll("[data-copy]")) {
  button.addEventListener("click", () => copy(button.dataset.copy, button, button.querySelector("span")));
}

// The cell runs once when it scrolls into view: read_csv takes its real 120 ms, then the bars
// grow. "Run again" shows the cached path: the stats load in 21 µs and the chart is just there.
const cell = document.querySelector(".cell");
if (cell) {
  const bars = [...cell.querySelectorAll(".bar path")];
  const values = [...cell.querySelectorAll(".value")];
  const status = cell.querySelector(".cell-status");
  const counter = cell.querySelector("[data-run]");
  const still = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const ease = "cubic-bezier(0.16, 1, 0.3, 1)";
  let runs = 0;

  const grow = (delay, stagger, duration) => {
    if (still) return;
    bars.forEach((bar, i) =>
      bar.animate([{ transform: "scaleY(0)" }, { transform: "scaleY(1)" }], {
        duration,
        delay: delay + i * stagger,
        easing: ease,
        fill: "backwards",
      }),
    );
    values.forEach((value, i) =>
      value.animate([{ opacity: 0 }, { opacity: 1 }], {
        duration: 300,
        delay: delay + i * stagger + duration * 0.6,
        fill: "backwards",
      }),
    );
  };

  const run = () => {
    cell.classList.add("ran");
    runs += 1;
    counter.textContent = runs;
    if (runs === 1) {
      status.textContent = "read_csv: parsing 630,000 rows…";
      setTimeout(() => {
        status.innerHTML =
          "<b>120 ms</b> read_csv parsed 630,000 rows, computed stats for 13 columns and saved them to metadata/train.json. count() drew from those stats.";
      }, 120);
      grow(120, 70, 700);
    } else {
      status.innerHTML =
        "<b>62 ms</b> read_csv parsed the file and loaded the stats from metadata/train.json in 21 µs. Nothing was computed again.";
      grow(0, 0, 220);
    }
  };

  cell.querySelector(".run").addEventListener("click", run);
  const seen = new IntersectionObserver(
    (entries) => {
      if (entries.some((entry) => entry.isIntersecting)) {
        seen.disconnect();
        run();
      }
    },
    { threshold: 0.35 },
  );
  seen.observe(cell);
}

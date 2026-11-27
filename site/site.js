// rusmlai site: Rust highlighting, copy buttons, the notebook cell on the landing page, and the sticky notes.
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

// Sticky notes: tabs on the right edge of every page, one set for the whole site, kept in this
// browser's localStorage (no account, nothing sent anywhere). Hovering a tab grows it into a preview;
// clicking grows it into the editor. Like browser tabs, there is always one empty tab at the end,
// never two, and at least three tabs.
const NOTE_KEY = "rusmlai-notes";
const NOTE_COLORS = [
  ["c1", "Olive", "#ffffff"],
  ["c2", "Cream", "#15171c"],
  ["c3", "Rust", "#ffffff"],
  ["c4", "Navy", "#ffffff"],
  ["c5", "Sky", "#15171c"],
  ["c6", "Gold", "#15171c"],
  ["c7", "Plum", "#ffffff"],
  ["c8", "Sage", "#15171c"],
];
const blankNote = (i) => ({ title: "", text: "", color: NOTE_COLORS[i % 8][0] });
const isEmptyNote = (note) => !note.title.trim() && !note.text.trim();

function tidyNotes(list) {
  const notes = list.slice();
  while (notes.length > 3 && isEmptyNote(notes.at(-1)) && isEmptyNote(notes.at(-2))) notes.pop();
  if (notes.length && !isEmptyNote(notes.at(-1))) notes.push(blankNote(notes.length));
  while (notes.length < 3) notes.push(blankNote(notes.length));
  return notes;
}

// The stored list is the reader's own, but it is still checked: strings, capped lengths, known colors.
function loadNotes() {
  try {
    const saved = JSON.parse(localStorage.getItem(NOTE_KEY) || "[]");
    if (!Array.isArray(saved)) return tidyNotes([]);
    const ids = NOTE_COLORS.map(([id]) => id);
    return tidyNotes(
      saved.slice(0, 50).map((note, i) => ({
        title: typeof note?.title === "string" ? note.title.slice(0, 40) : "",
        text: typeof note?.text === "string" ? note.text.slice(0, 1000) : "",
        color: ids.includes(note?.color) ? note.color : blankNote(i).color,
      })),
    );
  } catch {
    return tidyNotes([]);
  }
}

function saveNotes(notes) {
  try {
    localStorage.setItem(NOTE_KEY, JSON.stringify(notes));
    return "Saved";
  } catch {
    return "Not saved: storage is blocked";
  }
}

const noteRail = document.createElement("aside");
noteRail.className = "notes";
noteRail.setAttribute("aria-label", "Your notes");
const noteList = document.createElement("ul");
noteRail.append(noteList);
document.body.append(noteRail);
document.body.classList.add("has-notes");

let notes = loadNotes();
let openNote = -1;
let hoverNote = -1;
let confirmDelete = false;
let noteStatus = "Saved";

function makeTab(i) {
  const li = document.createElement("li");
  li.style.setProperty("--i", i); // staggers the slide-in (style.css, "Motion")
  li.innerHTML =
    '<button class="note-tab" type="button"><span class="note-face"><span></span></span>' +
    '<span class="note-peek" aria-hidden="true"><b></b><span></span><small>Click to edit</small></span></button>';
  const tab = li.querySelector(".note-tab");
  const peek = (on) => {
    hoverNote = on ? i : hoverNote === i ? -1 : hoverNote;
    renderNotes();
  };
  li.addEventListener("mouseenter", () => peek(true));
  li.addEventListener("mouseleave", () => peek(false));
  // Only keyboard focus peeks: focus that follows a click (or Close) would hold the card open.
  tab.addEventListener("focus", () => tab.matches(":focus-visible") && peek(true));
  tab.addEventListener("blur", () => peek(false));
  tab.addEventListener("click", () => {
    openNote = i;
    hoverNote = -1;
    confirmDelete = false;
    renderNotes();
    li.querySelector(".note-editor input").focus({ preventScroll: true });
  });
  li.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && openNote === i) closeNote();
  });
  return li;
}

function makeEditor(i, note) {
  const editor = document.createElement("div");
  editor.className = "note-editor";
  editor.setAttribute("role", "group");
  editor.setAttribute("aria-label", "Edit note");
  editor.innerHTML =
    '<label for="note-title">Title</label><input id="note-title" type="text" maxlength="40">' +
    '<label for="note-text">Note</label><textarea id="note-text" maxlength="1000" rows="6"></textarea>' +
    '<fieldset><legend>Color</legend><div class="note-swatches"></div></fieldset>' +
    '<div class="note-foot"><span aria-live="polite"></span><span class="note-actions">' +
    '<button type="button" data-delete>Delete</button><button type="button" data-close>Close</button></span></div>' +
    "<p>Kept in this browser only. No account, and nothing is sent anywhere.</p>";
  const title = editor.querySelector("input");
  const text = editor.querySelector("textarea");
  title.value = note.title;
  text.value = note.text;
  title.addEventListener("input", () => updateNote(i, { title: title.value }));
  text.addEventListener("input", () => updateNote(i, { text: text.value }));
  const swatches = editor.querySelector(".note-swatches");
  for (const [id, name] of NOTE_COLORS) {
    const swatch = document.createElement("button");
    swatch.type = "button";
    swatch.dataset.color = id;
    swatch.setAttribute("aria-label", name);
    swatch.style.setProperty("--c", `var(--${id})`);
    swatch.addEventListener("click", () => updateNote(i, { color: id }));
    swatches.append(swatch);
  }
  editor.querySelector("[data-delete]").addEventListener("click", () => deleteNote(i));
  editor.querySelector("[data-close]").addEventListener("click", closeNote);
  return editor;
}

function renderNotes() {
  while (noteList.children.length < notes.length) noteList.append(makeTab(noteList.children.length));
  while (noteList.children.length > notes.length) noteList.lastElementChild.remove();
  const height = Math.max(44, Math.min(140, Math.floor((560 - 4 * (notes.length - 1)) / notes.length)));
  notes.forEach((note, i) => {
    const li = noteList.children[i];
    const fg = NOTE_COLORS.find(([id]) => id === note.color)[2];
    const label = note.title.trim() || `Note ${i + 1}`;
    li.dataset.state = i === openNote ? "edit" : openNote < 0 && i === hoverNote ? "peek" : "closed";
    li.classList.toggle("empty", isEmptyNote(note));
    li.style.setProperty("--h", `${height}px`);
    li.style.setProperty("--bg", `var(--${note.color})`);
    li.style.setProperty("--fg", fg);
    const tab = li.querySelector(".note-tab");
    tab.setAttribute("aria-label", (isEmptyNote(note) ? "Empty note: " : "Note: ") + label);
    tab.setAttribute("aria-expanded", String(i === openNote));
    li.querySelector(".note-face span").textContent = label;
    li.querySelector(".note-peek b").textContent = label;
    li.querySelector(".note-peek span").textContent = note.text.trim() || "Empty. Click to write.";
    let editor = li.querySelector(".note-editor");
    if (i !== openNote) {
      editor?.remove();
      return;
    }
    if (!editor) li.append((editor = makeEditor(i, note)));
    editor.querySelector(".note-foot span").textContent = `${note.text.length} / 1000 · ${noteStatus}`;
    editor.querySelector("[data-delete]").textContent = confirmDelete ? "Delete it?" : "Delete";
    for (const swatch of editor.querySelectorAll("[data-color]")) {
      swatch.setAttribute("aria-pressed", String(swatch.dataset.color === note.color));
    }
  });
}

function updateNote(i, patch) {
  const list = notes.slice();
  list[i] = { ...list[i], ...patch };
  notes = tidyNotes(list);
  noteStatus = saveNotes(notes);
  confirmDelete = false;
  renderNotes();
}

function deleteNote(i) {
  if (!confirmDelete) {
    confirmDelete = true;
    renderNotes();
    return;
  }
  const list = notes.slice();
  list.splice(i, 1);
  notes = tidyNotes(list);
  noteStatus = saveNotes(notes);
  closeNote();
}

function closeNote() {
  const tab = noteList.children[openNote]?.querySelector(".note-tab");
  openNote = -1;
  hoverNote = -1;
  confirmDelete = false;
  renderNotes();
  tab?.focus({ preventScroll: true });
}

// Notes written in another browser tab show up here once this one has no note open.
addEventListener("storage", (event) => {
  if (event.key === NOTE_KEY && openNote < 0) {
    notes = loadNotes();
    renderNotes();
  }
});

renderNotes();

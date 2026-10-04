//! What users get back: a chart (`Figure`) or a table (`TableView`).
//! Both print in a terminal and render inline in evcxr notebooks.

use std::{
    cell::Cell,
    ffi::OsStr,
    fmt, fs, io,
    path::{Path, PathBuf},
    process::Command,
    sync::atomic::{AtomicUsize, Ordering},
};

use comfy_table::{Table, presets::UTF8_FULL};

use crate::{
    error::{OrFatal, warn},
    metadata::{constants::CACHE_DIRECTORY, utils::print_saved},
    plot::{chart::Chart, style::Style},
};

/// A chart. When it goes out of scope it shows itself in a notebook if nothing displayed it.
/// In a program it saves itself to `metadata/plots/{title}.svg` (next to the CSV) unless
/// `.save` or `.svg` was called.
pub struct Figure {
    pub(crate) chart: Chart,
    pub(crate) style: Style,
    folder: PathBuf,
    displayed: Cell<bool>,
    saved: Cell<bool>,
}

impl Figure {
    pub(crate) fn new(chart: Chart, csv: Option<&OsStr>) -> Self {
        let dir: &Path = csv
            .and_then(|path| Path::new(path).parent())
            .unwrap_or(Path::new(""));
        Self {
            chart,
            style: Style::default(),
            folder: dir.join(CACHE_DIRECTORY).join("plots"),
            displayed: Cell::new(false),
            saved: Cell::new(false),
        }
    }

    /// The chart as SVG, with its options on top of `set_theme` / `set_palette`.
    /// Taking the text counts as handling the chart, so it is not auto-saved.
    pub fn svg(&self) -> String {
        self.saved.set(true);
        self.render()
    }

    /// Writes the SVG file (creating folders) in place of the auto-save. In a notebook it
    /// also shows the chart. Never opens a viewer, so it is safe in loops.
    #[track_caller]
    pub fn save(&self, path: impl AsRef<Path>) {
        self.write(path.as_ref()).or_fatal();
        self.saved.set(true);
        if in_notebook() {
            self.evcxr_display();
        }
    }

    /// Notebook: shows the chart inline. Program: writes a temp SVG and opens the system viewer.
    #[track_caller]
    pub fn show(&self) {
        if in_notebook() {
            return self.evcxr_display();
        }
        static SHOWN: AtomicUsize = AtomicUsize::new(0);
        let n: usize = SHOWN.fetch_add(1, Ordering::Relaxed);
        let path: PathBuf =
            std::env::temp_dir().join(format!("rusmlai-{}-{n}.svg", std::process::id()));
        self.write(&path).or_fatal();
        let opener: &str = if cfg!(target_os = "macos") {
            "open"
        } else {
            "xdg-open"
        };
        let _ = Command::new(opener).arg(&path).spawn();
    }

    /// evcxr calls this on a cell's last expression. Inline HTML (not image/svg+xml)
    /// keeps the `<title>` hover tooltips working.
    pub fn evcxr_display(&self) {
        self.displayed.set(true);
        println!(
            "EVCXR_BEGIN_CONTENT text/html\n{}\nEVCXR_END_CONTENT",
            self.render()
        );
    }

    fn render(&self) -> String {
        self.chart.render(&self.style.resolved())
    }

    fn write(&self, path: &Path) -> Result<(), String> {
        let fail = |e: io::Error| format!("could not save {}: {e}", path.display());
        if let Some(dir) = path.parent().filter(|dir| !dir.as_os_str().is_empty()) {
            fs::create_dir_all(dir).map_err(fail)?;
        }
        fs::write(path, self.render()).map_err(fail)?;
        print_saved(path);
        Ok(())
    }
}

impl Drop for Figure {
    fn drop(&mut self) {
        if std::thread::panicking() {
            return; // a failed option (`.order(&["typo"])`) must not draw or write anything
        }
        if in_notebook() && !self.displayed.get() {
            self.evcxr_display();
        }
        if !in_notebook() && !self.saved.get() {
            let name: String = self
                .chart
                .title(&self.style.resolved())
                .chars()
                .filter(|c| c.is_alphanumeric() || " -_.".contains(*c))
                .collect();
            if let Err(e) = self.write(&self.folder.join(format!("{name}.svg"))) {
                warn(e);
            }
        }
    }
}

fn in_notebook() -> bool {
    std::env::var_os("EVCXR_IS_RUNTIME").is_some()
}

impl fmt::Display for Figure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "<Figure: {}>", self.chart.name())
    }
}

/// A table: comfy-table text in a terminal, an HTML table in a notebook.
pub struct TableView {
    text: String,
    html: String,
}

impl TableView {
    pub fn new(caption: &str, headers: &[String], rows: &[Vec<String>]) -> Self {
        let mut table: Table = Table::new();
        table.load_style(UTF8_FULL);
        table.set_header(headers);
        for row in rows {
            table.add_row(row);
        }

        let cells = |tag: &str, row: &[String]| -> String {
            row.iter()
                .map(|c| {
                    format!(
                        "<{tag} style=\"text-align:left;vertical-align:top\">{}</{tag}>",
                        escape(c).replace('\n', "<br>")
                    )
                })
                .collect()
        };
        let body: String = rows
            .iter()
            .map(|r| format!("<tr>{}</tr>", cells("td", r)))
            .collect();
        let html: String = format!(
            "<table style=\"background:#15171C;color:#F2F3F5;border-collapse:collapse\">\
             <caption style=\"text-align:left;padding-bottom:1em\">{}</caption><thead style=\"color:#8B8F9A\"><tr>{}</tr></thead>\
             <tbody>{body}</tbody></table>",
            escape(caption).replace('\n', "<br>"),
            cells("th", headers)
        );
        Self {
            text: if caption.is_empty() {
                table.to_string()
            } else {
                format!("{caption}\n\n{table}")
            },
            html,
        }
    }

    /// Prints the table: HTML inside an evcxr notebook, text everywhere else.
    pub fn show(&self) {
        if in_notebook() {
            self.evcxr_display();
        } else {
            println!("{self}");
        }
    }

    pub fn evcxr_display(&self) {
        println!(
            "EVCXR_BEGIN_CONTENT text/html\n{}\nEVCXR_END_CONTENT",
            self.html
        );
    }
}

impl fmt::Display for TableView {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.text)
    }
}

/// Escapes text for HTML and SVG. The chart renderer uses it too.
pub(crate) fn escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

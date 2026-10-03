//! What users get back: a chart (`Figure`) or a table (`TableView`).
//! Both print in a terminal and render inline in evcxr notebooks.

use std::{
    fmt, fs,
    path::{Path, PathBuf},
    process::Command,
    sync::atomic::{AtomicUsize, Ordering},
};

use comfy_table::{Table, presets::UTF8_FULL};

use crate::error::OrFatal;

#[must_use = "call .show() or .save(path) to see the plot"]
pub struct Figure {
    svg: String,
}

impl Figure {
    pub fn new(svg: String) -> Self {
        Self { svg }
    }

    pub fn svg(&self) -> &str {
        &self.svg
    }

    /// Writes the SVG file. Never opens a viewer, so it is safe in loops.
    #[track_caller]
    pub fn save(&self, path: impl AsRef<Path>) {
        let path: &Path = path.as_ref();
        fs::write(path, &self.svg)
            .map_err(|e| format!("could not save {}: {e}", path.display()))
            .or_fatal();
    }

    /// Notebook: shows the chart inline. Program: writes a temp SVG and opens the system viewer.
    #[track_caller]
    pub fn show(&self) {
        if std::env::var_os("EVCXR_IS_RUNTIME").is_some() {
            return self.evcxr_display();
        }
        static SHOWN: AtomicUsize = AtomicUsize::new(0);
        let n: usize = SHOWN.fetch_add(1, Ordering::Relaxed);
        let path: PathBuf =
            std::env::temp_dir().join(format!("rusmlai-{}-{n}.svg", std::process::id()));
        self.save(&path);
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
        println!(
            "EVCXR_BEGIN_CONTENT text/html\n{}\nEVCXR_END_CONTENT",
            self.svg
        );
    }
}

impl fmt::Display for Figure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "<Figure {} KB>", self.svg.len().div_ceil(1024))
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
        if std::env::var_os("EVCXR_IS_RUNTIME").is_some() {
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

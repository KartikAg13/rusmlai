//! What users get back: a chart (`Figure`) or a table (`TableView`).
//! Both print in a terminal and render inline in evcxr notebooks.

use std::{
    cell::Cell,
    ffi::OsStr,
    fmt, fs, io,
    path::{Path, PathBuf},
    process::Command,
    sync::{Arc, OnceLock},
};

use comfy_table::{Table, presets::UTF8_FULL};
use resvg::{
    tiny_skia::{IntSize, Pixmap, Transform},
    usvg::{self, fontdb},
};

use crate::{
    error::{OrFatal, warn},
    metadata::{constants::CACHE_DIRECTORY, utils::print_saved},
    plot::{chart::Chart, style::Style},
};

/// PNGs are drawn at twice the chart's size, so they stay sharp on high-DPI screens.
const PNG_SCALE: f32 = 2.0;
/// A bigger card (`.size`, many categories) gets a smaller scale: 40 M pixels are 160 MB.
const MAX_PNG_PIXELS: f32 = 40e6;

/// A chart. When it goes out of scope it shows itself in a notebook if nothing displayed it.
/// In a program it saves itself to `metadata/plots/{title}.png` (next to the CSV) unless
/// `.save` or `.svg` was called. A PNG opens as an image in any editor; an SVG opens as text.
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

    /// Writes the chart (creating folders) in place of the auto-save: a PNG for `.png`, the SVG
    /// for `.svg`, like matplotlib's `savefig`. In a notebook it also shows the chart. Never
    /// opens a viewer, so it is safe in loops.
    ///
    /// # Panics
    ///
    /// If the extension is not `.png` or `.svg`, or the file cannot be written.
    ///
    /// # Examples
    ///
    /// Basic:
    /// ```
    /// # use rusmlai::prelude::*;
    /// # let dir = std::env::temp_dir().join("rusmlai-save-doc");
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// ds.hist("cgpa", "", 10).save(dir.join("cgpa.png"));
    /// # assert!(dir.join("cgpa.png").is_file());
    /// ```
    ///
    /// Python:
    ///
    /// ```python
    /// import matplotlib.pyplot as plt
    /// import pandas as pd
    /// import seaborn as sns
    /// ds = pd.read_csv("tests/fixtures/data/students.csv")
    /// import os, tempfile
    /// dir = tempfile.mkdtemp()
    /// ax = sns.histplot(data=ds, x="cgpa", bins=10)
    /// ax.figure.savefig(os.path.join(dir, "cgpa.png"))
    /// assert os.path.isfile(os.path.join(dir, "cgpa.png"))
    /// ```
    ///
    /// Advanced: one SVG per column, in a loop:
    /// ```
    /// # use rusmlai::prelude::*;
    /// # let dir = std::env::temp_dir().join("rusmlai-save-doc");
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// for column in ["age", "cgpa"] {
    ///     ds.hist(column, "", 10).title(column).save(dir.join(format!("{column}.svg")));
    /// }
    /// # assert!(std::fs::read_to_string(dir.join("age.svg")).unwrap().starts_with("<svg"));
    /// ```
    ///
    /// Edge case: another extension panics instead of guessing a format:
    /// ```should_panic
    /// # use rusmlai::prelude::*;
    /// read_csv("tests/fixtures/data/students.csv").hist("cgpa", "", 10).save("cgpa.jpg"); // panics: name the file .png or .svg
    /// ```
    #[track_caller]
    pub fn save(&self, path: impl AsRef<Path>) {
        self.write(path.as_ref()).or_fatal();
        self.saved.set(true);
        if in_notebook() {
            self.evcxr_display();
        }
    }

    /// Notebook: shows the chart inline. Program: saves it where the auto-save would
    /// (`metadata/plots/{title}.png`) and opens it in the system viewer (`xdg-open` or
    /// `open`). (matplotlib: `plt.show()`)
    ///
    /// # Panics
    ///
    /// If the PNG cannot be written.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use rusmlai::prelude::*;
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// ds.countplot("batch", "").show();
    /// ```
    ///
    /// Python:
    ///
    /// ```python
    /// import matplotlib.pyplot as plt
    /// import pandas as pd
    /// import seaborn as sns
    /// ds = pd.read_csv("tests/fixtures/data/students.csv")
    /// sns.countplot(data=ds, x="batch")
    /// plt.show()
    /// ```
    ///
    /// In a loop, prefer [`Figure::save`]: `show` opens one viewer window per chart.
    #[track_caller]
    pub fn show(&self) {
        if in_notebook() {
            return self.evcxr_display();
        }
        let path: PathBuf = self.auto_path();
        self.write(&path).or_fatal();
        self.saved.set(true);
        open_in_viewer(&path);
    }

    /// evcxr calls this on a cell's last expression. Inline HTML (not image/svg+xml)
    /// keeps the `<title>` hover tooltips working.
    ///
    /// Outside a notebook it prints the HTML block to stdout, which is rarely what you want;
    /// use [`Figure::svg`] for the text.
    ///
    /// ```
    /// # use rusmlai::prelude::*;
    /// let fig = read_csv("tests/fixtures/data/students.csv").countplot("batch", "");
    /// fig.evcxr_display(); // prints EVCXR_BEGIN_CONTENT text/html ... EVCXR_END_CONTENT
    /// # let _ = fig.svg();
    /// ```
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

    /// `metadata/plots/{title}.png` next to the CSV.
    fn auto_path(&self) -> PathBuf {
        let name: String = self
            .chart
            .texts
            .title(&self.style.resolved())
            .chars()
            .filter(|c| c.is_alphanumeric() || " -_.".contains(*c))
            .collect();
        // A title of only symbols or dots would make a hidden ".png" with no name.
        let name: &str = match name.trim_matches(['.', ' ']) {
            "" => self.chart.name,
            name => name,
        };
        self.folder.join(format!("{name}.png"))
    }

    fn write(&self, path: &Path) -> Result<(), String> {
        let fail = |e: String| format!("could not save {}: {e}", path.display());
        let bytes: Vec<u8> = match path.extension().and_then(OsStr::to_str) {
            Some(ext) if ext.eq_ignore_ascii_case("svg") => self.render().into_bytes(),
            Some(ext) if ext.eq_ignore_ascii_case("png") => png(&self.render()).map_err(fail)?,
            _ => return Err(fail("name the file .png or .svg".to_string())),
        };
        if let Some(dir) = path.parent().filter(|dir| !dir.as_os_str().is_empty()) {
            fs::create_dir_all(dir).map_err(|e: io::Error| fail(e.to_string()))?;
        }
        fs::write(path, bytes).map_err(|e: io::Error| fail(e.to_string()))?;
        print_saved(path);
        Ok(())
    }
}

/// The SVG drawn at `PNG_SCALE` times its size with the system's fonts.
fn png(svg: &str) -> Result<Vec<u8>, String> {
    let options: usvg::Options = usvg::Options {
        fontdb: fonts(),
        ..usvg::Options::default()
    };
    let tree: usvg::Tree = usvg::Tree::from_str(svg, &options).map_err(|e| e.to_string())?;
    let (w, h): (f32, f32) = (tree.size().width(), tree.size().height());
    let scale: f32 = PNG_SCALE.min((MAX_PNG_PIXELS / (w * h)).sqrt());
    let size: IntSize = tree
        .size()
        .to_int_size()
        .scale_by(scale)
        .ok_or("the chart is too large for a PNG")?;
    let mut pixmap: Pixmap =
        Pixmap::new(size.width(), size.height()).ok_or("the chart is too large for a PNG")?;
    resvg::render(
        &tree,
        Transform::from_scale(scale, scale),
        &mut pixmap.as_mut(),
    );
    // Fast compression: tiny-skia's `encode_png` level takes about 0.4 s per chart in a debug
    // build, which is how most programs run. The card is opaque, so the premultiplied pixels
    // are the plain ones and need no copy.
    let mut bytes: Vec<u8> = Vec::new();
    let mut encoder: png::Encoder<&mut Vec<u8>> =
        png::Encoder::new(&mut bytes, size.width(), size.height());
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_compression(png::Compression::Fast);
    let mut writer: png::Writer<&mut Vec<u8>> =
        encoder.write_header().map_err(|e| e.to_string())?;
    writer
        .write_image_data(pixmap.data())
        .map_err(|e| e.to_string())?;
    writer.finish().map_err(|e| e.to_string())?;
    Ok(bytes)
}

/// The system fonts, loaded once per program: scanning them takes longer than drawing a chart.
/// A read-only cache, so evcxr's per-cell copy of this static is harmless (unlike a setting).
fn fonts() -> Arc<fontdb::Database> {
    static FONTS: OnceLock<Arc<fontdb::Database>> = OnceLock::new();
    FONTS
        .get_or_init(|| {
            let mut db: fontdb::Database = fontdb::Database::new();
            db.load_system_fonts();
            // fontdb takes "sans-serif" to mean Arial, which Linux rarely has.
            let sans: Option<&str> = [
                "Noto Sans",
                "DejaVu Sans",
                "Liberation Sans",
                "Arial",
                "Helvetica",
            ]
            .into_iter()
            .find(|name| {
                db.faces()
                    .any(|face| face.families.iter().any(|(family, _)| family == name))
            });
            match sans {
                Some(name) => db.set_sans_serif_family(name),
                None => warn("no sans-serif font found, so PNG charts have no text; save as .svg"),
            }
            Arc::new(db)
        })
        .clone()
}

impl Drop for Figure {
    fn drop(&mut self) {
        if std::thread::panicking() {
            return; // a failed option (`.order(&["typo"])`) must not draw or write anything
        }
        if in_notebook() && !self.displayed.get() {
            self.evcxr_display();
        }
        if !in_notebook()
            && !self.saved.get()
            && let Err(e) = self.write(&self.auto_path())
        {
            warn(e);
        }
    }
}

pub(crate) fn in_notebook() -> bool {
    std::env::var_os("EVCXR_IS_RUNTIME").is_some()
}

/// Opens `path` in the system's default app (`xdg-open`, or `open` on macOS), without waiting.
pub(crate) fn open_in_viewer(path: &Path) {
    let opener: &str = if cfg!(target_os = "macos") {
        "open"
    } else {
        "xdg-open"
    };
    let _ = Command::new(opener).arg(path).spawn();
}

impl fmt::Display for Figure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "<Figure: {}>", self.chart.name)
    }
}

/// A table: comfy-table text in a terminal, an HTML table in a notebook.
///
/// `info`, `describe`, `metadata`, `head` and `tail` print one. The `try_*` versions on
/// [`Metadata`](crate::metadata::Metadata) return it, so tests can read the text through
/// `Display` (`to_string()`).
///
/// # Examples
///
/// Basic: build and print one:
/// ```
/// use rusmlai::prelude::*;
///
/// let headers = ["model", "accuracy"].map(String::from);
/// let rows = vec![vec!["tree".to_string(), "0.91".to_string()]];
/// TableView::new("Scores", &headers, &rows).show();
/// ```
///
/// Advanced: the text has the caption, a blank line, then the table:
/// ```
/// # use rusmlai::prelude::*;
/// let ds = read_csv("tests/fixtures/data/students.csv");
/// let text: String = ds.metadata.try_info().to_string();
/// assert!(text.starts_with("Dataset: 12 rows x 6 columns\n\n"));
/// ```
///
/// Edge case: cells with newlines and HTML characters are safe:
/// ```
/// # use rusmlai::prelude::*;
/// let table = TableView::new("", &["a".to_string()], &[vec!["<b>\nx".to_string()]]);
/// assert!(table.to_string().contains("<b>"));
/// ```
pub struct TableView {
    text: String,
    html: String,
}

impl TableView {
    /// A table with a caption (`""` for none) above it. Rows shorter than `headers` leave
    /// cells empty. A `\n` in a cell starts a new line in the cell.
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

    /// Prints the HTML table for evcxr. [`TableView::show`] calls it in a notebook.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn escape_covers_html_characters() {
        assert_eq!(escape(r#"a & <b> "c""#), "a &amp; &lt;b&gt; &quot;c&quot;");
    }

    #[test]
    fn table_text_has_caption_then_blank_line_and_html_styles_each_cell() {
        let table: TableView =
            TableView::new("Cap\ntwo", &["h".to_string()], &[vec!["1\n2".into()]]);
        assert!(table.to_string().starts_with("Cap\ntwo\n\n┌"));
        assert!(table.html.contains("Cap<br>two") && table.html.contains("1<br>2"));
        let cells: usize =
            table.html.matches("<td style=").count() + table.html.matches("<th style=").count();
        assert_eq!(cells, 2);
        assert!(
            !TableView::new("", &["h".to_string()], &[])
                .to_string()
                .starts_with('\n')
        );
    }

    #[test]
    fn auto_save_name_keeps_safe_characters() {
        let fig: Figure = crate::dataset::Dataset::new(polars::df!("a/b" => ["x"]).unwrap(), "")
            .countplot("a/b", "")
            .title("../a/b: c?");
        let _ = fig.svg(); // no auto-save
        assert_eq!(fig.auto_path(), Path::new("metadata/plots/ab c.png"));
        let dots: Figure = crate::dataset::Dataset::new(polars::df!("x" => ["y"]).unwrap(), "")
            .countplot("x", "")
            .title("...");
        let _ = dots.svg();
        assert_eq!(dots.auto_path().file_name().unwrap(), "countplot.png");
    }

    #[test]
    fn png_is_twice_the_size_and_draws_text() {
        let chart = |text: &str| {
            format!(
                r#"<svg width="30" height="20" xmlns="http://www.w3.org/2000/svg"><rect width="30" height="20" fill="white"/><text x="2" y="14" font-family="sans-serif" font-size="12">{text}</text></svg>"#
            )
        };
        let bytes: Vec<u8> = png(&chart("Ab")).unwrap();
        assert_eq!(&bytes[..8], b"\x89PNG\r\n\x1a\n");
        let ihdr = |at: usize| u32::from_be_bytes(bytes[at..at + 4].try_into().unwrap());
        assert_eq!((ihdr(16), ihdr(20)), (60, 40));
        assert_ne!(bytes, png(&chart("")).unwrap(), "no font drew the text");
    }
}

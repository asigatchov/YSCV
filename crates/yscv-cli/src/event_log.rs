use std::fs;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use crate::error::AppError;
use crate::util::ensure_parent_dir;

#[derive(Debug)]
pub struct JsonlEventWriter {
    path: PathBuf,
    sink: BufWriter<fs::File>,
}

impl JsonlEventWriter {
    pub fn create(path: &Path) -> Result<Self, AppError> {
        ensure_parent_dir(path)?;
        let file = fs::File::create(path)?;
        Ok(Self {
            path: path.to_path_buf(),
            sink: BufWriter::new(file),
        })
    }

    pub fn write_record(&mut self, value: &serde_json::Value) -> Result<(), AppError> {
        serde_json::to_writer(&mut self.sink, value)?;
        self.sink.write_all(b"\n")?;
        Ok(())
    }

    pub fn flush(&mut self) -> Result<(), std::io::Error> {
        self.sink.flush()
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

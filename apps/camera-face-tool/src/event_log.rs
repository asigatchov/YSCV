use std::fs;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use crate::error::AppError;

#[derive(Debug)]
pub(crate) struct JsonlEventWriter {
    path: PathBuf,
    sink: BufWriter<fs::File>,
}

impl JsonlEventWriter {
    pub(crate) fn create(path: &Path) -> Result<Self, AppError> {
        ensure_parent_dir(path)?;
        let file = fs::File::create(path)?;
        Ok(Self {
            path: path.to_path_buf(),
            sink: BufWriter::new(file),
        })
    }

    pub(crate) fn write_record(&mut self, value: &serde_json::Value) -> Result<(), AppError> {
        serde_json::to_writer(&mut self.sink, value)?;
        self.sink.write_all(b"\n")?;
        Ok(())
    }

    pub(crate) fn flush(&mut self) -> Result<(), std::io::Error> {
        self.sink.flush()
    }

    pub(crate) fn path(&self) -> &Path {
        &self.path
    }
}

fn ensure_parent_dir(path: &Path) -> Result<(), std::io::Error> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }
    Ok(())
}

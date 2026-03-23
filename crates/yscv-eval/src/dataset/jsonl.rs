use crate::EvalError;

pub(crate) fn parse_dataset_jsonl<T, F>(text: &str, mut parse_line: F) -> Result<Vec<T>, EvalError>
where
    F: FnMut(&str, usize) -> Result<T, EvalError>,
{
    let mut out = Vec::new();
    for (line_idx, raw_line) in text.lines().enumerate() {
        let line_no = line_idx + 1;
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        out.push(parse_line(line, line_no)?);
    }
    Ok(out)
}

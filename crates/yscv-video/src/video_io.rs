use std::collections::HashMap;
use std::io::{Read, Write};
use std::path::Path;

use super::error::VideoError;
use super::frame::Rgb8Frame;

/// Video container metadata.
#[derive(Debug, Clone)]
pub struct VideoMeta {
    pub width: u32,
    pub height: u32,
    pub frame_count: u32,
    pub fps: f32,
    pub properties: HashMap<String, String>,
}

/// Reads raw RGB8 frames from a simple uncompressed video file.
///
/// Format: [8 bytes: magic "RCVVIDEO"] [4 bytes: width LE] [4 bytes: height LE] [4 bytes: frame_count LE] [4 bytes: fps as f32 LE bits] [frames: width*height*3 bytes each].
pub struct RawVideoReader {
    pub meta: VideoMeta,
    data: Vec<u8>,
    frame_offset: usize,
    current_frame: u32,
}

const MAGIC: &[u8; 8] = b"RCVVIDEO";

impl RawVideoReader {
    /// Opens a raw video file for reading.
    pub fn open(path: &Path) -> Result<Self, VideoError> {
        let mut file = std::fs::File::open(path)
            .map_err(|e| VideoError::Source(format!("{}: {e}", path.display())))?;
        let mut data = Vec::new();
        file.read_to_end(&mut data)
            .map_err(|e| VideoError::Source(e.to_string()))?;

        if data.len() < 24 || &data[..8] != MAGIC {
            return Err(VideoError::Source("invalid raw video file header".into()));
        }

        let width = u32::from_le_bytes([data[8], data[9], data[10], data[11]]);
        let height = u32::from_le_bytes([data[12], data[13], data[14], data[15]]);
        let frame_count = u32::from_le_bytes([data[16], data[17], data[18], data[19]]);
        let fps = f32::from_le_bytes([data[20], data[21], data[22], data[23]]);

        Ok(Self {
            meta: VideoMeta {
                width,
                height,
                frame_count,
                fps,
                properties: HashMap::new(),
            },
            data,
            frame_offset: 24,
            current_frame: 0,
        })
    }

    /// Reads the next frame, if available.
    pub fn next_frame(&mut self) -> Option<Rgb8Frame> {
        if self.current_frame >= self.meta.frame_count {
            return None;
        }
        let frame_size = self.meta.width as usize * self.meta.height as usize * 3;
        let start = self.frame_offset + self.current_frame as usize * frame_size;
        let end = start + frame_size;
        if end > self.data.len() {
            return None;
        }
        self.current_frame += 1;
        Rgb8Frame::from_bytes(
            self.current_frame as u64 - 1,
            0,
            self.meta.width as usize,
            self.meta.height as usize,
            bytes::Bytes::copy_from_slice(&self.data[start..end]),
        )
        .ok()
    }

    /// Resets to the beginning.
    pub fn seek_start(&mut self) {
        self.current_frame = 0;
    }

    /// Returns the frame count.
    pub fn frame_count(&self) -> u32 {
        self.meta.frame_count
    }
}

/// Writes raw RGB8 frames to a simple uncompressed video file.
pub struct RawVideoWriter {
    width: u32,
    height: u32,
    fps: f32,
    frames: Vec<Vec<u8>>,
}

impl RawVideoWriter {
    pub fn new(width: u32, height: u32, fps: f32) -> Self {
        Self {
            width,
            height,
            fps,
            frames: Vec::new(),
        }
    }

    /// Appends an RGB8 frame (must be width*height*3 bytes).
    pub fn push_frame(&mut self, rgb8_data: &[u8]) -> Result<(), VideoError> {
        let expected = self.width as usize * self.height as usize * 3;
        if rgb8_data.len() != expected {
            return Err(VideoError::Source(format!(
                "frame size mismatch: expected {expected}, got {}",
                rgb8_data.len()
            )));
        }
        self.frames.push(rgb8_data.to_vec());
        Ok(())
    }

    /// Writes the video to a file.
    pub fn save(&self, path: &Path) -> Result<(), VideoError> {
        let mut file = std::fs::File::create(path)
            .map_err(|e| VideoError::Source(format!("{}: {e}", path.display())))?;

        let wr = |f: &mut std::fs::File, d: &[u8]| -> Result<(), VideoError> {
            f.write_all(d)
                .map_err(|e| VideoError::Source(e.to_string()))
        };
        wr(&mut file, MAGIC)?;
        wr(&mut file, &self.width.to_le_bytes())?;
        wr(&mut file, &self.height.to_le_bytes())?;
        wr(&mut file, &(self.frames.len() as u32).to_le_bytes())?;
        wr(&mut file, &self.fps.to_le_bytes())?;

        for frame in &self.frames {
            wr(&mut file, frame)?;
        }

        Ok(())
    }

    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }
}

/// Image sequence reader: reads numbered image files as a video stream.
///
/// Pattern example: `frames/frame_%04d.png` -> reads `frame_0000.png`, `frame_0001.png`, etc.
pub struct ImageSequenceReader {
    pub width: usize,
    pub height: usize,
    paths: Vec<std::path::PathBuf>,
    current: usize,
}

impl ImageSequenceReader {
    /// Creates a reader from a sorted list of image file paths.
    pub fn from_paths(paths: Vec<std::path::PathBuf>) -> Self {
        Self {
            width: 0,
            height: 0,
            paths,
            current: 0,
        }
    }

    /// Returns the total number of frames.
    pub fn frame_count(&self) -> usize {
        self.paths.len()
    }

    /// Resets to the beginning.
    pub fn seek_start(&mut self) {
        self.current = 0;
    }

    /// Returns the next image path without loading.
    pub fn next_path(&mut self) -> Option<&Path> {
        if self.current >= self.paths.len() {
            return None;
        }
        let path = &self.paths[self.current];
        self.current += 1;
        Some(path)
    }
}

// ---------------------------------------------------------------------------
// MP4 / H.264 Video Reader
// ---------------------------------------------------------------------------

/// Reads H.264-encoded MP4 video files and decodes frames to RGB8.
///
/// Combines the MP4 box parser, Annex B NAL extraction, and H.264 decoder
/// into a single end-to-end reader.
///
/// ```ignore
/// let mut reader = Mp4VideoReader::open("input.mp4")?;
/// while let Some(frame) = reader.next_frame()? {
///     // frame.rgb8_data is Vec<u8>, frame.width / frame.height
/// }
/// ```
pub struct Mp4VideoReader {
    decoder: super::h264_decoder::H264Decoder,
    nal_units: Vec<super::codec::NalUnit>,
    current_nal: usize,
}

impl Mp4VideoReader {
    /// Open an MP4 file containing H.264 video.
    ///
    /// Reads the entire file, parses MP4 boxes to find the `mdat` box,
    /// and extracts NAL units from the raw data.
    pub fn open(path: &Path) -> Result<Self, VideoError> {
        let data = std::fs::read(path)
            .map_err(|e| VideoError::Source(format!("{}: {e}", path.display())))?;

        // Parse MP4 boxes to find mdat (media data)
        let boxes = super::codec::parse_mp4_boxes(&data)?;
        let mdat = boxes
            .iter()
            .find(|b| b.type_str() == "mdat")
            .ok_or_else(|| VideoError::ContainerParse("no mdat box found in MP4".into()))?;

        let mdat_start = (mdat.offset + mdat.header_size as u64) as usize;
        let mdat_end = (mdat.offset + mdat.size) as usize;
        if mdat_end > data.len() {
            return Err(VideoError::ContainerParse(
                "mdat box extends past EOF".into(),
            ));
        }

        let mdat_data = &data[mdat_start..mdat_end];

        // Try Annex B format first (start codes 0x000001 / 0x00000001)
        let mut nal_units = super::codec::parse_annex_b(mdat_data);

        // If no NAL units found with Annex B, try length-prefixed (AVCC) format
        if nal_units.is_empty() {
            nal_units = parse_avcc_nals(mdat_data);
        }

        // Also check for SPS/PPS in moov/stsd boxes (common in MP4)
        if let Some(moov) = boxes.iter().find(|b| b.type_str() == "moov") {
            let moov_start = (moov.offset + moov.header_size as u64) as usize;
            let moov_end = (moov.offset + moov.size) as usize;
            if moov_end <= data.len() {
                let moov_data = &data[moov_start..moov_end];
                let extra_nals = super::codec::parse_annex_b(moov_data);
                // Prepend parameter set NALs before video NALs
                let mut combined = extra_nals;
                combined.extend(nal_units);
                nal_units = combined;
            }
        }

        if nal_units.is_empty() {
            return Err(VideoError::ContainerParse(
                "no NAL units found in MP4 mdat".into(),
            ));
        }

        Ok(Self {
            decoder: super::h264_decoder::H264Decoder::new(),
            nal_units,
            current_nal: 0,
        })
    }

    /// Decode the next frame. Returns `None` when all NAL units are consumed.
    pub fn next_frame(&mut self) -> Result<Option<super::codec::DecodedFrame>, VideoError> {
        while self.current_nal < self.nal_units.len() {
            let nal = &self.nal_units[self.current_nal];
            self.current_nal += 1;
            if let Some(frame) = self.decoder.process_nal(nal)? {
                return Ok(Some(frame));
            }
        }
        Ok(None)
    }

    /// Reset to the beginning (re-decode from first NAL).
    pub fn seek_start(&mut self) {
        self.current_nal = 0;
        self.decoder = super::h264_decoder::H264Decoder::new();
    }

    /// Total number of NAL units found.
    pub fn nal_count(&self) -> usize {
        self.nal_units.len()
    }
}

/// Parse length-prefixed NAL units (AVCC format, common in MP4).
/// Each NAL is preceded by a 4-byte big-endian length.
fn parse_avcc_nals(data: &[u8]) -> Vec<super::codec::NalUnit> {
    let mut units = Vec::new();
    let mut i = 0;
    while i + 4 <= data.len() {
        let len = u32::from_be_bytes([data[i], data[i + 1], data[i + 2], data[i + 3]]) as usize;
        i += 4;
        if len == 0 || i + len > data.len() {
            break;
        }
        let header = data[i];
        let nal_type = super::codec::NalUnitType::from_byte(header & 0x1F);
        let nal_ref_idc = (header >> 5) & 3;
        units.push(super::codec::NalUnit {
            nal_type,
            nal_ref_idc,
            data: data[i..i + len].to_vec(),
        });
        i += len;
    }
    units
}

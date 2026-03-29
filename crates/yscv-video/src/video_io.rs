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

/// Detected video codec in an MP4 container.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mp4Codec {
    H264,
    Hevc,
}

/// Reads H.264 or HEVC encoded MP4 video files and decodes frames to RGB8.
///
/// Combines the MP4 box parser, NAL extraction, and the appropriate decoder
/// (H.264 or HEVC) into a single end-to-end reader. The codec is auto-detected
/// from the MP4 sample entry type (avc1/hvc1/hev1).
///
/// ```ignore
/// let mut reader = Mp4VideoReader::open("input.mp4")?;
/// while let Some(frame) = reader.next_frame()? {
///     // frame.rgb8_data is Vec<u8>, frame.width / frame.height
/// }
/// ```
pub struct Mp4VideoReader {
    decoder: Box<dyn super::codec::VideoDecoder>,
    codec: Mp4Codec,
    /// H.264 NAL units (used only for H.264 codec path)
    h264_nals: Vec<super::codec::NalUnit>,
    /// HEVC raw NAL unit data (used only for HEVC codec path)
    hevc_nals: Vec<Vec<u8>>,
    current_nal: usize,
}

impl Mp4VideoReader {
    /// Open an MP4 file containing H.264 or HEVC video.
    ///
    /// Reads the entire file, parses MP4 boxes to find the `mdat` box,
    /// auto-detects the codec from sample entry types (avc1/hvc1/hev1),
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

        // Detect codec from moov box (look for avc1/hvc1/hev1 sample entries)
        let detected_codec = detect_mp4_codec(&data, &boxes);

        match detected_codec {
            Mp4Codec::Hevc => {
                // HEVC: parse NALs with 2-byte headers
                let mut hevc_nals = parse_hevc_avcc_nals(mdat_data);
                if hevc_nals.is_empty() {
                    // Try Annex B
                    hevc_nals = super::hevc_decoder::parse_hevc_annex_b(mdat_data);
                }

                // Extract VPS/SPS/PPS from hvcC box in moov
                if let Some(moov) = boxes.iter().find(|b| b.type_str() == "moov") {
                    let moov_start = (moov.offset + moov.header_size as u64) as usize;
                    let moov_end = (moov.offset + moov.size) as usize;
                    if moov_end <= data.len() {
                        let extra = extract_hvcc_nals(&data[moov_start..moov_end]);
                        if !extra.is_empty() {
                            let mut combined = extra;
                            combined.extend(hevc_nals);
                            hevc_nals = combined;
                        }
                    }
                }

                if hevc_nals.is_empty() {
                    return Err(VideoError::ContainerParse(
                        "no HEVC NAL units found in MP4 mdat".into(),
                    ));
                }

                Ok(Self {
                    decoder: Box::new(super::hevc_decoder::HevcDecoder::new()),
                    codec: Mp4Codec::Hevc,
                    h264_nals: Vec::new(),
                    hevc_nals,
                    current_nal: 0,
                })
            }
            Mp4Codec::H264 => {
                // H.264 path (original behavior)
                let mut nal_units = super::codec::parse_annex_b(mdat_data);
                if nal_units.is_empty() {
                    nal_units = parse_avcc_nals(mdat_data);
                }

                if let Some(moov) = boxes.iter().find(|b| b.type_str() == "moov") {
                    let moov_start = (moov.offset + moov.header_size as u64) as usize;
                    let moov_end = (moov.offset + moov.size) as usize;
                    if moov_end <= data.len() {
                        let moov_data = &data[moov_start..moov_end];
                        let extra_nals = super::codec::parse_annex_b(moov_data);
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
                    decoder: Box::new(super::h264_decoder::H264Decoder::new()),
                    codec: Mp4Codec::H264,
                    h264_nals: nal_units,
                    hevc_nals: Vec::new(),
                    current_nal: 0,
                })
            }
        }
    }

    /// Decode the next frame. Returns `None` when all NAL units are consumed.
    pub fn next_frame(&mut self) -> Result<Option<super::codec::DecodedFrame>, VideoError> {
        match self.codec {
            Mp4Codec::H264 => {
                while self.current_nal < self.h264_nals.len() {
                    let nal = self.h264_nals[self.current_nal].clone();
                    self.current_nal += 1;
                    // Wrap NAL in Annex B start code for the VideoDecoder trait
                    let mut annex_b = vec![0x00, 0x00, 0x00, 0x01];
                    annex_b.extend_from_slice(&nal.data);
                    if let Some(frame) = self.decoder.decode(&annex_b, 0)? {
                        return Ok(Some(frame));
                    }
                }
                Ok(None)
            }
            Mp4Codec::Hevc => {
                while self.current_nal < self.hevc_nals.len() {
                    let nal_data = self.hevc_nals[self.current_nal].clone();
                    self.current_nal += 1;
                    // Wrap NAL in Annex B start code for the VideoDecoder trait
                    let mut annex_b = vec![0x00, 0x00, 0x00, 0x01];
                    annex_b.extend_from_slice(&nal_data);
                    if let Some(frame) = self.decoder.decode(&annex_b, 0)? {
                        return Ok(Some(frame));
                    }
                }
                Ok(None)
            }
        }
    }

    /// Reset to the beginning (re-decode from first NAL).
    pub fn seek_start(&mut self) {
        self.current_nal = 0;
        match self.codec {
            Mp4Codec::H264 => {
                self.decoder = Box::new(super::h264_decoder::H264Decoder::new());
            }
            Mp4Codec::Hevc => {
                self.decoder = Box::new(super::hevc_decoder::HevcDecoder::new());
            }
        }
    }

    /// Total number of NAL units found.
    pub fn nal_count(&self) -> usize {
        match self.codec {
            Mp4Codec::H264 => self.h264_nals.len(),
            Mp4Codec::Hevc => self.hevc_nals.len(),
        }
    }

    /// Returns the detected video codec.
    pub fn codec(&self) -> super::codec::VideoCodec {
        match self.codec {
            Mp4Codec::H264 => super::codec::VideoCodec::H264,
            Mp4Codec::Hevc => super::codec::VideoCodec::H265,
        }
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

/// Parse length-prefixed HEVC NAL units (HVCC/AVCC-like format).
/// Each NAL is preceded by a 4-byte big-endian length. Returns raw NAL data
/// including the 2-byte HEVC NAL header.
fn parse_hevc_avcc_nals(data: &[u8]) -> Vec<Vec<u8>> {
    let mut units = Vec::new();
    let mut i = 0;
    while i + 4 <= data.len() {
        let len = u32::from_be_bytes([data[i], data[i + 1], data[i + 2], data[i + 3]]) as usize;
        i += 4;
        if len < 2 || i + len > data.len() {
            break;
        }
        units.push(data[i..i + len].to_vec());
        i += len;
    }
    units
}

/// Detect video codec from MP4 moov box by scanning for sample entry types.
/// Looks for avc1/avc3 (H.264) or hvc1/hev1 (HEVC) in the raw moov data.
fn detect_mp4_codec(data: &[u8], boxes: &[super::codec::Mp4Box]) -> Mp4Codec {
    if let Some(moov) = boxes.iter().find(|b| b.type_str() == "moov") {
        let start = (moov.offset + moov.header_size as u64) as usize;
        let end = ((moov.offset + moov.size) as usize).min(data.len());
        if start < end {
            let moov_data = &data[start..end];
            // Scan for sample entry type codes in the moov box
            for i in 0..moov_data.len().saturating_sub(4) {
                let tag = &moov_data[i..i + 4];
                if tag == b"hvc1" || tag == b"hev1" {
                    return Mp4Codec::Hevc;
                }
            }
        }
    }
    Mp4Codec::H264 // default
}

/// Extract VPS/SPS/PPS NAL units from an hvcC (HEVC Decoder Configuration Record)
/// found inside the moov box. Returns raw NAL data suitable for HevcDecoder.
fn extract_hvcc_nals(moov_data: &[u8]) -> Vec<Vec<u8>> {
    let mut nals = Vec::new();

    // Scan for "hvcC" tag in moov data
    for i in 0..moov_data.len().saturating_sub(4) {
        if &moov_data[i..i + 4] == b"hvcC" {
            // hvcC box: skip 22 bytes of config header to reach arrays
            let config_start = i + 4;
            if config_start + 23 > moov_data.len() {
                break;
            }
            let num_arrays = moov_data[config_start + 22];
            let mut pos = config_start + 23;

            for _ in 0..num_arrays {
                if pos + 3 > moov_data.len() {
                    break;
                }
                let _array_completeness_and_type = moov_data[pos];
                pos += 1;
                let num_nalus =
                    u16::from_be_bytes([moov_data[pos], moov_data[pos + 1]]) as usize;
                pos += 2;

                for _ in 0..num_nalus {
                    if pos + 2 > moov_data.len() {
                        break;
                    }
                    let nal_len =
                        u16::from_be_bytes([moov_data[pos], moov_data[pos + 1]]) as usize;
                    pos += 2;
                    if pos + nal_len > moov_data.len() {
                        break;
                    }
                    nals.push(moov_data[pos..pos + nal_len].to_vec());
                    pos += nal_len;
                }
            }
            break;
        }
    }

    nals
}

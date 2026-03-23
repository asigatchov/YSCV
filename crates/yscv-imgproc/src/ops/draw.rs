use yscv_tensor::Tensor;

use super::super::ImgProcError;
use super::super::shape::hwc_shape;

/// Draw a rectangle on a `[H, W, 3]` image tensor (values `[0, 1]`).
///
/// `(x, y)` is the top-left corner, `(w, h)` is width and height in pixels.
/// `thickness` specifies line width; `0` fills the rectangle.
#[allow(unsafe_code)]
pub fn draw_rect(
    image: &mut Tensor,
    x: usize,
    y: usize,
    rect_w: usize,
    rect_h: usize,
    color: [f32; 3],
    thickness: usize,
) -> Result<(), ImgProcError> {
    let (img_h, img_w, channels) = hwc_shape(image)?;
    if channels != 3 {
        return Err(ImgProcError::InvalidChannelCount {
            expected: 3,
            got: channels,
        });
    }

    let data = image.data_mut();

    let set_pixel = |data: &mut [f32], py: usize, px: usize, color: &[f32; 3]| {
        if py < img_h && px < img_w {
            let idx = (py * img_w + px) * 3;
            data[idx] = color[0];
            data[idx + 1] = color[1];
            data[idx + 2] = color[2];
        }
    };

    if thickness == 0 {
        // Fill rectangle.
        for py in y..std::cmp::min(y + rect_h, img_h) {
            for px in x..std::cmp::min(x + rect_w, img_w) {
                set_pixel(data, py, px, &color);
            }
        }
    } else {
        // Draw outline with given thickness.
        for t in 0..thickness {
            // Top and bottom horizontal lines.
            for px in x.saturating_sub(t)..std::cmp::min(x + rect_w + t, img_w) {
                if y >= t {
                    set_pixel(data, y - t, px, &color);
                }
                if y + t < img_h {
                    set_pixel(data, y + t, px, &color);
                }
                let bot = y + rect_h.saturating_sub(1);
                if bot >= t {
                    set_pixel(data, bot - t, px, &color);
                }
                if bot + t < img_h {
                    set_pixel(data, bot + t, px, &color);
                }
            }
            // Left and right vertical lines.
            for py in y.saturating_sub(t)..std::cmp::min(y + rect_h + t, img_h) {
                if x >= t {
                    set_pixel(data, py, x - t, &color);
                }
                if x + t < img_w {
                    set_pixel(data, py, x + t, &color);
                }
                let right = x + rect_w.saturating_sub(1);
                if right >= t {
                    set_pixel(data, py, right - t, &color);
                }
                if right + t < img_w {
                    set_pixel(data, py, right + t, &color);
                }
            }
        }
    }

    Ok(())
}

/// Embedded 8×8 bitmap font covering ASCII 32–126.
/// Each character is stored as 8 bytes (one per row, MSB = left pixel).
const FONT_8X8: [[u8; 8]; 95] = {
    let mut f = [[0u8; 8]; 95];

    // space
    f[0] = [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
    // !
    f[1] = [0x18, 0x18, 0x18, 0x18, 0x18, 0x00, 0x18, 0x00];
    // "
    f[2] = [0x6C, 0x6C, 0x24, 0x00, 0x00, 0x00, 0x00, 0x00];
    // #
    f[3] = [0x6C, 0x6C, 0xFE, 0x6C, 0xFE, 0x6C, 0x6C, 0x00];
    // $
    f[4] = [0x18, 0x7E, 0xC0, 0x7C, 0x06, 0xFC, 0x18, 0x00];
    // %
    f[5] = [0x00, 0xC6, 0xCC, 0x18, 0x30, 0x66, 0xC6, 0x00];
    // &
    f[6] = [0x38, 0x6C, 0x38, 0x76, 0xDC, 0xCC, 0x76, 0x00];
    // '
    f[7] = [0x18, 0x18, 0x30, 0x00, 0x00, 0x00, 0x00, 0x00];
    // (
    f[8] = [0x0C, 0x18, 0x30, 0x30, 0x30, 0x18, 0x0C, 0x00];
    // )
    f[9] = [0x30, 0x18, 0x0C, 0x0C, 0x0C, 0x18, 0x30, 0x00];
    // *
    f[10] = [0x00, 0x66, 0x3C, 0xFF, 0x3C, 0x66, 0x00, 0x00];
    // +
    f[11] = [0x00, 0x18, 0x18, 0x7E, 0x18, 0x18, 0x00, 0x00];
    // ,
    f[12] = [0x00, 0x00, 0x00, 0x00, 0x00, 0x18, 0x18, 0x30];
    // -
    f[13] = [0x00, 0x00, 0x00, 0x7E, 0x00, 0x00, 0x00, 0x00];
    // .
    f[14] = [0x00, 0x00, 0x00, 0x00, 0x00, 0x18, 0x18, 0x00];
    // /
    f[15] = [0x06, 0x0C, 0x18, 0x30, 0x60, 0xC0, 0x80, 0x00];
    // 0
    f[16] = [0x7C, 0xC6, 0xCE, 0xDE, 0xF6, 0xE6, 0x7C, 0x00];
    // 1
    f[17] = [0x18, 0x38, 0x78, 0x18, 0x18, 0x18, 0x7E, 0x00];
    // 2
    f[18] = [0x7C, 0xC6, 0x06, 0x1C, 0x30, 0x66, 0xFE, 0x00];
    // 3
    f[19] = [0x7C, 0xC6, 0x06, 0x3C, 0x06, 0xC6, 0x7C, 0x00];
    // 4
    f[20] = [0x1C, 0x3C, 0x6C, 0xCC, 0xFE, 0x0C, 0x1E, 0x00];
    // 5
    f[21] = [0xFE, 0xC0, 0xFC, 0x06, 0x06, 0xC6, 0x7C, 0x00];
    // 6
    f[22] = [0x38, 0x60, 0xC0, 0xFC, 0xC6, 0xC6, 0x7C, 0x00];
    // 7
    f[23] = [0xFE, 0xC6, 0x0C, 0x18, 0x30, 0x30, 0x30, 0x00];
    // 8
    f[24] = [0x7C, 0xC6, 0xC6, 0x7C, 0xC6, 0xC6, 0x7C, 0x00];
    // 9
    f[25] = [0x7C, 0xC6, 0xC6, 0x7E, 0x06, 0x0C, 0x78, 0x00];
    // :
    f[26] = [0x00, 0x18, 0x18, 0x00, 0x00, 0x18, 0x18, 0x00];
    // ;
    f[27] = [0x00, 0x18, 0x18, 0x00, 0x00, 0x18, 0x18, 0x30];
    // <
    f[28] = [0x0C, 0x18, 0x30, 0x60, 0x30, 0x18, 0x0C, 0x00];
    // =
    f[29] = [0x00, 0x00, 0x7E, 0x00, 0x7E, 0x00, 0x00, 0x00];
    // >
    f[30] = [0x60, 0x30, 0x18, 0x0C, 0x18, 0x30, 0x60, 0x00];
    // ?
    f[31] = [0x7C, 0xC6, 0x0C, 0x18, 0x18, 0x00, 0x18, 0x00];
    // @
    f[32] = [0x7C, 0xC6, 0xDE, 0xDE, 0xDC, 0xC0, 0x7C, 0x00];
    // A
    f[33] = [0x38, 0x6C, 0xC6, 0xC6, 0xFE, 0xC6, 0xC6, 0x00];
    // B
    f[34] = [0xFC, 0x66, 0x66, 0x7C, 0x66, 0x66, 0xFC, 0x00];
    // C
    f[35] = [0x3C, 0x66, 0xC0, 0xC0, 0xC0, 0x66, 0x3C, 0x00];
    // D
    f[36] = [0xF8, 0x6C, 0x66, 0x66, 0x66, 0x6C, 0xF8, 0x00];
    // E
    f[37] = [0xFE, 0x62, 0x68, 0x78, 0x68, 0x62, 0xFE, 0x00];
    // F
    f[38] = [0xFE, 0x62, 0x68, 0x78, 0x68, 0x60, 0xF0, 0x00];
    // G
    f[39] = [0x3C, 0x66, 0xC0, 0xC0, 0xCE, 0x66, 0x3E, 0x00];
    // H
    f[40] = [0xC6, 0xC6, 0xC6, 0xFE, 0xC6, 0xC6, 0xC6, 0x00];
    // I
    f[41] = [0x3C, 0x18, 0x18, 0x18, 0x18, 0x18, 0x3C, 0x00];
    // J
    f[42] = [0x1E, 0x0C, 0x0C, 0x0C, 0xCC, 0xCC, 0x78, 0x00];
    // K
    f[43] = [0xE6, 0x66, 0x6C, 0x78, 0x6C, 0x66, 0xE6, 0x00];
    // L
    f[44] = [0xF0, 0x60, 0x60, 0x60, 0x62, 0x66, 0xFE, 0x00];
    // M
    f[45] = [0xC6, 0xEE, 0xFE, 0xD6, 0xC6, 0xC6, 0xC6, 0x00];
    // N
    f[46] = [0xC6, 0xE6, 0xF6, 0xDE, 0xCE, 0xC6, 0xC6, 0x00];
    // O
    f[47] = [0x7C, 0xC6, 0xC6, 0xC6, 0xC6, 0xC6, 0x7C, 0x00];
    // P
    f[48] = [0xFC, 0x66, 0x66, 0x7C, 0x60, 0x60, 0xF0, 0x00];
    // Q
    f[49] = [0x7C, 0xC6, 0xC6, 0xC6, 0xD6, 0xDE, 0x7C, 0x06];
    // R
    f[50] = [0xFC, 0x66, 0x66, 0x7C, 0x6C, 0x66, 0xE6, 0x00];
    // S
    f[51] = [0x7C, 0xC6, 0xC0, 0x7C, 0x06, 0xC6, 0x7C, 0x00];
    // T
    f[52] = [0x7E, 0x5A, 0x18, 0x18, 0x18, 0x18, 0x3C, 0x00];
    // U
    f[53] = [0xC6, 0xC6, 0xC6, 0xC6, 0xC6, 0xC6, 0x7C, 0x00];
    // V
    f[54] = [0xC6, 0xC6, 0xC6, 0xC6, 0x6C, 0x38, 0x10, 0x00];
    // W
    f[55] = [0xC6, 0xC6, 0xC6, 0xD6, 0xFE, 0xEE, 0xC6, 0x00];
    // X
    f[56] = [0xC6, 0xC6, 0x6C, 0x38, 0x6C, 0xC6, 0xC6, 0x00];
    // Y
    f[57] = [0x66, 0x66, 0x66, 0x3C, 0x18, 0x18, 0x3C, 0x00];
    // Z
    f[58] = [0xFE, 0xC6, 0x8C, 0x18, 0x32, 0x66, 0xFE, 0x00];
    // [
    f[59] = [0x3C, 0x30, 0x30, 0x30, 0x30, 0x30, 0x3C, 0x00];
    // backslash
    f[60] = [0xC0, 0x60, 0x30, 0x18, 0x0C, 0x06, 0x02, 0x00];
    // ]
    f[61] = [0x3C, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x3C, 0x00];
    // ^
    f[62] = [0x10, 0x38, 0x6C, 0xC6, 0x00, 0x00, 0x00, 0x00];
    // _
    f[63] = [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFF];
    // `
    f[64] = [0x30, 0x18, 0x0C, 0x00, 0x00, 0x00, 0x00, 0x00];
    // a
    f[65] = [0x00, 0x00, 0x78, 0x0C, 0x7C, 0xCC, 0x76, 0x00];
    // b
    f[66] = [0xE0, 0x60, 0x7C, 0x66, 0x66, 0x66, 0xDC, 0x00];
    // c
    f[67] = [0x00, 0x00, 0x7C, 0xC6, 0xC0, 0xC6, 0x7C, 0x00];
    // d
    f[68] = [0x1C, 0x0C, 0x7C, 0xCC, 0xCC, 0xCC, 0x76, 0x00];
    // e
    f[69] = [0x00, 0x00, 0x7C, 0xC6, 0xFE, 0xC0, 0x7C, 0x00];
    // f
    f[70] = [0x1C, 0x36, 0x30, 0x78, 0x30, 0x30, 0x78, 0x00];
    // g
    f[71] = [0x00, 0x00, 0x76, 0xCC, 0xCC, 0x7C, 0x0C, 0x78];
    // h
    f[72] = [0xE0, 0x60, 0x6C, 0x76, 0x66, 0x66, 0xE6, 0x00];
    // i
    f[73] = [0x18, 0x00, 0x38, 0x18, 0x18, 0x18, 0x3C, 0x00];
    // j
    f[74] = [0x06, 0x00, 0x0E, 0x06, 0x06, 0x66, 0x66, 0x3C];
    // k
    f[75] = [0xE0, 0x60, 0x66, 0x6C, 0x78, 0x6C, 0xE6, 0x00];
    // l
    f[76] = [0x38, 0x18, 0x18, 0x18, 0x18, 0x18, 0x3C, 0x00];
    // m
    f[77] = [0x00, 0x00, 0xEC, 0xFE, 0xD6, 0xC6, 0xC6, 0x00];
    // n
    f[78] = [0x00, 0x00, 0xDC, 0x66, 0x66, 0x66, 0x66, 0x00];
    // o
    f[79] = [0x00, 0x00, 0x7C, 0xC6, 0xC6, 0xC6, 0x7C, 0x00];
    // p
    f[80] = [0x00, 0x00, 0xDC, 0x66, 0x66, 0x7C, 0x60, 0xF0];
    // q
    f[81] = [0x00, 0x00, 0x76, 0xCC, 0xCC, 0x7C, 0x0C, 0x1E];
    // r
    f[82] = [0x00, 0x00, 0xDC, 0x76, 0x60, 0x60, 0xF0, 0x00];
    // s
    f[83] = [0x00, 0x00, 0x7C, 0xC0, 0x7C, 0x06, 0xFC, 0x00];
    // t
    f[84] = [0x10, 0x30, 0x7C, 0x30, 0x30, 0x34, 0x18, 0x00];
    // u
    f[85] = [0x00, 0x00, 0xCC, 0xCC, 0xCC, 0xCC, 0x76, 0x00];
    // v
    f[86] = [0x00, 0x00, 0xC6, 0xC6, 0xC6, 0x6C, 0x38, 0x00];
    // w
    f[87] = [0x00, 0x00, 0xC6, 0xC6, 0xD6, 0xFE, 0x6C, 0x00];
    // x
    f[88] = [0x00, 0x00, 0xC6, 0x6C, 0x38, 0x6C, 0xC6, 0x00];
    // y
    f[89] = [0x00, 0x00, 0xC6, 0xC6, 0xCE, 0x76, 0x06, 0xFC];
    // z
    f[90] = [0x00, 0x00, 0xFC, 0x98, 0x30, 0x64, 0xFC, 0x00];
    // {
    f[91] = [0x0E, 0x18, 0x18, 0x70, 0x18, 0x18, 0x0E, 0x00];
    // |
    f[92] = [0x18, 0x18, 0x18, 0x00, 0x18, 0x18, 0x18, 0x00];
    // }
    f[93] = [0x70, 0x18, 0x18, 0x0E, 0x18, 0x18, 0x70, 0x00];
    // ~
    f[94] = [0x76, 0xDC, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];

    f
};

/// Draw text on a `[H, W, 3]` image tensor using an embedded 8×8 bitmap font.
///
/// `(x, y)` is the top-left pixel of the first character. Characters outside the
/// image bounds are silently clipped. Only ASCII 32–126 is rendered; other bytes
/// are replaced with `?`.
pub fn draw_text(
    image: &mut Tensor,
    text: &str,
    x: usize,
    y: usize,
    color: [f32; 3],
) -> Result<(), ImgProcError> {
    let (img_h, img_w, channels) = hwc_shape(image)?;
    if channels != 3 {
        return Err(ImgProcError::InvalidChannelCount {
            expected: 3,
            got: channels,
        });
    }

    let data = image.data_mut();

    for (ci, ch) in text.bytes().enumerate() {
        let glyph_idx = if (32..=126).contains(&ch) {
            (ch - 32) as usize
        } else {
            31 // '?'
        };
        let glyph = &FONT_8X8[glyph_idx];
        let cx = x + ci * 8;
        for row in 0..8 {
            let py = y + row;
            if py >= img_h {
                break;
            }
            let bits = glyph[row];
            for col in 0..8 {
                let px = cx + col;
                if px >= img_w {
                    break;
                }
                if bits & (0x80 >> col) != 0 {
                    let idx = (py * img_w + px) * 3;
                    data[idx] = color[0];
                    data[idx + 1] = color[1];
                    data[idx + 2] = color[2];
                }
            }
        }
    }

    Ok(())
}

/// A detection result for drawing purposes.
pub struct Detection {
    pub x: usize,
    pub y: usize,
    pub width: usize,
    pub height: usize,
    pub score: f32,
    pub class_id: usize,
}

/// Draw detections (bounding boxes + labels) on a `[H, W, 3]` image tensor.
///
/// Each detection is drawn with a color picked from a built-in palette based on
/// `class_id`. If `labels` is provided and `class_id` is in range, the label and
/// score are drawn above the bounding box.
pub fn draw_detections(
    image: &mut Tensor,
    detections: &[Detection],
    labels: &[&str],
) -> Result<(), ImgProcError> {
    const PALETTE: [[f32; 3]; 10] = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [1.0, 0.5, 0.0],
        [0.5, 0.0, 1.0],
        [0.0, 0.5, 0.0],
        [0.5, 0.5, 0.5],
    ];

    for det in detections {
        let color = PALETTE[det.class_id % PALETTE.len()];
        draw_rect(image, det.x, det.y, det.width, det.height, color, 2)?;

        if det.class_id < labels.len() {
            let label = format!("{} {:.2}", labels[det.class_id], det.score);
            let text_y = if det.y >= 10 {
                det.y - 10
            } else {
                det.y + det.height + 2
            };
            draw_text(image, &label, det.x, text_y, color)?;
        }
    }

    Ok(())
}

/// Draw a line using Bresenham's algorithm on a `[H, W, 3]` image tensor.
#[allow(unsafe_code)]
pub fn draw_line(
    image: &mut Tensor,
    x0: i32,
    y0: i32,
    x1: i32,
    y1: i32,
    color: [f32; 3],
    thickness: usize,
) -> Result<(), ImgProcError> {
    let (img_h, img_w, channels) = hwc_shape(image)?;
    if channels != 3 {
        return Err(ImgProcError::InvalidChannelCount {
            expected: 3,
            got: channels,
        });
    }

    let data = image.data_mut();
    let half_t = thickness.saturating_sub(1) / 2;

    let set_thick_pixel = |data: &mut [f32], py: i32, px: i32| {
        for dy in 0..=half_t {
            for dx in 0..=half_t {
                for (sy, sx) in [
                    (py + dy as i32, px + dx as i32),
                    (py + dy as i32, px - dx as i32),
                    (py - (dy as i32), px + dx as i32),
                    (py - (dy as i32), px - dx as i32),
                ] {
                    if sy >= 0 && (sy as usize) < img_h && sx >= 0 && (sx as usize) < img_w {
                        let idx = (sy as usize * img_w + sx as usize) * 3;
                        data[idx] = color[0];
                        data[idx + 1] = color[1];
                        data[idx + 2] = color[2];
                    }
                }
            }
        }
    };

    let mut x = x0;
    let mut y = y0;
    let dx = (x1 - x0).abs();
    let dy = -(y1 - y0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;

    loop {
        set_thick_pixel(data, y, x);
        if x == x1 && y == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x += sx;
        }
        if e2 <= dx {
            err += dx;
            y += sy;
        }
    }

    Ok(())
}

/// Draw a circle using the midpoint circle algorithm on a `[H, W, 3]` image tensor.
///
/// `(cx, cy)` is the center, `radius` is the radius in pixels.
/// `thickness == 0` fills the circle.
#[allow(unsafe_code)]
pub fn draw_circle(
    image: &mut Tensor,
    cx: i32,
    cy: i32,
    radius: usize,
    color: [f32; 3],
    thickness: usize,
) -> Result<(), ImgProcError> {
    let (img_h, img_w, channels) = hwc_shape(image)?;
    if channels != 3 {
        return Err(ImgProcError::InvalidChannelCount {
            expected: 3,
            got: channels,
        });
    }

    let data = image.data_mut();
    let r = radius as i32;

    if thickness == 0 {
        // Fill circle
        for py in (cy - r).max(0)..=(cy + r).min(img_h as i32 - 1) {
            let dy = py - cy;
            let half_w = ((r * r - dy * dy) as f32).sqrt() as i32;
            for px in (cx - half_w).max(0)..=(cx + half_w).min(img_w as i32 - 1) {
                let idx = (py as usize * img_w + px as usize) * 3;
                data[idx] = color[0];
                data[idx + 1] = color[1];
                data[idx + 2] = color[2];
            }
        }
    } else {
        // Midpoint circle — draw 8 symmetric points
        let mut x = 0i32;
        let mut y = r;
        let mut d = 1 - r;

        let plot = |data: &mut [f32], px: i32, py: i32| {
            if px >= 0 && (px as usize) < img_w && py >= 0 && (py as usize) < img_h {
                let idx = (py as usize * img_w + px as usize) * 3;
                data[idx] = color[0];
                data[idx + 1] = color[1];
                data[idx + 2] = color[2];
            }
        };

        while x <= y {
            plot(data, cx + x, cy + y);
            plot(data, cx - x, cy + y);
            plot(data, cx + x, cy - y);
            plot(data, cx - x, cy - y);
            plot(data, cx + y, cy + x);
            plot(data, cx - y, cy + x);
            plot(data, cx + y, cy - x);
            plot(data, cx - y, cy - x);

            if d < 0 {
                d += 2 * x + 3;
            } else {
                d += 2 * (x - y) + 5;
                y -= 1;
            }
            x += 1;
        }
    }

    Ok(())
}

/// Draw connected line segments on a `[H, W, 3]` image tensor.
///
/// `points` is a list of `(x, y)` coordinates. If `closed` is true,
/// the last point connects back to the first.
pub fn draw_polylines(
    image: &mut Tensor,
    points: &[(i32, i32)],
    closed: bool,
    color: [f32; 3],
    thickness: usize,
) -> Result<(), ImgProcError> {
    if points.len() < 2 {
        return Ok(());
    }
    for i in 0..points.len() - 1 {
        draw_line(
            image,
            points[i].0,
            points[i].1,
            points[i + 1].0,
            points[i + 1].1,
            color,
            thickness,
        )?;
    }
    if closed {
        let last = points.len() - 1;
        draw_line(
            image,
            points[last].0,
            points[last].1,
            points[0].0,
            points[0].1,
            color,
            thickness,
        )?;
    }
    Ok(())
}

/// Fill a polygon using scanline fill on a `[H, W, 3]` image tensor.
#[allow(unsafe_code)]
pub fn fill_poly(
    image: &mut Tensor,
    points: &[(i32, i32)],
    color: [f32; 3],
) -> Result<(), ImgProcError> {
    let (img_h, img_w, channels) = hwc_shape(image)?;
    if channels != 3 {
        return Err(ImgProcError::InvalidChannelCount {
            expected: 3,
            got: channels,
        });
    }
    if points.len() < 3 {
        return Ok(());
    }

    let min_y = points
        .iter()
        .map(|p| p.1)
        .min()
        .expect("non-empty points")
        .max(0) as usize;
    let max_y = points
        .iter()
        .map(|p| p.1)
        .max()
        .expect("non-empty points")
        .min(img_h as i32 - 1) as usize;

    let data = image.data_mut();
    let n = points.len();

    for y in min_y..=max_y {
        let mut intersections: Vec<i32> = Vec::new();
        let yf = y as i32;
        for i in 0..n {
            let j = (i + 1) % n;
            let (y0, y1) = (points[i].1, points[j].1);
            let (x0, x1) = (points[i].0, points[j].0);
            if (y0 <= yf && y1 > yf) || (y1 <= yf && y0 > yf) {
                let x = x0 + (yf - y0) * (x1 - x0) / (y1 - y0);
                intersections.push(x);
            }
        }
        intersections.sort();

        for pair in intersections.chunks(2) {
            if pair.len() == 2 {
                let x_start = pair[0].max(0) as usize;
                let x_end = (pair[1] as usize).min(img_w - 1);
                for px in x_start..=x_end {
                    let idx = (y * img_w + px) * 3;
                    data[idx] = color[0];
                    data[idx + 1] = color[1];
                    data[idx + 2] = color[2];
                }
            }
        }
    }

    Ok(())
}

/// Draw text with integer scaling (nearest-neighbor scaled 8x8 bitmap font).
///
/// `scale` multiplies each pixel of the 8x8 glyph. scale=2 → 16x16 characters.
#[allow(unsafe_code)]
pub fn draw_text_scaled(
    image: &mut Tensor,
    text: &str,
    x: usize,
    y: usize,
    scale: usize,
    color: [f32; 3],
) -> Result<(), ImgProcError> {
    let (img_h, img_w, channels) = hwc_shape(image)?;
    if channels != 3 {
        return Err(ImgProcError::InvalidChannelCount {
            expected: 3,
            got: channels,
        });
    }
    let scale = scale.max(1);
    let data = image.data_mut();

    for (ci, ch) in text.bytes().enumerate() {
        let glyph_idx = if (32..=126).contains(&ch) {
            (ch - 32) as usize
        } else {
            31 // '?'
        };
        let glyph = &FONT_8X8[glyph_idx];
        let cx = x + ci * 8 * scale;
        for row in 0..8 {
            let bits = glyph[row];
            for col in 0..8 {
                if bits & (0x80 >> col) != 0 {
                    // Fill scale x scale block
                    for sy in 0..scale {
                        let py = y + row * scale + sy;
                        if py >= img_h {
                            break;
                        }
                        for sx in 0..scale {
                            let px = cx + col * scale + sx;
                            if px >= img_w {
                                break;
                            }
                            let idx = (py * img_w + px) * 3;
                            data[idx] = color[0];
                            data[idx + 1] = color[1];
                            data[idx + 2] = color[2];
                        }
                    }
                }
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn blank_image(h: usize, w: usize) -> Tensor {
        Tensor::from_vec(vec![h, w, 3], vec![0.0f32; h * w * 3]).unwrap()
    }

    #[test]
    fn test_draw_rect_fill() {
        let mut img = blank_image(10, 10);
        draw_rect(&mut img, 2, 2, 3, 3, [1.0, 0.0, 0.0], 0).unwrap();

        // Center pixel (3, 3) should be red.
        let idx = (3 * 10 + 3) * 3;
        let data = img.data();
        assert_eq!(data[idx], 1.0);
        assert_eq!(data[idx + 1], 0.0);

        // Pixel (0, 0) should still be black.
        assert_eq!(data[0], 0.0);
    }

    #[test]
    fn test_draw_rect_outline() {
        let mut img = blank_image(20, 20);
        draw_rect(&mut img, 5, 5, 10, 10, [0.0, 1.0, 0.0], 1).unwrap();

        // Top-left corner (5, 5) should be green.
        let idx = (5 * 20 + 5) * 3;
        let data = img.data();
        assert_eq!(data[idx + 1], 1.0);

        // Interior pixel (10, 10) should still be black.
        let idx2 = (10 * 20 + 10) * 3;
        assert_eq!(data[idx2], 0.0);
    }

    #[test]
    fn test_draw_text_renders_pixels() {
        let mut img = blank_image(16, 80);
        draw_text(&mut img, "Hi", 0, 0, [1.0, 1.0, 1.0]).unwrap();

        // Some pixels in the first 8x16 block should be set.
        let data = img.data();
        let white_count: usize = (0..8 * 16 * 3)
            .step_by(3)
            .filter(|&i| data[i] > 0.5)
            .count();
        assert!(
            white_count > 5,
            "expected some rendered pixels, got {white_count}"
        );
    }

    #[test]
    fn test_draw_detections_smoke() {
        let mut img = blank_image(100, 100);
        let dets = vec![
            Detection {
                x: 10,
                y: 10,
                width: 30,
                height: 30,
                score: 0.95,
                class_id: 0,
            },
            Detection {
                x: 50,
                y: 50,
                width: 20,
                height: 20,
                score: 0.80,
                class_id: 1,
            },
        ];
        draw_detections(&mut img, &dets, &["cat", "dog"]).unwrap();

        // Just verify it doesn't panic and modifies some pixels.
        let data = img.data();
        let non_zero = data.iter().filter(|&&v| v > 0.0).count();
        assert!(non_zero > 0, "expected some drawn pixels");
    }

    #[test]
    fn test_draw_line_diagonal() {
        let mut img = blank_image(20, 20);
        draw_line(&mut img, 0, 0, 19, 19, [1.0, 0.0, 0.0], 1).unwrap();
        // Diagonal pixel (10, 10) should be red
        let idx = (10 * 20 + 10) * 3;
        assert_eq!(img.data()[idx], 1.0);
    }

    #[test]
    fn test_draw_circle_filled() {
        let mut img = blank_image(50, 50);
        draw_circle(&mut img, 25, 25, 10, [0.0, 1.0, 0.0], 0).unwrap();
        // Center should be green
        let idx = (25 * 50 + 25) * 3;
        assert_eq!(img.data()[idx + 1], 1.0);
        // Far corner should be black
        assert_eq!(img.data()[0], 0.0);
    }

    #[test]
    fn test_draw_circle_outline() {
        let mut img = blank_image(50, 50);
        draw_circle(&mut img, 25, 25, 10, [1.0, 0.0, 0.0], 1).unwrap();
        // Center should remain black (outline only)
        let idx = (25 * 50 + 25) * 3;
        assert_eq!(img.data()[idx], 0.0);
        // A point on the circle should be drawn
        let on_circle = (25 * 50 + 35) * 3; // (35, 25) is on the circle
        assert_eq!(img.data()[on_circle], 1.0);
    }

    #[test]
    fn test_draw_polylines_closed() {
        let mut img = blank_image(20, 20);
        let pts = [(2, 2), (17, 2), (17, 17), (2, 17)];
        draw_polylines(&mut img, &pts, true, [0.0, 0.0, 1.0], 1).unwrap();
        let data = img.data();
        let blue_count = (0..data.len())
            .step_by(3)
            .filter(|&i| data[i + 2] > 0.5)
            .count();
        assert!(blue_count > 40, "expected outline pixels, got {blue_count}");
    }

    #[test]
    fn test_fill_poly_triangle() {
        let mut img = blank_image(30, 30);
        let pts = [(15, 2), (28, 27), (2, 27)];
        fill_poly(&mut img, &pts, [1.0, 0.0, 0.0]).unwrap();
        // Center of triangle should be filled
        let idx = (15 * 30 + 15) * 3;
        assert_eq!(img.data()[idx], 1.0);
    }

    #[test]
    fn test_draw_text_scaled_larger() {
        let mut img = blank_image(32, 64);
        draw_text_scaled(&mut img, "A", 0, 0, 2, [1.0, 1.0, 1.0]).unwrap();
        let data = img.data();
        let white_count = (0..data.len())
            .step_by(3)
            .filter(|&i| data[i] > 0.5)
            .count();
        // scale=2 should produce ~4x the pixels of scale=1
        assert!(
            white_count > 20,
            "expected scaled text pixels, got {white_count}"
        );
    }
}

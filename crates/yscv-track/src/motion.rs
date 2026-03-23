use yscv_detect::BoundingBox;

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct MotionState {
    pub(crate) center_vx: f32,
    pub(crate) center_vy: f32,
    pub(crate) width_v: f32,
    pub(crate) height_v: f32,
}

impl Default for MotionState {
    fn default() -> Self {
        Self {
            center_vx: 0.0,
            center_vy: 0.0,
            width_v: 0.0,
            height_v: 0.0,
        }
    }
}

pub(crate) fn update_motion_state(
    state: &mut MotionState,
    previous: BoundingBox,
    current: BoundingBox,
) {
    const ALPHA: f32 = 0.7;

    let (previous_cx, previous_cy) = bbox_center(previous);
    let (current_cx, current_cy) = bbox_center(current);
    let observed_center_vx = current_cx - previous_cx;
    let observed_center_vy = current_cy - previous_cy;
    let observed_width_v = current.width() - previous.width();
    let observed_height_v = current.height() - previous.height();

    state.center_vx = ALPHA * observed_center_vx + (1.0 - ALPHA) * state.center_vx;
    state.center_vy = ALPHA * observed_center_vy + (1.0 - ALPHA) * state.center_vy;
    state.width_v = ALPHA * observed_width_v + (1.0 - ALPHA) * state.width_v;
    state.height_v = ALPHA * observed_height_v + (1.0 - ALPHA) * state.height_v;
}

pub(crate) fn apply_motion(bbox: BoundingBox, state: &MotionState, dt: f32) -> BoundingBox {
    let (center_x, center_y) = bbox_center(bbox);
    let predicted_center_x = center_x + state.center_vx * dt;
    let predicted_center_y = center_y + state.center_vy * dt;
    let predicted_width = (bbox.width() + state.width_v * dt).max(1.0e-3);
    let predicted_height = (bbox.height() + state.height_v * dt).max(1.0e-3);

    BoundingBox {
        x1: predicted_center_x - predicted_width * 0.5,
        y1: predicted_center_y - predicted_height * 0.5,
        x2: predicted_center_x + predicted_width * 0.5,
        y2: predicted_center_y + predicted_height * 0.5,
    }
}

pub(crate) fn normalized_center_distance(left: BoundingBox, right: BoundingBox) -> f32 {
    let (left_cx, left_cy) = bbox_center(left);
    let (right_cx, right_cy) = bbox_center(right);
    let delta_x = left_cx - right_cx;
    let delta_y = left_cy - right_cy;
    let distance = (delta_x.powi(2) + delta_y.powi(2)).sqrt();
    let norm = ((left.width().max(right.width())).powi(2)
        + (left.height().max(right.height())).powi(2))
    .sqrt()
    .max(1.0e-3);
    distance / norm
}

pub(crate) fn bbox_size_similarity(left: BoundingBox, right: BoundingBox) -> f32 {
    let width = ratio_min_max(left.width(), right.width());
    let height = ratio_min_max(left.height(), right.height());
    width * height
}

fn bbox_center(bbox: BoundingBox) -> (f32, f32) {
    ((bbox.x1 + bbox.x2) * 0.5, (bbox.y1 + bbox.y2) * 0.5)
}

fn ratio_min_max(a: f32, b: f32) -> f32 {
    let min = a.min(b).max(1.0e-3);
    let max = a.max(b).max(1.0e-3);
    min / max
}

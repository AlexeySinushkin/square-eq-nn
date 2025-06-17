pub const WINDOW_WIDTH: usize = 1366;
pub const WINDOW_HEIGHT: usize = 768;
pub const COLOUR_BACKGROUND: u32 = 0x1e1f22;
pub const COLOUR_CIRCLE: u32 = 0xce7b47;
pub const COLOUR_LINK: u32 = 0xb7babf;
pub const COLOUR_ERROR: u32 = 0xc37ab6;

#[derive(Debug, Clone)]
pub struct Point {
    pub x: f32,
    pub y: f32,
}

pub struct NCircle {
    pub id: String,
    pub caption: Point,
    pub center: Point,    
    pub output: Point,
    pub radius: f32,
}

impl NCircle {
    pub fn new(id: String, x: f32, y: f32, radius: f32) -> Self {
        NCircle {
            id,
            center: Point { x, y },
            caption: Point {
                x,
                y: y - radius / 2.0,
            },
            output: Point {
                x,
                y: y + radius / 2.0,
            },
            radius,
        }
    }
}

pub struct Arrow {
    pub id: String,
    pub from: Point,
    pub to: Point,
    pub from_1: Point,
    pub from_2: Point,
    pub middle: Point,
}

impl Arrow {
    pub fn new(id: String, from: &NCircle, to: &NCircle) -> Self {
        // Circle A (start)
        let x1: f32 = from.center.x as f32;
        let y1: f32 = from.center.y as f32;
        let r1: f32 = from.radius as f32;
        // Circle B (end)
        let x2: f32 = to.center.x as f32;
        let y2: f32 = to.center.y as f32;
        let r2: f32 = to.radius as f32;

        // Direction vector from A to B
        let dx = x2 - x1;
        let dy = y2 - y1;
        let distance = (dx * dx + dy * dy).sqrt();
        let dir_x = dx / distance;
        let dir_y = dy / distance;

        // Start and end points on the edge of circles
        let start_x = x1 + dir_x * r1;
        let start_y = y1 + dir_y * r1;
        let end_x = x2 - dir_x * r2;
        let end_y = y2 - dir_y * r2;

        // Draw arrowhead
        let arrow_size = 10.0;
        let angle: f32 = 0.5;

        let sin_a = angle.sin();
        let cos_a = angle.cos();

        let arrow_dx1 = dir_x * cos_a - dir_y * sin_a;
        let arrow_dy1 = dir_x * sin_a + dir_y * cos_a;

        let arrow_dx2 = dir_x * cos_a + dir_y * sin_a;
        let arrow_dy2 = -dir_x * sin_a + dir_y * cos_a;

        // Return midpoint
        let mid_x = (start_x + end_x) / 2.0;
        let mid_y = (start_y + end_y) / 2.0;

        Arrow {
            id,
            from: Point {
                x: start_x,
                y: start_y,
            },
            from_1: Point {
                x: end_x - arrow_dx1 * arrow_size,
                y: end_y - arrow_dy1 * arrow_size,
            },
            from_2: Point {
                x: end_x - arrow_dx2 * arrow_size,
                y: end_y - arrow_dy2 * arrow_size,
            },
            to: Point { x: end_x, y: end_y },
            middle: Point { x: mid_x, y: mid_y },
        }
    }
    
    pub fn generate_id(from: &String, to: &String) -> String {
        format!("{}->{}", from, to)
    }
}

pub struct PositioningView {
    pub circles: Vec<NCircle>,
    pub arrows: Vec<Arrow>,
}

pub struct NValue {
    pub id: String,
    pub input: f32,
    pub value: f32,
    pub error: f32    
}
pub struct LValue {
    pub id: String,
    pub value: f32
}
pub struct Model {
    pub neuron_values: Vec<NValue>,
    pub link_values: Vec<LValue>,
}

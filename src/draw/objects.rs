
pub const WINDOW_WIDTH: usize = 1366;
pub const WINDOW_HEIGHT: usize = 768;
pub const CIRCLE_RADIUS: usize = 140;

pub const COLOUR_BACKGROUND: u32 = 0x1e1f22;
pub const COLOUR_CIRCLE: u32 = 0xce7b47;
pub const COLOUR_LINK: u32 = 0xb7babf;
pub const COLOUR_ERROR: u32 = 0xc37ab6;

pub struct Circle {
    pub id: String,
    pub x: f32,
    pub y: f32,
    pub radius: f32,
    pub color: u32, // RGBA
}
pub struct Link {
    pub src_id: String,
    pub dst_id: String,
    pub x: f32,
    pub y: f32,
}
pub struct Layer {    
    pub circles: Vec<Circle>,
}
pub struct PositioningView {
    pub layers: Vec<Layer>,
    pub links: Vec<Link>
}


pub struct Model {

}
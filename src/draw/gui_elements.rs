use macroquad::color::Color;
use macroquad::prelude::{draw_rectangle, draw_text_ex, measure_text, Rect, Vec2};
use crate::draw::font_objects::TextStyles;
use crate::draw::objects::{Point, COLOUR_LINK};

const ACTIVE_COLOUR : Color = Color::from_hex(COLOUR_LINK);
const INACTIVE_COLOUR : Color = Color::from_hex(COLOUR_LINK).with_alpha(0.3);

pub struct Button {
    label: String,
    pub rect: Rect,
    pub active: bool
}

impl Button {
    pub fn new(label: String, point: Point, text_style: &TextStyles) -> Self {
        let text_params = text_style.button();
        let dims = measure_text(&label, text_params.font, text_params.font_size, 1.0);
        let text_width = dims.width;
        let text_height = dims.height;
        Button { label, rect: Rect::new(point.x, point.y, text_width+20.0, text_height+12.0), active: false }
    }
    
    
    pub fn draw(&self, text_style: &TextStyles) {
        let color = if self.active { ACTIVE_COLOUR } else { INACTIVE_COLOUR };
        draw_rectangle(self.rect.x, self.rect.y, self.rect.w, self.rect.h, color);
        draw_text_ex(&self.label,
                     self.rect.x + 10.0,
                     self.rect.y + self.rect.h / 2.0 + 6.0,
                     text_style.button());
    }

    pub fn is_clicked(&self, mouse_pos: Vec2) -> bool {
        self.rect.contains(mouse_pos)
    }
}
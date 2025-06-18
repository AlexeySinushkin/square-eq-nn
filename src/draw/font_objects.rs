use macroquad::color::Color;
use macroquad::prelude::TextParams;
use macroquad::text::Font;
use crate::draw::objects::{COLOUR_BACKGROUND, COLOUR_CIRCLE, COLOUR_ERROR, COLOUR_LINK};

pub struct TextStyles {
    pub font: Font,
}

impl TextStyles {
    pub fn neuron_header(&self) -> TextParams {
        TextParams {
            font: Some(&self.font),
            font_size: 16,
            color: Color::from_hex(COLOUR_LINK),
            ..Default::default()
        }
    }

    pub fn neuron_error(&self) -> TextParams {
        TextParams {
            font: Some(&self.font),
            font_size: 16,
            color: Color::from_hex(COLOUR_ERROR),
            ..Default::default()
        }
    }

    pub fn link_weight(&self) -> TextParams {
        TextParams {
            font: Some(&self.font),
            font_size: 16,
            color: Color::from_hex(COLOUR_CIRCLE),
            ..Default::default()
        }
    }
    pub fn button(&self) -> TextParams {
        TextParams {
            font: Some(&self.font),
            font_size: 24,
            color: Color::from_hex(COLOUR_BACKGROUND),
            ..Default::default()
        }
    }
}
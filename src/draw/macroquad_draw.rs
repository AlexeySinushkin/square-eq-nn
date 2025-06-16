use crate::draw::font_objects::TextStyles;
use crate::draw::objects::{
    COLOUR_BACKGROUND, COLOUR_CIRCLE, COLOUR_LINK, Model, PositioningView, WINDOW_HEIGHT,
    WINDOW_WIDTH,
};
use macroquad::prelude::*;
use std::sync::mpsc::{self, Receiver};
use std::thread;
use std::time::Duration;

// Function that runs macroquad main loop
fn spawn_ui_thread(view: PositioningView, rx: Receiver<Model>) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        // You must call macroquad via this attribute in a standalone thread:
        macroquad::Window::from_config(
            Conf {
                window_title: "Macroquad Thread Example".into(),
                window_width: WINDOW_WIDTH as i32,
                window_height: WINDOW_HEIGHT as i32,
                ..Default::default()
            },
            async move {
                let mut message = "Hello, World!".to_string();
                let font = load_ttf_font("assets/arial.ttf")
                    .await
                    .expect("Failed to load Arial font");
                let text_styles = TextStyles { font };
                loop {
                    // Try to receive updated text
                    if let Ok(new_msg) = rx.try_recv() {
                        //message = new_msg;
                    }
                    draw_background(&view, &text_styles);

                    //draw_text_ex(&message, 30.0, 100.0, text_styles.neuron_header());

                    next_frame().await;
                }
            },
        );
    })
}

fn draw_text_center(text: &str, x: usize, y: usize, text_style: &TextStyles) {
    let text_params = text_style.neuron_header();
    let dims = measure_text(text, text_params.font, text_params.font_size, 1.0);
    let text_x = x as f32 - dims.width / 2.0;
    let text_y = y as f32 + dims.offset_y / 2.0; // Optional: vertically center

    draw_text_ex(text, text_x, text_y, text_params);
}

fn draw_background(view: &PositioningView, text_style: &TextStyles) {
    clear_background(Color::from_hex(COLOUR_BACKGROUND));
    for circle in view.circles.iter() {
        draw_circle(
            circle.x as f32,
            circle.y as f32,
            circle.radius as f32,
            Color::from_hex(COLOUR_CIRCLE),
        );

        draw_circle(
            circle.x as f32,
            circle.y as f32,
            (circle.radius - 2) as f32,
            Color::from_hex(COLOUR_BACKGROUND),
        );
        draw_text_center(
            &circle.id,
            circle.x,
            circle.y - circle.radius / 2,
            text_style,
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::draw::macroquad_draw::spawn_ui_thread;
    use crate::draw::objects::Model;
    use crate::draw::view::build_view;
    use crate::nn_build::build_nn;
    use std::sync::mpsc;

    #[test]
    #[ignore]
    fn draw_test() {
        let nn = build_nn();
        let view = build_view(&nn);

        let (tx, rx) = mpsc::channel::<Model>();
        let join_handle = spawn_ui_thread(view, rx);
        join_handle.join().unwrap();
    }
}

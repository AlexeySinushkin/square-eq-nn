use crate::draw::font_objects::TextStyles;
use crate::draw::objects::{Arrow, COLOUR_BACKGROUND, COLOUR_CIRCLE, COLOUR_LINK, Model, NCircle, Point, PositioningView, WINDOW_HEIGHT, WINDOW_WIDTH};
use macroquad::prelude::*;
use std::sync::mpsc::{Receiver, Sender};
use std::thread;
use crate::draw::gui_elements::Button;
use crate::execution_objects::Events;

// Function that runs macroquad main loop
pub(crate) fn spawn_ui_thread(view: PositioningView, rx: Receiver<Model>, tx: Sender<Events>) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        // You must call macroquad via this attribute in a standalone thread:
        macroquad::Window::from_config(
            Conf {
                window_title: "Neural Network State".into(),
                window_width: WINDOW_WIDTH as i32,
                window_height: WINDOW_HEIGHT as i32,
                ..Default::default()
            },
            async move {
                let font = load_ttf_font("assets/arial.ttf")
                    .await
                    .expect("Failed to load Arial font");
                let text_styles = TextStyles { font };
                let mut model : Option<Model> = None;
                let iteration_point = Point{x: 100.0, y: 20.0};
                let point = Point { x: WINDOW_WIDTH as f32 / 2.0, y: WINDOW_HEIGHT as f32 - 100.0 };
                let mut pause_button = Button::new("PAUSE".to_string(), point.clone(), &text_styles);

                let mut stepping_button = Button::new("STEPPING".to_string(),
                                                      Point { x: pause_button.rect.x + pause_button.rect.w + 10.0, y: point.y },
                                                      &text_styles);
                let mut play_button = Button::new("PLAY".to_string(),
                                                  Point { x: stepping_button.rect.x + stepping_button.rect.w + 10.0, y: point.y },
                                                  &text_styles);
                loop {

                    draw_background(&view, &text_styles);
                    if let Ok(new_msg) = rx.try_recv() {
                        model = Some(new_msg);
                    }
                    if let Some(model) = model.as_ref() {
                        draw_values(&view, &model,  &text_styles);
                        
                        let iteration = format!("{}", model.iterations);
                        draw_text_center(&iteration, &iteration_point, text_styles.neuron_error());
                        pause_button.active = model.button_pause_active;                        
                        stepping_button.active = model.button_stepping_active;
                        play_button.active = model.button_play_active;
                        pause_button.draw(&text_styles);
                        stepping_button.draw(&text_styles);
                        play_button.draw(&text_styles);
                    }
                    
                    let mouse_pos = mouse_position().into();
                    if is_mouse_button_pressed(MouseButton::Left) {
                        if pause_button.is_clicked(mouse_pos) {
                            tx.send(Events::PauseRequested).unwrap();;
                        }
                        if stepping_button.is_clicked(mouse_pos) {
                            tx.send(Events::SteppingRequested).unwrap();;
                        }
                        if play_button.is_clicked(mouse_pos) {
                            tx.send(Events::PlayRequested).unwrap();;
                        }
                    }
                    next_frame().await;
                }
            },
        );
    })
}

fn draw_values(view: &PositioningView, model: &Model, text_style: &TextStyles) {
    for circle_value in model.neuron_values.iter() {
        if let Some(circle) = view.circles.iter().find(|c| c.id == circle_value.id) {
            let error = format!("{:.5}", circle_value.error);
            draw_text_center(&error, &circle.center, text_style.neuron_error());
            let output = format!("{:.2}", circle_value.value);
            draw_text_center(&output, &circle.output, text_style.neuron_header())
        }
    }
    for link_value in model.link_values.iter() {
        if let Some(arrow) = view.arrows.iter().find(|c| c.id == link_value.id) {
            let weight = format!("{:.5}", link_value.value);
            draw_text_center(&weight, &arrow.middle, text_style.link_weight())
        }
    }
}

pub fn draw_text_center(text: &str, point: &Point, text_params: TextParams) {
    // Measure the text size
    let dims = measure_text(text, text_params.font, text_params.font_size, 1.0);
    let text_width = dims.width;
    let text_height = dims.height;

    // Calculate top-left corner for background and text
    let text_x = point.x - text_width / 2.0;
    let text_y = point.y + dims.offset_y / 2.0;

    // Draw background rectangle (with padding)
    let padding = 2.0;
    draw_rectangle(
        text_x - padding,
        text_y - text_height - padding,
        text_width + 2.0 * padding,
        text_height + 2.0 * padding,
        Color::from_hex(COLOUR_BACKGROUND).with_alpha(0.5), 
    );

    // Draw the text over it
    draw_text_ex(text, text_x, text_y, text_params);
}

fn draw_background(view: &PositioningView, text_style: &TextStyles) {
    clear_background(Color::from_hex(COLOUR_BACKGROUND));
    for circle in view.circles.iter() {
        draw_neuron_circle(circle, text_style);
    }
    for arrow in view.arrows.iter() {
        draw_arrow(arrow);
    }
}
fn draw_neuron_circle(circle: &NCircle, text_style: &TextStyles) {
    draw_circle(
        circle.center.x as f32,
        circle.center.y as f32,
        circle.radius as f32,
        Color::from_hex(COLOUR_CIRCLE),
    );

    draw_circle(
        circle.center.x as f32,
        circle.center.y as f32,
        (circle.radius - 2.0) as f32,
        Color::from_hex(COLOUR_BACKGROUND),
    );
    draw_text_center(&circle.id, &circle.caption, text_style.neuron_header());
}

fn draw_arrow(arrow: &Arrow) {
    let color: Color = Color::from_hex(COLOUR_LINK);
    draw_line(
        arrow.from.x,
        arrow.from.y,
        arrow.to.x,
        arrow.to.y,
        2.0,
        color,
    );
    draw_line(
        arrow.from_1.x,
        arrow.from_1.y,
        arrow.to.x,
        arrow.to.y,
        2.0,
        color,
    );
    draw_line(
        arrow.from_2.x,
        arrow.from_2.y,
        arrow.to.x,
        arrow.to.y,
        2.0,
        color,
    );
}

#[cfg(test)]
mod tests {
    use crate::draw::macroquad_draw::spawn_ui_thread;
    use crate::draw::objects::{LValue, Model, NValue};
    use crate::draw::view::build_view;
    use crate::nn_build::build_nn;
    use rand::Rng;
    use std::sync::mpsc;
    use std::thread::sleep;
    use std::time::Duration;

    #[test]
    #[ignore]
    fn draw_test() {
        let nn = build_nn();
        let view = build_view(&nn);
        let mut neuron_values: Vec<NValue> = vec![];
        let mut link_values: Vec<LValue> = vec![];
        let mut rng = rand::rng();

        for n in view.circles.iter() {
            neuron_values.push(NValue {
                id: n.id.clone(),
                input: rng.random(),
                value: rng.random(),
                error: rng.random(),
            })
        }
        for l in view.arrows.iter() {
            link_values.push(LValue {
                id: l.id.clone(),
                value: rng.random(),
            })
        }

        let (tx, rx) = mpsc::channel::<Model>();
        let join_handle = spawn_ui_thread(view, rx);
        sleep(Duration::from_secs(3));
        tx.send(Model { neuron_values, link_values, iterations: 0 }).unwrap();
        
        join_handle.join().unwrap();
    }
}

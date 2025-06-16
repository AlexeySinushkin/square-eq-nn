use crate::draw::objects::{
    CIRCLE_RADIUS, COLOUR_BACKGROUND, COLOUR_CIRCLE, COLOUR_LINK, Circle, Link, Model,
    PositioningView, WINDOW_HEIGHT, WINDOW_WIDTH,
};
use crate::nn_objects::Network;
use minifb::{Key, Window, WindowOptions};
use std::sync::mpsc;
use std::thread::sleep;
use std::time::Duration;

pub fn init(
    buffer_background: Vec<u32>,
    rx: mpsc::Receiver<Model>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut window = Window::new(
        "Square equation neural network",
        WINDOW_WIDTH,
        WINDOW_HEIGHT,
        WindowOptions::default(),
    )
    .unwrap();

    while window.is_open() && !window.is_key_down(Key::Escape) {
        let mut buffer = buffer_background.clone();
        // Display buffer
        window
            .update_with_buffer(&buffer, WINDOW_WIDTH, WINDOW_HEIGHT)
            .unwrap();

        // Limit to ~25 FPS
        sleep(Duration::from_millis(40));
    }
    Ok(())
}

pub fn build_view(nn: &Network) -> PositioningView {
    let mut circles: Vec<Circle> = vec![];
    let links: Vec<Link> = vec![];
    let layer_width = WINDOW_WIDTH / nn.layers_count;
    let layer_height = WINDOW_HEIGHT * 8 / 10;
    let padding_top = layer_height / 10;
    let circle_radius = layer_width / 6;
    let mut x = layer_width / 2;
    for layer in nn.layers.iter() {
        let layer_height = layer_height - (padding_top * 2);
        let neurons_count = layer.neurons.iter().filter(|n| !n.is_dummy()).count();
        if neurons_count == 0 {
            break;
        }        
        let neuron_space = layer_height / neurons_count;
        let mut y = padding_top + neuron_space / 2;
        for neuron in layer.neurons.iter().filter(|n| !n.is_dummy()) {
            circles.push(Circle {
                id: neuron.id.clone(),
                x,
                y,
                radius: circle_radius,
                color: COLOUR_CIRCLE,
            });
            y += neuron_space
        }
        x += layer_width;
    }
    PositioningView { circles, links }
}

pub fn build_backgroud(view: &PositioningView) -> Vec<u32> {
    // Create a 2D array (canvas)
    let mut canvas = [[COLOUR_BACKGROUND; WINDOW_WIDTH]; WINDOW_HEIGHT];
    
    // Convert 2D canvas into a flat Vec<u32> buffer
    let mut buffer = Vec::<u32>::with_capacity(WINDOW_WIDTH * WINDOW_HEIGHT);

    for row in &canvas {
        buffer.extend_from_slice(row);
    }

    for circle in view.circles.iter() {
        draw_circle(&mut buffer, circle);
    }

    buffer
}

fn draw_circle(buffer: &mut [u32], circle: &Circle) {
    let mut radius = circle.radius as i32;
    let cx = circle.x as i32;
    let cy = circle.y as i32;
    let mut r2 = radius * radius;

    for y in -radius..=radius {
        for x in -radius..=radius {
            if x * x + y * y <= r2 {
                let px = cx + x;
                let py = cy + y;
                if px >= 0
                    && py >= 0
                    && (px as usize) < WINDOW_WIDTH
                    && (py as usize) < WINDOW_HEIGHT
                {
                    buffer[py as usize * WINDOW_WIDTH + px as usize] = circle.color;
                }
            }
        }
    }
    radius -= 2;
    r2 = radius * radius;
    for y in -radius..=radius {
        for x in -radius..=radius {
            if x * x + y * y <= r2 {
                let px = cx + x;
                let py = cy + y;
                if px >= 0
                    && py >= 0
                    && (px as usize) < WINDOW_WIDTH
                    && (py as usize) < WINDOW_HEIGHT
                {
                    buffer[py as usize * WINDOW_WIDTH + px as usize] = COLOUR_BACKGROUND;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::draw::draw::{build_backgroud, build_view, init};
    use crate::draw::objects::Model;
    use crate::nn_build::build_nn;
    use std::sync::mpsc;

    #[test]
    #[ignore]
    fn draw_test() {
        let nn = build_nn();
        let view = build_view(&nn);
        let background = build_backgroud(&view);
        let (tx, rx) = mpsc::channel::<Model>();
        init(background, rx);
    }
}

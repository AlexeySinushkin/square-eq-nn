use std::sync::mpsc;
use std::thread::sleep;
use std::time::Duration;
use minifb::{Key, Window, WindowOptions};
use crate::draw::objects::{Circle, Layer, Link, Model, PositioningView, CIRCLE_RADIUS, COLOUR_BACKGROUND, COLOUR_CIRCLE, COLOUR_LINK, WINDOW_HEIGHT, WINDOW_WIDTH};
use crate::nn_objects::Network;

pub fn init(buffer_background: Vec<u32>, rx: mpsc::Receiver<Model>) -> Result<(), Box<dyn std::error::Error>> {    

    let mut window = Window::new("Square equation neural network", WINDOW_WIDTH, WINDOW_HEIGHT, WindowOptions::default())
        .unwrap();

    while window.is_open() && !window.is_key_down(Key::Escape) {
        let mut buffer = buffer_background.clone();
        // Draw circle at center (100, 100) with radius 40
        draw_circle(&mut buffer, 150, 100, 70, COLOUR_CIRCLE);
        draw_circle(&mut buffer, 150, 100, 68, COLOUR_BACKGROUND);
        // Display buffer
        window
            .update_with_buffer(&buffer, WINDOW_WIDTH, WINDOW_HEIGHT)
            .unwrap();

        // Limit to ~25 FPS
        sleep(Duration::from_millis(40));
    }
    Ok(())
}


pub fn build_view(nn: &Network) -> PositioningView
{
    let layers: Vec<Layer> = vec![];
    let links: Vec<Link> = vec![];
    
    PositioningView { layers, links }
}

pub fn build_backgroud(view: &PositioningView) -> Vec<u32> {
    // Create a 2D array (canvas)
    let mut canvas = [[COLOUR_BACKGROUND; WINDOW_WIDTH]; WINDOW_HEIGHT];
    
    canvas[1][1] = COLOUR_LINK;
    // Convert 2D canvas into a flat Vec<u32> buffer
    let mut buffer = Vec::<u32>::with_capacity(WINDOW_WIDTH * WINDOW_HEIGHT);
    for row in &canvas {
        buffer.extend_from_slice(row);
    }
    buffer
}

fn draw_circle(buffer: &mut [u32], cx: i32, cy: i32, radius: i32, colour: u32) {
    let r2 = radius * radius;

    for y in -radius..=radius {
        for x in -radius..=radius {
            if x * x + y * y <= r2 {
                let px = cx + x;
                let py = cy + y;
                if px >= 0 && py >= 0 && (px as usize) < WINDOW_WIDTH && (py as usize) < WINDOW_HEIGHT {
                    buffer[py as usize * WINDOW_WIDTH + px as usize] = colour; // Blue
                }
            }
        }
    }
}


#[cfg(test)]
mod tests {
    use crate::nn_build::build_nn;
    use std::sync::mpsc;
    use crate::draw::draw::{build_backgroud, build_view, init};
    use crate::draw::objects::Model;

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
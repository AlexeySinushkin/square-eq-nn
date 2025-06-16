use crate::draw::objects::{Circle, Link, PositioningView, COLOUR_CIRCLE, WINDOW_HEIGHT, WINDOW_WIDTH};
use crate::nn_objects::Network;

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
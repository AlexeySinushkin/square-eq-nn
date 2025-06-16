use crate::draw::objects::{
    Arrow, NCircle, PositioningView, WINDOW_HEIGHT, WINDOW_WIDTH,
};
use crate::nn_objects::Network;

pub fn build_view(nn: &Network) -> PositioningView {
    let mut circles: Vec<NCircle> = vec![];
    let mut arrows: Vec<Arrow> = vec![];
    let layer_width = (WINDOW_WIDTH / nn.layers_count) as f32;
    let layer_height = (WINDOW_HEIGHT as f32 * 0.8) as f32;
    let padding_top = (layer_height / 10.0) as f32;
    let circle_radius = layer_width as f32 / 6.0;
    let mut x = layer_width as f32 / 2.0;

    for layer in nn.layers.iter() {
        let layer_height = layer_height - (padding_top * 2.0);
        let neurons_count = layer.neurons.iter().filter(|n| !n.is_dummy()).count();
        if neurons_count == 0 {
            break;
        }
        let neuron_space = (layer_height / neurons_count as f32) as f32;
        let mut y = (padding_top + neuron_space / 2.0) as f32;
        for neuron in layer.neurons.iter().filter(|n| !n.is_dummy()) {
            let id = neuron.id.clone();
            circles.push(NCircle::new(id, x, y, circle_radius));
            y += neuron_space
        }
        x += layer_width;

        for neuron in layer.neurons.iter().filter(|n| !n.is_dummy()) {
            for link in neuron.input_links.iter().filter(|l| !l.is_dummy()) {
                let circle_from = circles.iter().find(|c| c.id == link.source_id).unwrap();
                let circle_to = circles.iter().find(|c| c.id == neuron.id).unwrap();
                let id = format!("{}->{}", link.source_id, neuron.id);
                arrows.push(Arrow::new(id, circle_from, circle_to))
            }
        }
    }
    PositioningView { circles, arrows }
}

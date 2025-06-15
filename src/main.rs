mod nn_objects;

use crate::nn_objects::{Layer, Link, Neuron};
use serde::Deserialize;
use serde_json;
use std::fs;

#[derive(Deserialize, Debug)]
struct TrainItem {
    a: f32,
    b: f32,
    c: f32,
    x1: f32,
    x2: f32,
}
const LAYERS_COUNT: usize = 4;
type Network = [Layer; LAYERS_COUNT];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let train_file_content = fs::read_to_string("./train.json")?;
    let train_items: Vec<TrainItem> = serde_json::from_str(&train_file_content)?;

    let mut nn = build_nn();

    for epoch in 0..100 {
        for item in &train_items {
            forward(&mut nn, &item);
            backward(&mut nn, &item, 0.1);
        }
        let test_item = train_items.first().unwrap();
        let x1_output = nn.last().unwrap().neurons[0].output;
        let x2_output = nn.last().unwrap().neurons[1].output;
        forward(&mut nn, test_item);
        let error_x1 = loss(test_item.x1, x1_output);
        let error_x2 = loss(test_item.x2, x2_output);
        println!("epoch {epoch}: x1 error =  {error_x1}, x2 error = {error_x2}");
    }


    Ok(())
}
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
fn sigmoid_derivative(x: f32) -> f32 {
    let s = sigmoid(x);
    s * (1.0 - s)
}
fn loss(target: f32, value: f32) -> f32 {
    (target - value).powi(2)
}
fn forward(nn: &mut Network, train_item: &TrainItem) {
    nn[0].neurons[0].output = train_item.a;
    nn[0].neurons[1].output = train_item.b;
    nn[0].neurons[3].output = train_item.c;

    for layer_index in 1..LAYERS_COUNT {
        let (prev, current) = nn.split_at_mut(layer_index);
        let prev_layer = &prev[layer_index - 1];
        let current_layer = &mut current[0];

        for neuron in &mut current_layer.neurons {
            if !neuron.is_dummy() {
                let sum: f32 = neuron
                    .input_links
                    .iter()
                    .filter(|link| !link.is_dummy())
                    .map(|link| {
                        return link.weight * prev_layer.get_value(&link.source_id);
                    })
                    .fold(0.0, |acc, e| acc + e);
                neuron.sum_input = sum;
                neuron.output = sigmoid(sum); 
            }
        }
    }
}

fn backward(nn: &mut Network, train_item: &TrainItem, learning_rate: f32) {
    let output_neuron_x1 = &nn[LAYERS_COUNT - 1].neurons[0];
    let output_neuron_x2 = &nn[LAYERS_COUNT - 1].neurons[1];
    
    let e_x1 = (train_item.x1 - output_neuron_x1.output)
        * sigmoid_derivative(output_neuron_x1.sum_input);
    let e_x2 = (train_item.x2 - output_neuron_x2.output)
        * sigmoid_derivative(output_neuron_x2.sum_input);

    nn[LAYERS_COUNT - 1].neurons[0].error = e_x1;
    nn[LAYERS_COUNT - 1].neurons[1].error = e_x2;
      
    for layer_index in (1..LAYERS_COUNT).rev() {
        let (prev, current) = nn.split_at_mut(layer_index);
        let prev_layer = &mut prev[layer_index - 1];
        let current_layer = &mut current[0];
        //weight updates
        for neuron in &mut current_layer.neurons {
            if !neuron.is_dummy() {
                for link in neuron.input_links.iter_mut() {
                    if !link.is_dummy() {
                        let derive = sigmoid_derivative(neuron.sum_input);
                        let delta = neuron.error * derive * prev_layer.get_value(&link.source_id);
                        link.weight += delta * learning_rate;                        
                    }
                }                
            }
        }
        //error updates
        for prev_neuron in &mut prev_layer.neurons {
            if prev_neuron.is_dummy() {
                continue;
            }

            let mut error_sum = 0.0;
            for neuron in &current_layer.neurons {
                if neuron.is_dummy() {
                    continue;
                }
                for link in &neuron.input_links {
                    if link.source_id == prev_neuron.id {
                        error_sum += link.weight * neuron.error;
                    }
                }
            }
            prev_neuron.error = error_sum * sigmoid_derivative(prev_neuron.sum_input);
        }
    }
}

fn build_nn() -> Network {
    let a = Neuron::new_input("a".to_string());
    let b = Neuron::new_input("b".to_string());
    let c = Neuron::new_input("c".to_string());
    let input_layer = Layer {
        neurons: [a, b, c, Neuron::new_dummy()],
    };

    let a_m1 = Link::new("a".to_string(), 0.1);
    let b_m1 = Link::new("b".to_string(), 0.1);
    let c_m1 = Link::new("c".to_string(), 0.1);
    let m1 = Neuron::new_middle(
        "m1".to_string(),
        0.23,
        [a_m1, b_m1, c_m1, Link::new_dummy()],
    );

    let a_m2 = Link::new("a".to_string(), 0.1);
    let b_m2 = Link::new("b".to_string(), 0.1);
    let c_m2 = Link::new("c".to_string(), 0.1);
    let m2 = Neuron::new_middle(
        "m2".to_string(),
        0.24,
        [a_m2, b_m2, c_m2, Link::new_dummy()],
    );

    let a_m3 = Link::new("a".to_string(), 0.1);
    let b_m3 = Link::new("b".to_string(), 0.1);
    let c_m3 = Link::new("c".to_string(), 0.1);
    let m3 = Neuron::new_middle(
        "m3".to_string(),
        0.24,
        [a_m3, b_m3, c_m3, Link::new_dummy()],
    );

    let a_m4 = Link::new("a".to_string(), 0.1);
    let b_m4 = Link::new("b".to_string(), 0.1);
    let c_m4 = Link::new("c".to_string(), 0.1);
    let m4 = Neuron::new_middle(
        "m4".to_string(),
        0.24,
        [a_m4, b_m4, c_m4, Link::new_dummy()],
    );

    let layer_m = Layer {
        neurons: [m1, m2, m3, m4],
    };

    let m1_n1 = Link::new("m1".to_string(), 0.1);
    let m2_n1 = Link::new("m2".to_string(), 0.1);
    let m3_n1 = Link::new("m3".to_string(), 0.1);
    let m4_n1 = Link::new("m4".to_string(), 0.1);
    let n1 = Neuron::new_middle("n1".to_string(), 0.23,  [m1_n1, m2_n1, m3_n1, m4_n1]);

    let m1_n2 = Link::new("m1".to_string(), 0.1);
    let m2_n2 = Link::new("m2".to_string(), 0.1);
    let m3_n2 = Link::new("m3".to_string(), 0.1);
    let m4_n2 = Link::new("m4".to_string(), 0.1);
    let n2 = Neuron::new_middle("n2".to_string(), 0.23,  [m1_n2, m2_n2, m3_n2, m4_n2]);

    let m1_n3 = Link::new("m1".to_string(), 0.1);
    let m2_n3 = Link::new("m2".to_string(), 0.1);
    let m3_n3 = Link::new("m3".to_string(), 0.1);
    let m4_n3 = Link::new("m4".to_string(), 0.1);
    let n3 = Neuron::new_middle("n3".to_string(), 0.23,  [m1_n3, m2_n3, m3_n3, m4_n3]);

    let layer_n = Layer {
        neurons: [n1, n2, n3, Neuron::new_dummy()],
    };

    let n1_x1 = Link::new("n1".to_string(), 0.1);
    let n2_x1 = Link::new("n2".to_string(), 0.1);
    let n3_x1 = Link::new("n3".to_string(), 0.1);
    let x1 = Neuron::new_middle(
        "x1".to_string(),
        0.23,
        [n1_x1, n2_x1, n3_x1, Link::new_dummy()],
    );

    let n1_x2 = Link::new("n1".to_string(), 0.1);
    let n2_x2 = Link::new("n2".to_string(), 0.1);
    let n3_x2 = Link::new("n3".to_string(), 0.1);
    let x2 = Neuron::new_middle(
        "x2".to_string(),
        0.23,
        [n1_x2, n2_x2, n3_x2, Link::new_dummy()],
    );

    let output_layer = Layer {
        neurons: [x1, x2, Neuron::new_dummy(), Neuron::new_dummy()],
    };

    [input_layer, layer_m, layer_n, output_layer]
}

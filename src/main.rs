mod nn_objects;
mod nn_build;
mod train_data;
mod activation_functions;
mod draw;

use crate::nn_objects::Network;
use crate::nn_build::build_nn;
use crate::train_data::{load_train, shuffle, TrainItem};




fn main() -> Result<(), Box<dyn std::error::Error>> {
    
    let mut train_items: Vec<TrainItem> = load_train()?;
    let mut nn = build_nn();

    let mut learning_rate = 0.1;
    for epoch in 0..10000 {
        shuffle(&mut train_items);
        
        for item in & train_items {
            forward(&mut nn, &item);
            backward(&mut nn, &item, learning_rate);
        }

        let mut total_loss_x1 = 0.0;
        let mut total_loss_x2 = 0.0;

        for item in &train_items {
            forward(&mut nn, item);
            let x1_out = nn.last().neurons[0].output;
            let x2_out = nn.last().neurons[1].output;
            total_loss_x1 += loss(item.x1, x1_out);
            total_loss_x2 += loss(item.x2, x2_out);
        }

        let avg_loss_x1 = total_loss_x1 / train_items.len() as f32;
        let avg_loss_x2 = total_loss_x2 / train_items.len() as f32;        
        
        println!("epoch {epoch}: avg x1 error = {avg_loss_x1}, avg x2 error = {avg_loss_x2}");
        
        if learning_rate > 0.01{
            learning_rate /= 2.0;
        }
    }

    Ok(())
}

fn loss(target: f32, value: f32) -> f32 {
    (target - value).powi(2)
}



fn forward(nn: &mut Network, train_item: &TrainItem) {
    nn.layers[0].neurons[0].output = train_item.a;
    nn.layers[0].neurons[1].output = train_item.b;
    nn.layers[0].neurons[3].output = train_item.c;

    for layer_index in 1..nn.layers_count {
        let (prev, current) = nn.layers.split_at_mut(layer_index);
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
                neuron.output = activation_functions::apply(&neuron.function_name, sum); 
            }
        }
    }
}

fn backward(nn: &mut Network, train_item: &TrainItem, learning_rate: f32) {
    let output_neuron_x1 = &nn.layers[nn.layers_count - 1].neurons[0];
    let output_neuron_x2 = &nn.layers[nn.layers_count - 1].neurons[1];
    
    let e_x1 = (train_item.x1 - output_neuron_x1.output)
        * activation_functions::derivative(&output_neuron_x1.function_name, output_neuron_x1.sum_input);
    let e_x2 = (train_item.x2 - output_neuron_x2.output)
        * activation_functions::derivative(&output_neuron_x2.function_name, output_neuron_x2.sum_input);

    nn.layers[nn.layers_count - 1].neurons[0].error = e_x1;
    nn.layers[nn.layers_count - 1].neurons[1].error = e_x2;
      
    for layer_index in (1..nn.layers_count).rev() {
        let (prev, current) = nn.layers.split_at_mut(layer_index);
        let prev_layer = &mut prev[layer_index - 1];
        let current_layer = &mut current[0];
        //weight updates
        for neuron in &mut current_layer.neurons {
            if !neuron.is_dummy() {
                for link in neuron.input_links.iter_mut() {
                    if !link.is_dummy() {
                        let derive = activation_functions::derivative(&neuron.function_name, neuron.sum_input);
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
            prev_neuron.error = error_sum * activation_functions::derivative(&prev_neuron.function_name, prev_neuron.sum_input);
        }
    }
}

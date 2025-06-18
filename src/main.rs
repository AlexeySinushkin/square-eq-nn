mod activation_functions;
mod draw;
mod draw_adapter;
mod execution_objects;
mod nn_build;
mod nn_objects;
mod train_data;

use crate::activation_functions::{apply, derivative};
use crate::draw::macroquad_draw::spawn_ui_thread;
use crate::draw::objects::Model;
use crate::draw::view::build_view;
use crate::draw_adapter::DrawAdapter;
use crate::execution_objects::{Events, ExecutionObjects, RunMode};
use crate::nn_build::{build_nn, build_nn1};
use crate::nn_objects::Network;
use rand::prelude::ThreadRng;
use rand::{Rng, rng};
use std::io::Error;
use std::sync::mpsc;
use std::sync::mpsc::{Receiver, Sender};
use std::thread::sleep;
use std::time::{Duration, Instant};

const STEPPING_DURATION: Duration = Duration::from_millis(1000);

fn main() -> Result<(), Box<dyn std::error::Error>> {
    //let nn_json = fs::read_to_string("./neural-networks/kx_b/nn2.json")?;
    //let mut nn: Network = serde_json::from_str(&nn_json)?;
    let mut nn: Network = build_nn1();

    let (tx_data, rx_data) = mpsc::channel::<Model>();
    let (tx_events, rx_events) = mpsc::channel::<Events>();
    let view = build_view(&nn);
    let join_handle = spawn_ui_thread(view, rx_data, tx_events);
    let mut adapter = DrawAdapter::new(tx_data);

    let mut learning_rate = 0.1;
    let mut rng = rand::rng();
    let mut iteration = 0;
    let mut last_step = Instant::now();
    let mut run_mode = RunMode::Pause;

    let k: f32 = rng.random_range(-10..10) as f32;
    let x: f32 = rng.random_range(-10..10) as f32;
    let b: f32 = rng.random_range(-10..10) as f32;
    let y = k * x + b;    
    let mut execution = ExecutionContext {
        nn,
        iteration,
        error: 1.0,
        last_step,
        learning_rate,
        rng,
        run_mode,
        tx_adapter: adapter,
        rx_events,
    };

    loop {
        execution.train_loop(k, x, b, y).expect("correct train loop");
        iteration += 1;
        if execution.error < 0.1 {
            break;
        }
    }

    join_handle.join().unwrap();
    Ok(())
}

struct ExecutionContext {
    nn: Network,
    iteration: usize,
    error: f32,
    last_step: Instant,
    learning_rate: f32,
    rng: ThreadRng,
    run_mode: RunMode,
    tx_adapter: DrawAdapter,
    rx_events: Receiver<Events>,
}

impl ExecutionContext {
    pub fn train_loop(&mut self, k: f32, x: f32, b: f32, y: f32) -> Result<(), Box<dyn std::error::Error>> {
/*
        let k: f32 = self.rng.random_range(-10..10) as f32;
        let x: f32 = self.rng.random_range(-10..10) as f32;
        let b: f32 = self.rng.random_range(-10..10) as f32;
        let y = k * x + b;*/

        self.nn.layers[0].neurons[0].output = k;
        self.nn.layers[0].neurons[1].output = x;
        self.nn.layers[0].neurons[2].output = b;
        self.forward();
        self.send_state();
        self.hang_out();

        let y_neuron = &mut self.nn.layers[self.nn.layers_count - 1].neurons[0];
        y_neuron.error = Self::loss(y, y_neuron.output);
        println!("y error: {}", y_neuron.error);

        self.backward();
        self.send_state();
        self.hang_out();

        self.iteration += 1;
        if self.learning_rate > 0.001 {
            self.learning_rate *= 0.999;
        }
        Ok(())
    }

    fn forward(&mut self) {
        for layer_index in 1..self.nn.layers_count {
            let (prev, current) = self.nn.layers.split_at_mut(layer_index);
            let prev_layer = &prev[layer_index - 1];
            let current_layer = &mut current[0];

            for neuron in &mut current_layer.neurons.iter_mut().filter(|n| !n.is_dummy()) {
                let sum: f32 = neuron
                    .input_links
                    .iter()
                    .filter(|link| !link.is_dummy())
                    .map(|link| {
                        return link.weight * prev_layer.get_value(&link.source_id);
                    })
                    .fold(0.0, |acc, e| acc + e);
                neuron.sum_input = sum;
                neuron.output = apply(&neuron.function_name, sum);
            }
        }
    }

    fn backward(&mut self) {
        for layer_index in (1..self.nn.layers_count).rev() {
            let (prev, current) = self.nn.layers.split_at_mut(layer_index);
            let prev_layer = &mut prev[layer_index - 1];
            let current_layer = &mut current[0];

            //распространяем ошибку
            for prev_neuron in &mut prev_layer.neurons.iter_mut().filter(|n| !n.is_dummy()) {
                //суммируем все ошибки, которые внес нейрон(ы) предыдущего слоя
                let mut error_sum = 0.0;
                for neuron in current_layer.neurons.iter().filter(|n| !n.is_dummy()) {
                    //если есть связь между prev_neuron и нейроном текущего слоя
                    for link in neuron
                        .input_links
                        .iter()
                        .filter(|l| l.source_id == prev_neuron.id)
                    {
                        error_sum += link.weight * neuron.error;
                    }
                }
                prev_neuron.error = error_sum;
            }
        }

        //обновляем веса
        for layer_index in 1..self.nn.layers_count {
            let (prev, current) = self.nn.layers.split_at_mut(layer_index);
            let prev_layer = &prev[layer_index - 1];
            let current_layer = &mut current[0];

            for neuron in &mut current_layer.neurons.iter_mut().filter(|n| !n.is_dummy()) {
                for link in neuron.input_links.iter_mut().filter(|l| !l.is_dummy()) {
                    let derive = derivative(&neuron.function_name, neuron.sum_input);
                    let mut delta = neuron.error * derive * prev_layer.get_value(&link.source_id);
                    delta *= self.learning_rate;
                    link.weight += delta;
                }
            }
        }
    }

    fn send_state(&mut self) {
        let execution_objects = ExecutionObjects {
            iteration: self.iteration,
            run_mode: self.run_mode,
        };
        self.tx_adapter.send_timed(&self.nn, &execution_objects);
    }

    fn send_state_immidiately(&mut self) {
        let execution_objects = ExecutionObjects {
            iteration: self.iteration,
            run_mode: self.run_mode,
        };
        self.tx_adapter.send(&self.nn, &execution_objects);
    }

    fn hang_out(&mut self) {
        if self.run_mode == RunMode::Stepping || self.run_mode == RunMode::Pause {
            loop {
                if let Ok(event) = self.rx_events.recv_timeout(STEPPING_DURATION) {
                    match event {
                        Events::PauseRequested => {
                            self.run_mode = RunMode::Pause;
                            self.send_state_immidiately();
                            break;
                        },
                        Events::SteppingRequested => {
                            self.run_mode = RunMode::Stepping;
                            break;
                        }
                        Events::PlayRequested => {
                            self.run_mode = RunMode::Running;
                            break;
                        }
                    }
                } else {
                    if self.run_mode == RunMode::Stepping {
                        break;
                    }
                }
            }
        }
        if self.run_mode == RunMode::Running {
            if let Ok(event) = self.rx_events.try_recv() {
                match event {
                    Events::PauseRequested => {
                        self.run_mode = RunMode::Pause;
                        self.send_state_immidiately();
                    },
                    Events::SteppingRequested => self.run_mode = RunMode::Stepping,
                    Events::PlayRequested => self.run_mode = RunMode::Running,
                }
            }
        }
    }

    fn assert_not_nan(value: f32) {
        if value.is_nan() {
            panic!("Encountered NaN!");
        }
    }

    fn loss(target: f32, value: f32) -> f32 {
        let sign = (target - value).signum();
        let t = target.abs();
        let o = value.abs();
        let diff = (t-o).abs();
        diff/t.max(o)*sign
    }
}

/*
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


    let mut train_items: Vec<TrainItem> = load_train()?;
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
*/

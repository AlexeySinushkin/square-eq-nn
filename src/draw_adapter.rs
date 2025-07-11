use std::cmp::PartialEq;
use std::ops::Sub;
use std::sync::mpsc::Sender;
use std::time::{Duration, Instant};
use crate::draw::objects::{LValue, Model, NValue};
use crate::execution_objects::{ExecutionObjects, RunMode};
use crate::nn_objects::Network;

const FRAME_RATE: Duration = Duration::from_millis(1000 / 20);
pub struct DrawAdapter {
    tx: Sender<Model>,
    last_sent: Instant,
}


impl DrawAdapter {
    pub fn new(tx: Sender<Model>) -> Self {
        Self { tx , last_sent: Instant::now().sub(FRAME_RATE) }
    }
    
    pub fn send_timed(&mut self, nn: &Network, env: &ExecutionObjects) {
        if self.last_sent.elapsed() >= FRAME_RATE {
            self.send(nn, env);
        }
    }
    
    pub fn send(&mut self, nn: &Network, env: &ExecutionObjects) {
        let mut neuron_values: Vec<NValue> = vec![];
        let mut link_values: Vec<LValue> = vec![];
        for layer in nn.layers.iter() {
            for n in layer.neurons.iter().filter(|n| !n.is_dummy()) {
                neuron_values.push(NValue {
                    id: n.id.clone(),
                    input: n.sum_input,
                    value: n.output,
                    error: n.error,
                });
                for l in n.input_links.iter().filter(|l| !l.is_dummy()) {
                    link_values.push(LValue {
                        id: generate_id(&l.source_id, &n.id),
                        value: l.weight,
                    })
                }
            }        
        }

        self.tx.send(Model { neuron_values, link_values, 
            iterations: env.iteration,
            button_pause_active : env.run_mode==RunMode::Pause,
            button_stepping_active: env.run_mode==RunMode::Stepping,
            button_play_active: env.run_mode==RunMode::Running,
        }).unwrap();
        self.last_sent = Instant::now()
    }
}

fn generate_id(from: &String, to: &String) -> String {
    format!("{}->{}", from, to)
}
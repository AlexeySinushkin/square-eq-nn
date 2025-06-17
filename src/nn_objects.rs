use serde::{Deserialize, Serialize};

pub const MAX_LINKS: usize = 4;
pub const MAX_NEURONS_PER_LAYER: usize = 4;
pub const MAX_LAYERS_COUNT: usize = 7;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Link {
    pub source_id: String,
    pub weight: f32
}
impl Link {
    pub fn new(source_id: String, weight: f32) -> Self {
        Link {
            source_id,
            weight
        }
    }
    pub fn new_dummy() -> Self {
        Link {
            source_id: "".to_string(),
            weight: 0.0,
        }
    }
    pub fn is_dummy(&self) -> bool {
        self.source_id.is_empty()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActivationFunction {
    None,
    Sigmoid,
    Square,
    Sqrt,
    Linear,
    Relu
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Neuron {
    pub id: String,
    pub output: f32,
    pub sum_input: f32,
    pub error: f32,
    pub function_name: ActivationFunction,
    pub input_links: [Link; MAX_LINKS],
}

impl Neuron {
    pub fn new_input(id: String) -> Self {
        Neuron {
            id,
            output: 0.0,
            sum_input: 0.0,
            error: 1.0,
            function_name: ActivationFunction::None,
            input_links: std::array::from_fn(|_| Link::new_dummy()),
        }
    }
    pub fn new_middle(id: String, value: f32, function: ActivationFunction, link: [Link; MAX_LINKS]) -> Self {
        Neuron {
            id,
            output: value,
            sum_input: 0.0,
            error: 1.0,
            function_name: function,
            input_links: link,
        }
    }
    pub fn new_dummy() -> Self {
        Neuron {
            id: "".to_string(), // empty string here
            output: 0.0,
            sum_input: 0.0,
            error: 1.0,
            function_name: ActivationFunction::None,
            input_links: std::array::from_fn(|_| Link::new_dummy()),
        }
    }
    pub fn is_dummy(&self) -> bool {
        self.id.is_empty()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Layer {
    pub neurons: [Neuron; MAX_NEURONS_PER_LAYER],
}

impl Layer {
    pub fn new_dummy() -> Self {
        Layer {
            neurons: std::array::from_fn(|_| Neuron::new_dummy()),
        }
    }
    pub fn get_value(&self, neuron_id: &String) -> f32 {
        self.neurons.iter()
            .find(|n| n.id.eq(neuron_id)).unwrap().output
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Network {
    pub layers: [Layer; MAX_LAYERS_COUNT],
    pub layers_count: usize,
}

impl Network {
    pub fn last(&self) -> &Layer {
        &self.layers[self.layers_count-1]
    }
}
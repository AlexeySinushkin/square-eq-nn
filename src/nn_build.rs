use crate::Network;
use crate::nn_objects::{ActivationFunction, Layer, Link, Neuron};

pub fn build_nn() -> Network {
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
        ActivationFunction::Sigmoid,
        [a_m1, b_m1, c_m1, Link::new_dummy()],
    );

    let a_m2 = Link::new("a".to_string(), 0.1);
    let b_m2 = Link::new("b".to_string(), 0.1);
    let c_m2 = Link::new("c".to_string(), 0.1);
    let m2 = Neuron::new_middle(
        "m2".to_string(),
        0.24,
        ActivationFunction::Sigmoid,
        [a_m2, b_m2, c_m2, Link::new_dummy()],
    );

    let a_m3 = Link::new("a".to_string(), 0.1);
    let b_m3 = Link::new("b".to_string(), 0.1);
    let c_m3 = Link::new("c".to_string(), 0.1);
    let m3 = Neuron::new_middle(
        "m3".to_string(),
        0.24,
        ActivationFunction::Sigmoid,
        [a_m3, b_m3, c_m3, Link::new_dummy()],
    );

    let a_m4 = Link::new("a".to_string(), 0.1);
    let b_m4 = Link::new("b".to_string(), 0.1);
    let c_m4 = Link::new("c".to_string(), 0.1);
    let m4 = Neuron::new_middle(
        "m4".to_string(),
        0.24,
        ActivationFunction::Sigmoid,
        [a_m4, b_m4, c_m4, Link::new_dummy()],
    );

    let layer_m = Layer {
        neurons: [m1, m2, m3, m4],
    };

    let m1_n1 = Link::new("m1".to_string(), 0.1);
    let m2_n1 = Link::new("m2".to_string(), 0.1);
    let m3_n1 = Link::new("m3".to_string(), 0.1);
    let m4_n1 = Link::new("m4".to_string(), 0.1);
    let n1 = Neuron::new_middle(
        "n1".to_string(),
        0.23,
        ActivationFunction::Sigmoid,
        [m1_n1, m2_n1, m3_n1, m4_n1],
    );

    let m1_n2 = Link::new("m1".to_string(), 0.1);
    let m2_n2 = Link::new("m2".to_string(), 0.1);
    let m3_n2 = Link::new("m3".to_string(), 0.1);
    let m4_n2 = Link::new("m4".to_string(), 0.1);
    let n2 = Neuron::new_middle(
        "n2".to_string(),
        0.23,
        ActivationFunction::Sigmoid,
        [m1_n2, m2_n2, m3_n2, m4_n2],
    );

    let m1_n3 = Link::new("m1".to_string(), 0.1);
    let m2_n3 = Link::new("m2".to_string(), 0.1);
    let m3_n3 = Link::new("m3".to_string(), 0.1);
    let m4_n3 = Link::new("m4".to_string(), 0.1);
    let n3 = Neuron::new_middle(
        "n3".to_string(),
        0.23,
        ActivationFunction::Sigmoid,
        [m1_n3, m2_n3, m3_n3, m4_n3],
    );

    let layer_n = Layer {
        neurons: [n1, n2, n3, Neuron::new_dummy()],
    };

    let n1_x1 = Link::new("n1".to_string(), 0.1);
    let n2_x1 = Link::new("n2".to_string(), 0.1);
    let n3_x1 = Link::new("n3".to_string(), 0.1);
    let x1 = Neuron::new_middle(
        "x1".to_string(),
        0.23,
        ActivationFunction::Sigmoid,
        [n1_x1, n2_x1, n3_x1, Link::new_dummy()],
    );

    let n1_x2 = Link::new("n1".to_string(), 0.1);
    let n2_x2 = Link::new("n2".to_string(), 0.1);
    let n3_x2 = Link::new("n3".to_string(), 0.1);
    let x2 = Neuron::new_middle(
        "x2".to_string(),
        0.23,
        ActivationFunction::Sigmoid,
        [n1_x2, n2_x2, n3_x2, Link::new_dummy()],
    );

    let output_layer = Layer {
        neurons: [x1, x2, Neuron::new_dummy(), Neuron::new_dummy()],
    };

    Network {
        layers: [input_layer, layer_m, layer_n, output_layer, Layer::new_dummy(), Layer::new_dummy(), Layer::new_dummy()],
        layers_count: 4,
    }
}

pub fn build_nn1() -> Network {
    let k = Neuron::new_input("k".to_string());
    let x = Neuron::new_input("x".to_string());
    let b = Neuron::new_input("b".to_string());
    let input_layer = Layer {
        neurons: [k, x, b, Neuron::new_dummy()],
    };

    let weight = 0.5;
    let l1 = Link::new("k".to_string(), weight);
    let l2 = Link::new("x".to_string(), weight);
    let l3 = Link::new("b".to_string(), weight);
    let m1 = Neuron::new_middle(
        "m1".to_string(),
        0.23,
        ActivationFunction::Linear,
        [l1, l2, l3, Link::new_dummy()],
    );

    let l1 = Link::new("k".to_string(), weight);
    let l2 = Link::new("x".to_string(), weight);
    let l3 = Link::new("b".to_string(), weight);
    let m2 = Neuron::new_middle(
        "m2".to_string(),
        0.24,
        ActivationFunction::Square,
        [l1, l2, l3, Link::new_dummy()],
    );

    let l1 = Link::new("k".to_string(), weight);
    let l2 = Link::new("x".to_string(), weight);
    let l3 = Link::new("b".to_string(), weight);
    let m3 = Neuron::new_middle(
        "m3".to_string(),
        0.24,
        ActivationFunction::Linear,
        [l1, l2, l3, Link::new_dummy()],
    );

    let l1 = Link::new("k".to_string(), weight);
    let l2 = Link::new("x".to_string(), weight);
    let l3 = Link::new("b".to_string(), weight);
    let m4 = Neuron::new_middle(
        "m4".to_string(),
        0.24,
        ActivationFunction::Relu,
        [l1, l2, l3, Link::new_dummy()],
    );



    let layer_m = Layer {
        neurons: [m1, m2, m3, m4],
    };


    let l1 = Link::new("m1".to_string(), weight);
    let l2 = Link::new("m2".to_string(), weight);
    let l3 = Link::new("m3".to_string(), weight);
    let l4 = Link::new("m4".to_string(), weight);
    let y = Neuron::new_middle(
        "y".to_string(),
        0.23,
        ActivationFunction::Linear,
        [l1, l2, l3, l4],
    );


    let output_layer = Layer {
        neurons: [y, Neuron::new_dummy(), Neuron::new_dummy(), Neuron::new_dummy()],
    };

    Network {
        layers: [input_layer, layer_m, output_layer, Layer::new_dummy(), Layer::new_dummy(), Layer::new_dummy(), Layer::new_dummy()],
        layers_count: 3,
    }
}

#[cfg(test)]
mod tests {
    use crate::nn_build::build_nn;
    use std::fs;

    #[test]
    #[ignore]
    fn store_nn() {
        let nn = build_nn();
        let json = serde_json::to_string_pretty(&nn).unwrap();
        fs::write("nn1.json", json).unwrap();
    }
}

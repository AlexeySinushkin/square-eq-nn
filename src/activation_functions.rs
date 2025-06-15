use crate::nn_objects::ActivationFunction;
use crate::nn_objects::ActivationFunction::*;

pub fn apply(function: &ActivationFunction, x: f32) -> f32 {
    match function {
        Sigmoid => sigmoid(x),
        Square => square(x),
        Sqrt => sqrt(x),
        _ => x,
    }
}

pub fn derivative(function: &ActivationFunction, x: f32) -> f32 {
    match function {
        Sigmoid => sigmoid_derivative(x),
        Square => square_derivative(x),
        Sqrt => sqrt_derivative(x),
        _ => x,
    }
}


fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
fn sigmoid_derivative(x: f32) -> f32 {
    let s = sigmoid(x);
    s * (1.0 - s)
}

fn square(x: f32) -> f32 {
    x * x
}

fn square_derivative(x: f32) -> f32 {
    2.0 * x
}

fn sqrt(x: f32) -> f32 {
    x.sqrt()
}

fn sqrt_derivative(x: f32) -> f32 {
    if x <= 0.0 {
        0.0 // or f32::NAN or panic, depending on desired behavior
    } else {
        0.5 / x.sqrt()
    }
}
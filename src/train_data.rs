use rand::seq::SliceRandom;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::fs;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainItem {
    pub a: f32,
    pub b: f32,
    pub c: f32,
    pub x1: f32,
    pub x2: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainItemCommon {
    pub input_1: f32,
    pub input_2: f32,
    pub input_3: f32,
    pub input_4: f32,
    pub output_1: f32,
    pub output_2: f32,
    pub output_3: f32,
    pub output_4: f32,
}

pub fn load_kx_b() -> Vec<TrainItemCommon>{
    let mut rng = rand::rng();
    let mut result = vec![];
    
    for _i in 0..100 {
        let k: f32 = rng.random_range(-10..10) as f32;
        let x: f32 = rng.random_range(-10..10) as f32;
        let b: f32 = rng.random_range(-10..10) as f32;
        let y = k * x + b;       
        result.push(TrainItemCommon{
            input_1: k,
            input_2: x,
            input_3: b,
            input_4: 0.0,
            output_1: y,
            output_2:0.0,
            output_3:0.0,
            output_4:0.0
        })
    }
    result    
}


pub fn load_train() -> Result<Vec<TrainItem>, Box<dyn std::error::Error>> {
    let train_file_content = fs::read_to_string("./train.json")?;
    let train_items: Vec<TrainItem> = serde_json::from_str(&train_file_content)?;
    Ok(train_items)
}

pub fn shuffle<T>(train_items: &mut [T]) {
    let mut rng = rand::rng();
    train_items.shuffle(&mut rng);
}

#[cfg(test)]
mod tests {
    use crate::train_data::{load_train, TrainItem};
    use std::fs;
    const EPSILON: f32 = 1e-3;
    
    fn calculate_eq(item: &TrainItem) -> (f32, f32) {
        let d = item.b.powi(2) - 4.0 * item.a * item.c;
        let x1 = (-item.b - d.sqrt()) / 2.0 * item.a;
        let x2 = (-item.b + d.sqrt()) / 2.0 * item.a;
        (x1, x2)
    }
    
    #[test]
    #[ignore]
    fn recalculate_train_set() {
        let mut train_data: Vec<TrainItem> = load_train().unwrap();
        for item in train_data.iter_mut() {
            let (x1, x2) = calculate_eq(&item);
            item.x1 = x1;
            item.x2 = x2;
        }
        let json = serde_json::to_string_pretty(&train_data).unwrap();
        fs::write("train-recalculated.json", json).unwrap();
    }

    #[test]
    fn validate_train_data() {
        for item in load_train().unwrap() {
            let (x1, x2) = calculate_eq(&item);
            println!("x1 = {x1}, x2 = {x2}, {:?}", item);            
            assert!(
                (x1 - item.x1).abs() < EPSILON,
                "x1 mismatch: expected {}, got {}",
                x1,
                item.x1
            );
            assert!(
                (x2 - item.x2).abs() < EPSILON,
                "x2 mismatch: expected {}, got {} ",
                x2,
                item.x2
            );
        }
    }
}

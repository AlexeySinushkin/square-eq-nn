use rand::seq::SliceRandom;
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
    use std::fs;
    use crate::train_data::{TrainItem, load_train};
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

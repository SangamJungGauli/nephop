use nephop::Direction;
use nephop::random::random_search;
use nephop::space::{Kshetra, Params};

fn main() {
    // A pretend model: the loss is lowest at lr = 0.01 and batch = 64.
    let loss = |p: &Params| {
        let lr = p.float("lr");
        let batch = p.int("batch") as f64;
        (lr.log10() + 2.0).powi(2) + ((batch - 64.0) / 64.0).powi(2)
    };

    let kshetra = Kshetra::new()
        .log_float("lr", 1e-5, 1e-1)
        .int("batch", 8, 256);

    let yatra = random_search(&kshetra, loss, 500, 42, Direction::Minimize);

    if let Some(lakshya) = yatra.lakshya() {
        println!("Yatra complete: {} paila", yatra.paila.len());
        println!("Lakshya:");
        println!("  lr    = {:.5}", lakshya.params.float("lr"));
        println!("  batch = {}", lakshya.params.int("batch"));
        println!("  loss  = {:.4}", lakshya.y);
    }
}

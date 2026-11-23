use nephop::Disha;
use nephop::random::{Yatra, grid_search, random_search};
use nephop::space::{Kshetra, Params};
use nephop::tpe::TpeSampler;

fn show(name: &str, yatra: &Yatra) {
    println!("{name} ({} paila):", yatra.paila.len());
    if let Some(lakshya) = yatra.lakshya() {
        println!("  lr    = {:.5}", lakshya.params.float("lr"));
        println!("  batch = {}", lakshya.params.int("batch"));
        println!("  optim = {}", lakshya.params.choice("optimizer"));
        println!("  loss  = {:.4}", lakshya.y);
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // A pretend model: the loss is lowest at lr = 0.01 and batch = 64.
    let loss = |p: &Params| {
        let lr = p.float("lr");
        let batch = p.int("batch") as f64;
        (lr.log10() + 2.0).powi(2) + ((batch - 64.0) / 64.0).powi(2)
    };

    let kshetra = Kshetra::new()
        .log_float("lr", 1e-5, 1e-1)
        .int("batch", 8, 256)
        .choice("optimizer", &["adam", "sgd"]);

    // Grid search: 5 values for each number, every choice, every combination.
    let points = 5;
    let grid = grid_search(&kshetra, loss, points, Disha::Fedi);

    // Random search with the same number of paila, for a fair comparison.
    let random = random_search(&kshetra, loss, grid.paila.len(), 42, Disha::Fedi);

    // TPE learns from earlier paila, again with the same budget.
    let mut sampler = TpeSampler::new(42, Disha::Fedi);
    let mut tpe = Yatra::new(Disha::Fedi);
    for _ in 0..grid.paila.len() {
        let params = tpe.ask(&kshetra, &mut sampler);
        let y = loss(&params);
        tpe.tell(params, y);
    }

    show("Grid search", &grid);
    show("Random search", &random);
    show("TPE", &tpe);

    random.save("yatra.json")?;
    println!("Saved the random search to yatra.json");

    // Prove it: read the file back and check the Lakshya matches.
    let loaded = Yatra::load("yatra.json")?;
    assert_eq!(loaded.lakshya(), random.lakshya());
    Ok(())
}

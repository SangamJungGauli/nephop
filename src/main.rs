use nephop::random::random_search;
use nephop::{Direction, grid_search};

fn main() {
    let loss = |x: f64| (x - 3.0).powi(2) + 2.0;

    let grid = grid_search(loss, -10.0, 10.0, 2000);
    println!("Grid   : x = {:.3}, y = {:.3}", grid.x, grid.y);

    let yatra = random_search(loss, -10.0, 10.0, 1000, 42, Direction::Minimize);
    if let Some(lakshya) = yatra.lakshya() {
        println!(
            "Random (minimize): x = {:.3}, y = {:.3}",
            lakshya.x, lakshya.y
        );
    }

    let accuracy = |x: f64| -(x - 3.0).powi(2) + 2.0;
    let yatra = random_search(accuracy, -10.0, 10.0, 1000, 42, Direction::Maximize);
    if let Some(lakshya) = yatra.lakshya() {
        println!(
            "Random (maximize): x = {:.3}, y = {:.3}",
            lakshya.x, lakshya.y
        );
    }
}

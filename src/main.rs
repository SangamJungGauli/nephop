use nephop::grid_search;
use nephop::random::random_search;

fn main() {
    let f = |x: f64| (x - 3.0).powi(2) + 2.0;

    let grid = grid_search(f, -10.0, 10.0, 2000);
    println!("Grid   : x = {:.3}, y = {:.3}", grid.x, grid.y);

    let yatra = random_search(f, -10.0, 10.0, 1000, 42);
    if let Some(lakshya) = yatra.lakshya() {
        println!("Random : x = {:.3}, y = {:.3}", lakshya.x, lakshya.y);
    }
}

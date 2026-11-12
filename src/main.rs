use nephop::grid_search;

fn main() {
    let best = grid_search(|x| (x - 3.0).powi(2) + 2.0, -10.0, 10.0, 2000);
    println!("best x = {:.3}, y = {:.3}", best.x, best.y);
}

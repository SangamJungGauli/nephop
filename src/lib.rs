/// The best point found by a search
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Best {
    pub x: f64,
    pub y: f64,
}

/// Try evenly spaced x values in [low, high] and return the x
/// with the smallest objective(x)
pub fn grid_search<F>(objective: F, low: f64, high: f64, steps: usize) -> Best
where
    F: Fn(f64) -> f64,
{
    let mut best = Best {
        x: low,
        y: objective(low),
    };
    for i in 0..=steps {
        let x = low + ((high - low) * (i as f64) / (steps as f64));
        let y = objective(x);
        if y < best.y {
            best = Best { x, y };
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_minimum_of_quadratic() {
        let best = grid_search(|x| (x - 3.0).powi(2) + 2.0, -10.0, 10.0, 2000);
        assert!((best.x - 3.0).abs() < 0.01);
    }

    #[test]
    fn linear_minimum_is_at_the_edge() {
        let best = grid_search(|x| 2.0 * x + 1.0, -10.0, 10.0, 100);
        assert_eq!(best.x, -10.0);
    }
}

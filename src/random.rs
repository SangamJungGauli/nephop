use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

/// A Paila (पाइला, "step"): one attempt, the x we tried and the y it gave.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Paila {
    pub x: f64,
    pub y: f64,
}

/// A Yatra (यात्रा, "journey"): one complete search, with every Paila in order.
#[derive(Debug, Default)]
pub struct Yatra {
    pub paila: Vec<Paila>,
}

impl Yatra {
    pub fn new() -> Self {
        Self::default()
    }

    /// The Lakshya (लक्ष्य, "goal"): the best Paila found.
    /// For now "best" means the smallest y (the valley); a
    /// minimize/maximize direction will be added later.
    pub fn lakshya(&self) -> Option<&Paila> {
        self.paila.iter().min_by(|a, b| a.y.total_cmp(&b.y))
    }
}

/// Take `n_paila` random steps with x in [low, high].
pub fn random_search<F>(objective: F, low: f64, high: f64, n_paila: usize, seed: u64) -> Yatra
where
    F: Fn(f64) -> f64,
{
    let mut rng = StdRng::seed_from_u64(seed);
    let mut yatra = Yatra::new();
    for _ in 0..n_paila {
        let x = rng.random_range(low..=high);
        yatra.paila.push(Paila { x, y: objective(x) });
    }
    yatra
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_near_minimum() {
        let yatra = random_search(|x| (x - 3.0).powi(2) + 2.0, -10.0, 10.0, 1000, 42);
        let best = yatra.lakshya().unwrap();
        assert!((best.x - 3.0).abs() < 0.5);
        assert_eq!(yatra.paila.len(), 1000);
    }

    #[test]
    fn same_seed_same_result() {
        let a = random_search(|x| x * x, -5.0, 5.0, 10, 7);
        let b = random_search(|x| x * x, -5.0, 5.0, 10, 7);
        assert_eq!(a.paila, b.paila);
    }

    #[test]
    fn empty_yatra_has_no_lakshya() {
        assert!(Yatra::new().lakshya().is_none());
    }
}

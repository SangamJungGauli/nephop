use crate::Direction;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

/// A Paila (पाइला, "step"): one attempt, the x we tried and the y it gave.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Paila {
    pub x: f64,
    pub y: f64,
}

/// A Yatra (यात्रा, "journey"): one complete search, with every Paila in order.
#[derive(Debug)]
pub struct Yatra {
    pub direction: Direction,
    pub paila: Vec<Paila>,
}

impl Yatra {
    pub fn new(direction: Direction) -> Self {
        Self {
            direction,
            paila: Vec::new(),
        }
    }

    /// The Lakshya (लक्ष्य, "goal"): the best Paila found, judged by the
    /// Yatra's direction. Returns None if there are no paila yet.
    pub fn lakshya(&self) -> Option<&Paila> {
        match self.direction {
            Direction::Minimize => self.paila.iter().min_by(|a, b| a.y.total_cmp(&b.y)),
            Direction::Maximize => self.paila.iter().max_by(|a, b| a.y.total_cmp(&b.y)),
        }
    }
}

/// Take `n_paila` random steps with x in [low, high].
pub fn random_search<F>(
    objective: F,
    low: f64,
    high: f64,
    n_paila: usize,
    seed: u64,
    direction: Direction,
) -> Yatra
where
    F: Fn(f64) -> f64,
{
    let mut rng = StdRng::seed_from_u64(seed);
    let mut yatra = Yatra::new(direction);
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
    fn minimize_finds_the_valley() {
        let yatra = random_search(
            |x| (x - 3.0).powi(2) + 2.0,
            -10.0,
            10.0,
            1000,
            42,
            Direction::Minimize,
        );
        let best = yatra.lakshya().unwrap();
        assert!((best.x - 3.0).abs() < 0.5);
        assert_eq!(yatra.paila.len(), 1000);
    }

    #[test]
    fn maximize_finds_the_peak() {
        // An upside-down bowl: the highest point is at x = 3, y = 2.
        let yatra = random_search(
            |x| -(x - 3.0).powi(2) + 2.0,
            -10.0,
            10.0,
            1000,
            42,
            Direction::Maximize,
        );
        let best = yatra.lakshya().unwrap();
        assert!((best.x - 3.0).abs() < 0.5);
        assert!(best.y > 1.9);
    }

    #[test]
    fn same_seed_same_result() {
        let a = random_search(|x| x * x, -5.0, 5.0, 10, 7, Direction::Minimize);
        let b = random_search(|x| x * x, -5.0, 5.0, 10, 7, Direction::Minimize);
        assert_eq!(a.paila, b.paila);
    }

    #[test]
    fn empty_yatra_has_no_lakshya() {
        assert!(Yatra::new(Direction::Minimize).lakshya().is_none());
    }
}

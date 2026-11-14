use crate::Direction;
use crate::space::{Kshetra, Params};
use rand::SeedableRng;
use rand::rngs::StdRng;

/// A Paila (पाइला, "step"): one attempt, the settings we tried and the
/// score `y` they gave.
#[derive(Debug, Clone, PartialEq)]
pub struct Paila {
    pub params: Params,
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
    /// Yatra's direction. Returns None if there are no paila
    /// yet or if every score is NaN or infinite (those are skipped)
    pub fn lakshya(&self) -> Option<&Paila> {
        let finite = self.paila.iter().filter(|p| p.y.is_finite());
        match self.direction {
            Direction::Minimize => finite.min_by(|a, b| a.y.total_cmp(&b.y)),
            Direction::Maximize => finite.max_by(|a, b| a.y.total_cmp(&b.y)),
        }
    }
}

/// Take `n_paila` random steps through the `kshetra`, scoring each
/// with `objective`.
pub fn random_search<F>(
    kshetra: &Kshetra,
    objective: F,
    n_paila: usize,
    seed: u64,
    direction: Direction,
) -> Yatra
where
    F: Fn(&Params) -> f64,
{
    let mut rng = StdRng::seed_from_u64(seed);
    let mut yatra = Yatra::new(direction);
    for _ in 0..n_paila {
        let params = kshetra.sample(&mut rng);
        let y = objective(&params);
        yatra.paila.push(Paila { params, y });
    }
    yatra
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bowl(p: &Params) -> f64 {
        (p.float("x") - 3.0).powi(2) + (p.float("y") + 1.0).powi(2) + 2.0
    }

    fn xy_space() -> Kshetra {
        Kshetra::new()
            .float("x", -10.0, 10.0)
            .float("y", -10.0, 10.0)
    }

    #[test]
    fn minimize_finds_the_valley_in_two_dimensions() {
        let yatra = random_search(&xy_space(), bowl, 2000, 42, Direction::Minimize);
        let best = yatra.lakshya().unwrap();
        assert!((best.params.float("x") - 3.0).abs() < 1.0);
        assert!((best.params.float("y") + 1.0).abs() < 1.0);
        assert_eq!(yatra.paila.len(), 2000);
    }

    #[test]
    fn maximize_finds_the_peak() {
        let upside_down = |p: &Params| -bowl(p);
        let yatra = random_search(&xy_space(), upside_down, 2000, 42, Direction::Maximize);
        let best = yatra.lakshya().unwrap();
        assert!((best.params.float("x") - 3.0).abs() < 1.0);
    }

    #[test]
    fn every_value_stays_inside_its_range() {
        let kshetra = Kshetra::new()
            .log_float("lr", 1e-5, 1e-1)
            .int("batch", 8, 256);
        let yatra = random_search(&kshetra, |_| 0.0, 500, 1, Direction::Minimize);
        for paila in &yatra.paila {
            let lr = paila.params.float("lr");
            let batch = paila.params.int("batch");
            assert!((1e-5..=1e-1).contains(&lr));
            assert!((8..=256).contains(&batch));
        }
    }

    #[test]
    fn same_seed_same_result() {
        let a = random_search(&xy_space(), bowl, 10, 7, Direction::Minimize);
        let b = random_search(&xy_space(), bowl, 10, 7, Direction::Minimize);
        assert_eq!(a.paila, b.paila);
    }

    #[test]
    fn nan_is_never_the_lakshya() {
        let mut yatra = Yatra::new(Direction::Maximize);
        for y in [1.0, f64::NAN, 5.0, 2.0] {
            yatra.paila.push(Paila {
                params: Params::default(),
                y,
            });
        }
        assert_eq!(yatra.lakshya().unwrap().y, 5.0)
    }

    #[test]
    fn all_nan_yatra_has_no_laksya() {
        let mut yatra = Yatra::new(Direction::Minimize);
        for y in [f64::NAN, f64::NAN, f64::NAN] {
            yatra.paila.push(Paila {
                params: Params::default(),
                y,
            });
        }
        assert!(yatra.lakshya().is_none());
    }

    #[test]
    fn empty_yatra_has_no_lakshya() {
        assert!(Yatra::new(Direction::Minimize).lakshya().is_none());
    }

    #[test]
    #[should_panic(expected = "was added twice")]
    fn duplicate_names_are_rejected() {
        let _ = Kshetra::new().float("x", 0.0, 1.0).float("x", 0.0, 1.0);
    }
}

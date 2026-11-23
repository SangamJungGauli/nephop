use crate::random::Paila;
use crate::space::{Kshetra, Params};
use rand::SeedableRng;
use rand::rngs::StdRng;

/// A Sampler decides which settings to try next.
pub trait Sampler {
    /// Suggest the next settings to try. `history` holds every Paila
    /// so far, so smarter samplers can learn from it.
    fn sample(&mut self, kshetra: &Kshetra, history: &[Paila]) -> Params;
}

/// Picks every value at random. The same seed gives the same picks.
pub struct RandomSampler {
    rng: StdRng,
}

impl RandomSampler {
    pub fn new(seed: u64) -> Self {
        Self {
            rng: StdRng::seed_from_u64(seed),
        }
    }
}

impl Sampler for RandomSampler {
    fn sample(&mut self, kshetra: &Kshetra, _history: &[Paila]) -> Params {
        kshetra.sample(&mut self.rng)
    }
}

/// Walks through every combination on a grid, one per call. Floats
/// get `points` values each; ints and categoricals use all their values.
/// After the last combination it starts over from the first.
pub struct GridSampler {
    points: usize,
    next: usize,
}

impl GridSampler {
    /// `points` is how many values to try per float parameter (at least 2).
    pub fn new(points: usize) -> Self {
        assert!(points >= 2, "a grid needs at least 2 points per float");
        Self { points, next: 0 }
    }
}

impl Sampler for GridSampler {
    fn sample(&mut self, kshetra: &Kshetra, _history: &[Paila]) -> Params {
        let size = kshetra.grid_size(self.points);
        let params = kshetra.grid_point(self.points, self.next % size);
        self.next += 1;
        params
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_same_samples() {
        let kshetra = Kshetra::new().float("X", 0.0, 1.0);
        let mut a = RandomSampler::new(7);
        let mut b = RandomSampler::new(7);
        for _ in 0..10 {
            assert_eq!(a.sample(&kshetra, &[]), b.sample(&kshetra, &[]));
        }
    }

    #[test]
    fn grid_visits_every_combination_once() {
        let kshetra = Kshetra::new()
            .float("x", 0.0, 1.0)
            .int("n", 1, 2)
            .choice("opt", &["adam", "sgd"]);
        assert_eq!(kshetra.grid_size(3), 3 * 2 * 2);
        let mut sampler = GridSampler::new(3);
        let mut seen = Vec::new();
        for _ in 0..12 {
            let p = sampler.sample(&kshetra, &[]);
            let key = (
                p.float("x").to_bits(),
                p.int("n"),
                p.choice("opt").to_string(),
            );
            assert!(!seen.contains(&key));
            seen.push(key);
        }
        // The 13th call starts over.
        let again = sampler.sample(&kshetra, &[]);
        assert_eq!(again, kshetra.grid_point(3, 0));
    }

    #[test]
    fn grid_floats_include_both_ends() {
        let kshetra = Kshetra::new().log_float("lr", 1e-4, 1e-1);
        let lrs: Vec<f64> = (0..4)
            .map(|i| kshetra.grid_point(4, i).float("lr"))
            .collect();
        assert_eq!(lrs[0], 1e-4);
        assert_eq!(lrs[3], 1e-1);
        assert!((lrs[1] - 1e-3).abs() < 1e-9);
    }

    #[test]
    fn wide_int_range_is_thinned_to_points() {
        let kshetra = Kshetra::new().int("batch", 8, 256);
        assert_eq!(kshetra.grid_size(5), 5);
        assert_eq!(kshetra.grid_point(5, 0).int("batch"), 8);
        assert_eq!(kshetra.grid_point(5, 4).int("batch"), 256);
    }

    #[test]
    fn full_i64_range_does_not_overflow() {
        let kshetra = Kshetra::new().int("x", i64::MIN, i64::MAX);
        assert_eq!(kshetra.grid_size(3), 3);
        assert_eq!(kshetra.grid_point(3, 0).int("x"), i64::MIN);
        assert_eq!(kshetra.grid_point(3, 2).int("x"), i64::MAX);
    }

    #[test]
    fn uncountable_grid_is_none() {
        let mut kshetra = Kshetra::new();
        for i in 0..70 {
            kshetra = kshetra.int(&format!("i{i}"), 0, 9);
        }
        assert_eq!(kshetra.try_grid_size(2), None);
    }

    #[test]
    #[should_panic(expected = "at least 2 points")]
    fn one_point_grid_is_rejected() {
        let _ = GridSampler::new(1);
    }
}

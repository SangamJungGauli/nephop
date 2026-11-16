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
}

use crate::Disha;
use crate::sampler::{RandomSampler, Sampler};
use crate::space::{Kshetra, Params};

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
    pub disha: Disha,
    pub paila: Vec<Paila>,
}

impl Yatra {
    pub fn new(disha: Disha) -> Self {
        Self {
            disha,
            paila: Vec::new(),
        }
    }

    /// The Lakshya (लक्ष्य, "goal"): the best Paila found, judged by the
    /// Yatra's disha. Returns None if there are no paila
    /// yet or if every score is NaN or infinite (those are skipped)
    pub fn lakshya(&self) -> Option<&Paila> {
        let finite = self.paila.iter().filter(|p| p.y.is_finite());
        match self.disha {
            Disha::Fedi => finite.min_by(|a, b| a.y.total_cmp(&b.y)),
            Disha::Chuchuro => finite.max_by(|a, b| a.y.total_cmp(&b.y)),
        }
    }

    /// Sodha (सोध, "ask"): get the next settings to try from the sampler.
    pub fn sodha(&self, kshetra: &Kshetra, sampler: &mut impl Sampler) -> Params {
        sampler.sample(kshetra, &self.paila)
    }

    /// Bhana (भन, "tell"): report the score `y` those settings gave.
    /// This records a new Paila.
    pub fn bhana(&mut self, params: Params, y: f64) {
        self.paila.push(Paila { params, y });
    }

    /// English name for `sodha`.
    pub fn ask(&self, kshetra: &Kshetra, sampler: &mut impl Sampler) -> Params {
        self.sodha(kshetra, sampler)
    }

    /// English name for `bhana`.
    pub fn tell(&mut self, params: Params, y: f64) {
        self.bhana(params, y)
    }
}

/// Take `n_paila` random steps through the `kshetra`, scoring each
/// with `objective`.
pub fn random_search<F>(
    kshetra: &Kshetra,
    objective: F,
    n_paila: usize,
    seed: u64,
    disha: Disha,
) -> Yatra
where
    F: Fn(&Params) -> f64,
{
    let mut sampler = RandomSampler::new(seed);
    let mut yatra = Yatra::new(disha);
    for _ in 0..n_paila {
        let params = yatra.sodha(kshetra, &mut sampler);
        let y = objective(&params);
        yatra.bhana(params, y);
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
        let yatra = random_search(&xy_space(), bowl, 2000, 42, Disha::Fedi);
        let best = yatra.lakshya().unwrap();
        assert!((best.params.float("x") - 3.0).abs() < 1.0);
        assert!((best.params.float("y") + 1.0).abs() < 1.0);
        assert_eq!(yatra.paila.len(), 2000);
    }

    #[test]
    fn maximize_finds_the_peak() {
        let upside_down = |p: &Params| -bowl(p);
        let yatra = random_search(&xy_space(), upside_down, 2000, 42, Disha::Chuchuro);
        let best = yatra.lakshya().unwrap();
        assert!((best.params.float("x") - 3.0).abs() < 1.0);
    }

    #[test]
    fn every_value_stays_inside_its_range() {
        let kshetra = Kshetra::new()
            .log_float("lr", 1e-5, 1e-1)
            .int("batch", 8, 256);
        let yatra = random_search(&kshetra, |_| 0.0, 500, 1, Disha::Fedi);
        for paila in &yatra.paila {
            let lr = paila.params.float("lr");
            let batch = paila.params.int("batch");
            assert!((1e-5..=1e-1).contains(&lr));
            assert!((8..=256).contains(&batch));
        }
    }

    #[test]
    fn same_seed_same_result() {
        let a = random_search(&xy_space(), bowl, 10, 7, Disha::Fedi);
        let b = random_search(&xy_space(), bowl, 10, 7, Disha::Fedi);
        assert_eq!(a.paila, b.paila);
    }

    #[test]
    fn nan_is_never_the_lakshya() {
        let mut yatra = Yatra::new(Disha::Chuchuro);
        for y in [1.0, f64::NAN, 5.0, 2.0] {
            yatra.paila.push(Paila {
                params: Params::default(),
                y,
            });
        }
        assert_eq!(yatra.lakshya().unwrap().y, 5.0)
    }

    #[test]
    fn all_nan_yatra_has_no_lakshya() {
        let mut yatra = Yatra::new(Disha::Fedi);
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
        assert!(Yatra::new(Disha::Fedi).lakshya().is_none());
    }

    #[test]
    #[should_panic(expected = "was added twice")]
    fn duplicate_names_are_rejected() {
        let _ = Kshetra::new().float("x", 0.0, 1.0).float("x", 0.0, 1.0);
    }

    #[test]
    fn choice_only_returns_listed_options() {
        let options = ["adam", "sgd", "rmsprop"];
        let kshetra = Kshetra::new().choice("optimizer", &options);
        let yatra = random_search(&kshetra, |_| 0.0, 200, 1, Disha::Chuchuro);
        for paila in &yatra.paila {
            assert!(options.contains(&paila.params.choice("optimizer")));
        }
    }

    #[test]
    #[should_panic(expected = "need at least one choice")]
    fn empty_choice_list_is_rejected() {
        let _ = Kshetra::new().choice("x", &[]);
    }

    #[test]
    fn sodha_bhana_matches_random_search() {
        let kshetra = xy_space();
        let mut sampler = RandomSampler::new(7);
        let mut yatra = Yatra::new(Disha::Fedi);
        for _ in 0..10 {
            let params = yatra.sodha(&kshetra, &mut sampler);
            let y = bowl(&params);
            yatra.bhana(params, y);
        }
        let expected = random_search(&kshetra, bowl, 10, 7, Disha::Fedi);
        assert_eq!(yatra.paila, expected.paila);
    }

    #[test]
    fn ask_and_tell_are_the_english_names() {
        let kshetra = xy_space();
        let mut sampler = RandomSampler::new(1);
        let mut yatra = Yatra::new(Disha::Fedi);
        let params = yatra.ask(&kshetra, &mut sampler);
        yatra.tell(params, 1.5);
        assert_eq!(yatra.paila.len(), 1);
        assert_eq!(yatra.paila[0].y, 1.5);
    }
}

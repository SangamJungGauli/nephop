use crate::Disha;
use crate::random::Paila;
use crate::sampler::Sampler;
use crate::space::{Kshetra, Param, Params, Value};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use std::f64::consts::PI;

/// Share of the Paila (best first) treated as "good".
const GOOD_SHARE: f64 = 0.25;

/// Chance that a categorical kernel keeps the value of its Paila.
const CHOICE_STAY: f64 = 0.8;

/// A sampler that learns from the history, in the style of TPE (Tree-
/// structured Parzen Estimator). After a few random Paila it splits the
/// history into good and bad, models where each group lies, and suggests
/// settings that look likely to be good and unlikely to be bad.
///
/// It models all parameters together, so it can learn that one choice
/// only works with certain numbers (for example Adam with a small
/// learning rate).
///
/// It needs to know the `Disha` of the Yatra it feeds, so pass the same
/// one to both. The same seed gives the same suggestions.
pub struct TpeSampler {
    rng: StdRng,
    disha: Disha,
    n_startup: usize,
    n_candidates: usize,
}

/// How one parameter is laid out for the model.
enum Dim {
    /// A number, on a log scale if `log`. `lo`..`hi` are in model units.
    Number {
        lo: f64,
        hi: f64,
        log: bool,
        is_int: bool,
    },
    /// One of `k` options, stored as its position.
    Choice { k: usize },
}

impl Dim {
    fn of(param: &Param) -> Dim {
        match param {
            Param::Float { low, high, log } => {
                let (lo, hi) = if *log {
                    (low.ln(), high.ln())
                } else {
                    (*low, *high)
                };
                Dim::Number {
                    lo,
                    hi,
                    log: *log,
                    is_int: false,
                }
            }
            Param::Int { low, high } => Dim::Number {
                lo: *low as f64,
                hi: *high as f64,
                log: false,
                is_int: true,
            },
            Param::Categorical { choices } => Dim::Choice { k: choices.len() },
        }
    }

    /// Width used where a range of zero would break the maths.
    fn span(lo: f64, hi: f64) -> f64 {
        if hi > lo { hi - lo } else { 1.0 }
    }

    /// A value in model units, or None if it does not fit this parameter.
    fn encode(&self, param: &Param, value: &Value) -> Option<f64> {
        match (self, param, value) {
            (Dim::Number { log, .. }, _, Value::Float(v)) => {
                let x = if *log { v.ln() } else { *v };
                x.is_finite().then_some(x)
            }
            (Dim::Number { log: false, .. }, _, Value::Int(v)) => Some(*v as f64),
            (Dim::Choice { .. }, Param::Categorical { choices }, Value::Str(s)) => {
                choices.iter().position(|c| c == s).map(|i| i as f64)
            }
            _ => None,
        }
    }

    fn decode(&self, param: &Param, x: f64) -> Value {
        match (self, param) {
            (Dim::Number { is_int: true, .. }, Param::Int { low, high }) => {
                Value::Int((x.round() as i64).clamp(*low, *high))
            }
            (Dim::Number { log: true, .. }, Param::Float { low, high, .. }) => {
                Value::Float(x.exp().clamp(*low, *high))
            }
            (Dim::Number { .. }, Param::Float { low, high, .. }) => {
                Value::Float(x.clamp(*low, *high))
            }
            (Dim::Choice { .. }, Param::Categorical { choices }) => {
                Value::Str(choices[x as usize].clone())
            }
            _ => unreachable!("dimension built from this parameter"),
        }
    }
}

/// A group of Paila (as points in model units) with kernel widths.
struct Model {
    points: Vec<Vec<f64>>,
    /// Width of the Gaussian for each dimension (unused for choices).
    sigmas: Vec<f64>,
}

impl Model {
    fn new(points: Vec<Vec<f64>>, dims: &[Dim]) -> Model {
        // Wider kernels when there are few points or many dimensions.
        let shrink = (points.len() as f64 + 1.0).powf(-1.0 / (dims.len() as f64 + 4.0));
        let sigmas = dims
            .iter()
            .map(|d| match d {
                Dim::Number { lo, hi, .. } => {
                    let span = Dim::span(*lo, *hi);
                    (span * 0.5 * shrink).max(span * 0.01)
                }
                Dim::Choice { .. } => 0.0,
            })
            .collect();
        Model { points, sigmas }
    }

    /// Log of how likely `x` is under the model: an average over one
    /// kernel per Paila plus one wide kernel that rules nothing out.
    fn log_density(&self, dims: &[Dim], x: &[f64]) -> f64 {
        let mut terms: Vec<f64> = self
            .points
            .iter()
            .map(|point| {
                dims.iter()
                    .enumerate()
                    .map(|(j, d)| match d {
                        Dim::Number { .. } => log_normal(x[j], point[j], self.sigmas[j]),
                        Dim::Choice { k } => log_choice(*k, x[j] == point[j]),
                    })
                    .sum::<f64>()
            })
            .collect();
        terms.push(
            dims.iter()
                .enumerate()
                .map(|(j, d)| match d {
                    Dim::Number { lo, hi, .. } => {
                        log_normal(x[j], lo + (hi - lo) / 2.0, Dim::span(*lo, *hi))
                    }
                    Dim::Choice { k } => -(*k as f64).ln(),
                })
                .sum(),
        );
        let top = terms.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let sum: f64 = terms.iter().map(|t| (t - top).exp()).sum();
        top + sum.ln() - (terms.len() as f64).ln()
    }
}

fn log_normal(x: f64, mu: f64, sigma: f64) -> f64 {
    let z = (x - mu) / sigma;
    -0.5 * z * z - sigma.ln() - 0.5 * (2.0 * PI).ln()
}

/// Log chance of a choice under a kernel centred on one option.
fn log_choice(k: usize, same: bool) -> f64 {
    if k == 1 {
        0.0
    } else if same {
        CHOICE_STAY.ln()
    } else {
        ((1.0 - CHOICE_STAY) / (k as f64 - 1.0)).ln()
    }
}

impl TpeSampler {
    /// Random for the first 10 Paila, then learning, trying 24 candidates.
    pub fn new(seed: u64, disha: Disha) -> Self {
        Self {
            rng: StdRng::seed_from_u64(seed),
            disha,
            n_startup: 10,
            n_candidates: 24,
        }
    }

    /// How many random Paila to take before learning starts (at least 2,
    /// so there is something to compare).
    pub fn with_startup(mut self, n_startup: usize) -> Self {
        assert!(n_startup >= 2, "need at least 2 startup paila");
        self.n_startup = n_startup;
        self
    }

    /// How many candidates to weigh per suggestion (at least 1).
    pub fn with_candidates(mut self, n_candidates: usize) -> Self {
        assert!(n_candidates >= 1, "need at least 1 candidate");
        self.n_candidates = n_candidates;
        self
    }

    /// A standard normal draw (Box-Muller).
    fn normal(&mut self) -> f64 {
        let u1: f64 = self.rng.random_range(f64::EPSILON..1.0);
        let u2: f64 = self.rng.random_range(0.0..1.0);
        (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
    }

    /// Draw one candidate from the good model: pick a Paila (or the wide
    /// kernel) and wiggle each of its values.
    fn draw(&mut self, good: &Model, dims: &[Dim]) -> Vec<f64> {
        let pick = self.rng.random_range(0..=good.points.len());
        let centre = good.points.get(pick); // None means the wide kernel
        dims.iter()
            .enumerate()
            .map(|(j, d)| match d {
                Dim::Number { lo, hi, is_int, .. } => {
                    let (mu, sigma) = match centre {
                        Some(point) => (point[j], good.sigmas[j]),
                        None => (lo + (hi - lo) / 2.0, Dim::span(*lo, *hi)),
                    };
                    let mut x = mu + sigma * self.normal();
                    for _ in 0..10 {
                        if (*lo..=*hi).contains(&x) {
                            break;
                        }
                        x = mu + sigma * self.normal();
                    }
                    x = x.clamp(*lo, *hi);
                    if *is_int { x.round() } else { x }
                }
                Dim::Choice { k } => match centre {
                    Some(point) if *k == 1 || self.rng.random_range(0.0..1.0) < CHOICE_STAY => {
                        point[j]
                    }
                    Some(point) => {
                        // Any option except the one the Paila had.
                        let other = self.rng.random_range(0..*k - 1);
                        if other >= point[j] as usize {
                            (other + 1) as f64
                        } else {
                            other as f64
                        }
                    }
                    None => self.rng.random_range(0..*k) as f64,
                },
            })
            .collect()
    }
}

impl Sampler for TpeSampler {
    fn sample(&mut self, kshetra: &Kshetra, history: &[Paila]) -> Params {
        let layout: Vec<(&str, &Param)> = kshetra.iter().collect();
        let dims: Vec<Dim> = layout.iter().map(|(_, param)| Dim::of(param)).collect();

        // Finite scores only, and only Paila that have every parameter.
        let mut usable: Vec<(f64, Vec<f64>)> = history
            .iter()
            .filter(|p| p.y.is_finite())
            .filter_map(|p| {
                let point: Option<Vec<f64>> = layout
                    .iter()
                    .zip(&dims)
                    .map(|((name, param), dim)| dim.encode(param, &p.params.get(name)?))
                    .collect();
                Some((p.y, point?))
            })
            .collect();
        if usable.len() < self.n_startup || layout.is_empty() {
            return kshetra.sample(&mut self.rng);
        }

        // Best first, whichever way "best" points.
        match self.disha {
            Disha::Fedi => usable.sort_by(|a, b| a.0.total_cmp(&b.0)),
            Disha::Chuchuro => usable.sort_by(|a, b| b.0.total_cmp(&a.0)),
        }
        let n_good =
            ((usable.len() as f64 * GOOD_SHARE).ceil() as usize).clamp(1, usable.len() - 1);
        let bad_points = usable.split_off(n_good);
        let good = Model::new(usable.into_iter().map(|(_, p)| p).collect(), &dims);
        let bad = Model::new(bad_points.into_iter().map(|(_, p)| p).collect(), &dims);

        let mut best: Option<(f64, Vec<f64>)> = None;
        for _ in 0..self.n_candidates {
            let candidate = self.draw(&good, &dims);
            let score = good.log_density(&dims, &candidate) - bad.log_density(&dims, &candidate);
            if best.as_ref().is_none_or(|(s, _)| score > *s) {
                best = Some((score, candidate));
            }
        }
        let (_, x) = best.expect("at least one candidate");

        let mut params = Params::default();
        for (((name, param), dim), x) in layout.iter().zip(&dims).zip(x) {
            params.insert(name, dim.decode(param, x));
        }
        params
    }
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

    /// Run `n` ask/tell steps and return the best score.
    fn run(
        kshetra: &Kshetra,
        sampler: &mut impl Sampler,
        objective: impl Fn(&Params) -> f64,
        n: usize,
        disha: Disha,
    ) -> crate::random::Yatra {
        let mut yatra = crate::random::Yatra::new(disha);
        for _ in 0..n {
            let params = yatra.ask(kshetra, sampler);
            let y = objective(&params);
            yatra.tell(params, y);
        }
        yatra
    }

    #[test]
    fn beats_random_search_on_average() {
        let budget = 60;
        let seeds = 0..20u64;
        let mut tpe_total = 0.0;
        let mut random_total = 0.0;
        for seed in seeds.clone() {
            let mut tpe = TpeSampler::new(seed, Disha::Fedi);
            let mut random = crate::sampler::RandomSampler::new(seed);
            tpe_total += run(&xy_space(), &mut tpe, bowl, budget, Disha::Fedi)
                .lakshya()
                .unwrap()
                .y;
            random_total += run(&xy_space(), &mut random, bowl, budget, Disha::Fedi)
                .lakshya()
                .unwrap()
                .y;
        }
        let n = seeds.count() as f64;
        let (tpe_mean, random_mean) = (tpe_total / n, random_total / n);
        assert!(
            tpe_mean < random_mean,
            "tpe {tpe_mean:.3} should beat random {random_mean:.3}"
        );
    }

    #[test]
    fn maximizing_works_too() {
        let mut total_tpe = 0.0;
        let mut total_random = 0.0;
        for seed in 0..20u64 {
            let upside_down = |p: &Params| -bowl(p);
            let mut tpe = TpeSampler::new(seed, Disha::Chuchuro);
            let mut random = crate::sampler::RandomSampler::new(seed);
            total_tpe += run(&xy_space(), &mut tpe, upside_down, 60, Disha::Chuchuro)
                .lakshya()
                .unwrap()
                .y;
            total_random += run(&xy_space(), &mut random, upside_down, 60, Disha::Chuchuro)
                .lakshya()
                .unwrap()
                .y;
        }
        assert!(total_tpe > total_random);
    }

    #[test]
    fn every_suggestion_stays_in_range() {
        let kshetra = Kshetra::new()
            .log_float("lr", 1e-5, 1e-1)
            .float("x", -2.0, 2.0)
            .int("batch", 8, 256)
            .int("fixed", 5, 5)
            .choice("opt", &["adam", "sgd", "rmsprop"]);
        let objective = |p: &Params| p.float("lr").log10().abs() + p.float("x").abs();
        let mut sampler = TpeSampler::new(1, Disha::Fedi);
        let yatra = run(&kshetra, &mut sampler, objective, 200, Disha::Fedi);
        for paila in &yatra.paila {
            assert!((1e-5..=1e-1).contains(&paila.params.float("lr")));
            assert!((-2.0..=2.0).contains(&paila.params.float("x")));
            assert!((8..=256).contains(&paila.params.int("batch")));
            assert_eq!(paila.params.int("fixed"), 5);
            assert!(["adam", "sgd", "rmsprop"].contains(&paila.params.choice("opt")));
        }
    }

    #[test]
    fn learns_which_choice_is_best() {
        let kshetra = Kshetra::new()
            .float("x", 0.0, 1.0)
            .choice("opt", &["adam", "sgd", "rmsprop"]);
        let objective = |p: &Params| {
            if p.choice("opt") == "adam" { 0.0 } else { 1.0 }
        };
        let mut sampler = TpeSampler::new(3, Disha::Fedi);
        let yatra = run(&kshetra, &mut sampler, objective, 80, Disha::Fedi);
        let adam_late = yatra.paila[30..]
            .iter()
            .filter(|p| p.params.choice("opt") == "adam")
            .count();
        assert!(adam_late > 35, "adam picked only {adam_late} of 50 times");
    }

    #[test]
    fn learns_a_choice_that_only_wins_with_the_right_numbers() {
        // Adam is better, but only near lr = 0.01. Judged one parameter at
        // a time, early bad Adam trials make it look worse than SGD.
        let kshetra = Kshetra::new()
            .log_float("lr", 1e-5, 1e-1)
            .int("batch", 8, 256)
            .choice("optimizer", &["adam", "sgd"]);
        let objective = |p: &Params| {
            let penalty = if p.choice("optimizer") == "adam" {
                0.0
            } else {
                0.3
            };
            (p.float("lr").log10() + 2.0).powi(2)
                + ((p.int("batch") as f64 - 64.0) / 64.0).powi(2)
                + penalty
        };
        let mut wins = 0;
        for seed in 0..40u64 {
            let mut tpe = TpeSampler::new(seed, Disha::Fedi);
            let mut random = crate::sampler::RandomSampler::new(seed);
            let a = run(&kshetra, &mut tpe, objective, 150, Disha::Fedi);
            let b = run(&kshetra, &mut random, objective, 150, Disha::Fedi);
            if a.lakshya().unwrap().y < b.lakshya().unwrap().y {
                wins += 1;
            }
        }
        assert!(wins >= 25, "tpe beat random on only {wins} of 40 seeds");
    }

    #[test]
    fn same_seed_same_suggestions() {
        let go = || {
            let mut sampler = TpeSampler::new(9, Disha::Fedi);
            run(&xy_space(), &mut sampler, bowl, 30, Disha::Fedi).paila
        };
        assert_eq!(go(), go());
    }

    #[test]
    fn nan_and_infinite_scores_are_ignored() {
        let mut sampler = TpeSampler::new(2, Disha::Fedi);
        let mut yatra = crate::random::Yatra::new(Disha::Fedi);
        for i in 0..40 {
            let params = yatra.ask(&xy_space(), &mut sampler);
            let y = match i % 3 {
                0 => f64::NAN,
                1 => f64::INFINITY,
                _ => bowl(&params),
            };
            yatra.tell(params, y);
        }
        assert!(yatra.lakshya().is_some());
    }

    #[test]
    fn parameters_missing_from_history_fall_back_to_random() {
        // History made before "z" existed: TPE must still suggest it.
        let old_space = Kshetra::new().float("x", 0.0, 1.0);
        let mut yatra = crate::random::Yatra::new(Disha::Fedi);
        let mut random = crate::sampler::RandomSampler::new(1);
        for _ in 0..15 {
            let params = yatra.ask(&old_space, &mut random);
            yatra.tell(params, 1.0);
        }
        let new_space = Kshetra::new().float("x", 0.0, 1.0).float("z", 0.0, 1.0);
        let mut tpe = TpeSampler::new(1, Disha::Fedi);
        let params = yatra.ask(&new_space, &mut tpe);
        assert!((0.0..=1.0).contains(&params.float("z")));
    }

    #[test]
    #[should_panic(expected = "at least 2 startup")]
    fn tiny_startup_is_rejected() {
        let _ = TpeSampler::new(1, Disha::Fedi).with_startup(1);
    }
}

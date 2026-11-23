use rand::RngExt;
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A concrete value chosen for one parameter.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Value {
    Float(f64),
    Int(i64),
    Str(String),
}

/// The allowed range for one parameter.
#[derive(Debug, Clone, PartialEq)]
pub enum Param {
    /// A decimal number between `low` and `high`.
    /// With `log: true` the numbers are spread evenly on a log scale,
    /// which suits things like learning rates (0.00001 ... 0.1).
    Float { low: f64, high: f64, log: bool },
    /// A whole number between `low` and `high` (both included).
    Int { low: i64, high: i64 },
    /// one option out of fixed list (for example, an optimizer name).
    Categorical { choices: Vec<String> },
}

impl Param {
    /// The values a grid visits for this parameter. Floats get `points`
    /// evenly spaced values (on a log scale if `log`), ints get every
    /// whole number (or `points` of them if the range is wider), and
    /// categoricals get every choice.
    fn grid_values(&self, points: usize) -> Vec<Value> {
        match self {
            Param::Float { low, high, log } => (0..points)
                .map(|i| {
                    if i == 0 {
                        return Value::Float(*low);
                    }
                    if i == points - 1 {
                        return Value::Float(*high);
                    }
                    let t = i as f64 / (points - 1) as f64;
                    let v = if *log {
                        (low.ln() + t * (high.ln() - low.ln())).exp()
                    } else {
                        low + t * (high - low)
                    };
                    // Rounding can push a value a hair outside the range.
                    Value::Float(v.clamp(*low, *high))
                })
                .collect(),
            Param::Int { low, high } => {
                let count = (*high as i128 - *low as i128) as u128 + 1;
                if count <= points as u128 {
                    (*low..=*high).map(Value::Int).collect()
                } else {
                    let mut values: Vec<i64> = (0..points)
                        .map(|i| {
                            let t = i as f64 / (points - 1) as f64;
                            (*low as f64 + t * (*high as f64 - *low as f64)).round() as i64
                        })
                        .map(|v| v.clamp(*low, *high))
                        .collect();
                    values.dedup();
                    values.into_iter().map(Value::Int).collect()
                }
            }
            Param::Categorical { choices } => choices.iter().cloned().map(Value::Str).collect(),
        }
    }

    fn sample(&self, rng: &mut StdRng) -> Value {
        match self {
            Param::Float {
                low,
                high,
                log: false,
            } => Value::Float(rng.random_range(*low..=*high)),
            Param::Float {
                low,
                high,
                log: true,
            } => {
                let exponent = rng.random_range(low.ln()..=high.ln());
                // Rounding can push the result a hair outside the range.
                Value::Float(exponent.exp().clamp(*low, *high))
            }
            Param::Int { low, high } => Value::Int(rng.random_range(*low..=*high)),
            Param::Categorical { choices } => {
                let i = rng.random_range(0..choices.len());
                Value::Str(choices[i].clone())
            }
        }
    }
}

/// The values chosen for one Paila, looked up by name.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct Params {
    values: HashMap<String, Value>,
}

impl Params {
    /// Set the value of the parameter called `name`.
    pub fn insert(&mut self, name: &str, value: Value) {
        self.values.insert(name.to_string(), value);
    }

    /// Every parameter name with its value, in no particular order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &Value)> {
        self.values.iter().map(|(k, v)| (k.as_str(), v))
    }

    pub fn get(&self, name: &str) -> Option<Value> {
        self.values.get(name).cloned()
    }

    /// The float parameter called `name`.
    ///
    /// # Panics
    /// If there is no float parameter with that name (usually a typo).
    pub fn float(&self, name: &str) -> f64 {
        match self.get(name) {
            Some(Value::Float(v)) => v,
            _ => panic!("no float parameter named `{name}`"),
        }
    }

    /// The chosen option for the categorical parameter called `name`.
    ///
    /// # Panics
    /// If there is no categorical parameter with that name (usually a typo).
    pub fn choice(&self, name: &str) -> &str {
        match self.values.get(name) {
            Some(Value::Str(s)) => s,
            _ => panic!("no categorical parameter named `{name}`"),
        }
    }

    /// The int parameter called `name`.
    ///
    /// # Panics
    /// If there is no int parameter with that name (usually a typo).
    pub fn int(&self, name: &str) -> i64 {
        match self.get(name) {
            Some(Value::Int(v)) => v,
            _ => panic!("no int parameter named `{name}`"),
        }
    }
}

/// A Kshetra (क्षेत्र, "field"): the search space, meaning every
/// parameter to tune and the range each one may take.
#[derive(Debug, Clone, Default)]
pub struct Kshetra {
    params: Vec<(String, Param)>,
}

impl Kshetra {
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether a parameter called `name` was already added.
    pub fn contains(&self, name: &str) -> bool {
        self.params.iter().any(|(n, _)| n == name)
    }

    fn add(mut self, name: &str, param: Param) -> Self {
        assert!(!self.contains(name), "parameter `{name}` was added twice");
        self.params.push((name.to_string(), param));
        self
    }

    /// Add a decimal parameter, spread evenly between `low` and `high`.
    pub fn float(self, name: &str, low: f64, high: f64) -> Self {
        assert!(low < high, "`{name}`: low must be smaller than high");
        self.add(
            name,
            Param::Float {
                low,
                high,
                log: false,
            },
        )
    }

    /// Add a decimal parameter spread evenly on a log scale.
    /// `low` must be above 0.
    pub fn log_float(self, name: &str, low: f64, high: f64) -> Self {
        assert!(
            0.0 < low && low < high,
            "`{name}`: need 0 < low < high for a log scale"
        );
        self.add(
            name,
            Param::Float {
                low,
                high,
                log: true,
            },
        )
    }

    /// Add a whole-number parameter, `low` and `high` both allowed.
    pub fn int(self, name: &str, low: i64, high: i64) -> Self {
        assert!(low <= high, "`{name}`: low must not be above high");
        self.add(name, Param::Int { low, high })
    }

    /// Add a parameter that takes one option from `choices`.
    pub fn choice(self, name: &str, choices: &[&str]) -> Self {
        assert!(!choices.is_empty(), "`{name}`: need at least one choice");
        let choices = choices.iter().map(|c| c.to_string()).collect();
        self.add(name, Param::Categorical { choices })
    }

    /// How many settings a grid with `points` values per float has, or
    /// None if that number does not fit in a `usize`.
    ///
    /// # Panics
    /// If `points` is below 2.
    pub fn try_grid_size(&self, points: usize) -> Option<usize> {
        assert!(points >= 2, "a grid needs at least 2 points per float");
        self.params
            .iter()
            .map(|(_, p)| p.grid_values(points).len())
            .try_fold(1usize, |total, n| total.checked_mul(n))
    }

    /// How many settings a grid with `points` values per float has.
    ///
    /// # Panics
    /// If `points` is below 2, or the count does not fit in a `usize`.
    pub fn grid_size(&self, points: usize) -> usize {
        self.try_grid_size(points)
            .expect("the grid is too large to count")
    }

    /// The `index`-th setting of the grid (the last parameter changes
    /// fastest). `index` must be below `grid_size(points)`.
    pub fn grid_point(&self, points: usize, index: usize) -> Params {
        assert!(points >= 2, "a grid needs at least 2 points per float");
        let mut rest = index;
        let mut params = Params::default();
        for (name, param) in self.params.iter().rev() {
            let values = param.grid_values(points);
            params
                .values
                .insert(name.clone(), values[rest % values.len()].clone());
            rest /= values.len();
        }
        assert!(rest == 0, "grid index {index} is out of range");
        params
    }

    /// Pick one random value for every parameter.
    pub fn sample(&self, rng: &mut StdRng) -> Params {
        let mut params = Params::default();
        for (name, param) in &self.params {
            params.values.insert(name.clone(), param.sample(rng));
        }
        params
    }
}

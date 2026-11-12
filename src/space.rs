use rand::RngExt;
use rand::rngs::StdRng;
use std::collections::HashMap;

/// A concrete number chosen for one parameter.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Value {
    Float(f64),
    Int(i64),
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
}

impl Param {
    fn sample(&self, rng: &mut StdRng) -> Value {
        match *self {
            Param::Float {
                low,
                high,
                log: false,
            } => Value::Float(rng.random_range(low..=high)),
            Param::Float {
                low,
                high,
                log: true,
            } => {
                let exponent = rng.random_range(low.ln()..=high.ln());
                // Rounding can push the result a hair outside the range.
                Value::Float(exponent.exp().clamp(low, high))
            }
            Param::Int { low, high } => Value::Int(rng.random_range(low..=high)),
        }
    }
}

/// The values chosen for one Paila, looked up by name.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Params {
    values: HashMap<String, Value>,
}

impl Params {
    pub fn get(&self, name: &str) -> Option<Value> {
        self.values.get(name).copied()
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

    fn add(mut self, name: &str, param: Param) -> Self {
        assert!(
            !self.params.iter().any(|(n, _)| n == name),
            "parameter `{name}` was added twice"
        );
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

    /// Pick one random value for every parameter.
    pub fn sample(&self, rng: &mut StdRng) -> Params {
        let mut params = Params::default();
        for (name, param) in &self.params {
            params.values.insert(name.clone(), param.sample(rng));
        }
        params
    }
}

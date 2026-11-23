//! Python bindings for NepHOp.
//!
//! Parameters cross the boundary as plain dicts, for example
//! `{"lr": 0.01, "batch": 64, "optimizer": "adam"}`: floats stay floats,
//! ints stay ints and categorical choices are strings.

use nephop_core::Disha;
use nephop_core::random as core_yatra;
use nephop_core::sampler::{GridSampler as CoreGrid, RandomSampler as CoreRandom, Sampler};
use nephop_core::space::{Kshetra as CoreKshetra, Params, Value};
use pyo3::exceptions::{PyOSError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

fn parse_disha(name: &str) -> PyResult<Disha> {
    match name.to_ascii_lowercase().as_str() {
        "minimize" | "fedi" => Ok(Disha::Fedi),
        "maximize" | "chuchuro" => Ok(Disha::Chuchuro),
        _ => Err(PyValueError::new_err(format!(
            "unknown disha `{name}`: use \"minimize\" or \"maximize\""
        ))),
    }
}

fn check_points(points: usize) -> PyResult<()> {
    if points < 2 {
        return Err(PyValueError::new_err(
            "a grid needs at least 2 points per float",
        ));
    }
    Ok(())
}

fn params_to_dict<'py>(py: Python<'py>, params: &Params) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    for (name, value) in params.iter() {
        match value {
            Value::Float(v) => dict.set_item(name, v)?,
            Value::Int(v) => dict.set_item(name, v)?,
            Value::Str(v) => dict.set_item(name, v)?,
        }
    }
    Ok(dict)
}

fn dict_to_params(dict: &Bound<'_, PyDict>) -> PyResult<Params> {
    let mut params = Params::default();
    for (key, value) in dict.iter() {
        let name: String = key.extract()?;
        // Try int before float: a Python float never extracts as an int.
        let value = if let Ok(v) = value.extract::<i64>() {
            Value::Int(v)
        } else if let Ok(v) = value.extract::<f64>() {
            Value::Float(v)
        } else if let Ok(v) = value.extract::<String>() {
            Value::Str(v)
        } else {
            return Err(PyValueError::new_err(format!(
                "parameter `{name}` must be a float, int or str"
            )));
        };
        params.insert(&name, value);
    }
    Ok(params)
}

/// A Kshetra: the search space.
#[pyclass(name = "Kshetra")]
#[derive(Default)]
struct Kshetra {
    inner: CoreKshetra,
}

impl Kshetra {
    fn check_name(&self, name: &str) -> PyResult<()> {
        if self.inner.contains(name) {
            return Err(PyValueError::new_err(format!(
                "parameter `{name}` was added twice"
            )));
        }
        Ok(())
    }
}

#[pymethods]
impl Kshetra {
    #[new]
    fn new() -> Self {
        Self::default()
    }

    /// Add a decimal parameter, spread evenly between `low` and `high`.
    fn float<'py>(
        mut slf: PyRefMut<'py, Self>,
        name: &str,
        low: f64,
        high: f64,
    ) -> PyResult<PyRefMut<'py, Self>> {
        slf.check_name(name)?;
        if !(low.is_finite() && high.is_finite() && low < high) {
            return Err(PyValueError::new_err(format!(
                "`{name}`: low must be smaller than high (both finite)"
            )));
        }
        let kshetra = std::mem::take(&mut slf.inner);
        slf.inner = kshetra.float(name, low, high);
        Ok(slf)
    }

    /// Add a decimal parameter spread evenly on a log scale.
    fn log_float<'py>(
        mut slf: PyRefMut<'py, Self>,
        name: &str,
        low: f64,
        high: f64,
    ) -> PyResult<PyRefMut<'py, Self>> {
        slf.check_name(name)?;
        if !(0.0 < low && low < high && high.is_finite()) {
            return Err(PyValueError::new_err(format!(
                "`{name}`: need 0 < low < high for a log scale"
            )));
        }
        let kshetra = std::mem::take(&mut slf.inner);
        slf.inner = kshetra.log_float(name, low, high);
        Ok(slf)
    }

    /// Add a whole-number parameter, `low` and `high` both allowed.
    fn int<'py>(
        mut slf: PyRefMut<'py, Self>,
        name: &str,
        low: i64,
        high: i64,
    ) -> PyResult<PyRefMut<'py, Self>> {
        slf.check_name(name)?;
        if low > high {
            return Err(PyValueError::new_err(format!(
                "`{name}`: low must not be above high"
            )));
        }
        let kshetra = std::mem::take(&mut slf.inner);
        slf.inner = kshetra.int(name, low, high);
        Ok(slf)
    }

    /// Add a parameter that takes one option from `choices`.
    fn choice<'py>(
        mut slf: PyRefMut<'py, Self>,
        name: &str,
        choices: Vec<String>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        slf.check_name(name)?;
        if choices.is_empty() {
            return Err(PyValueError::new_err(format!(
                "`{name}`: need at least one choice"
            )));
        }
        let choices: Vec<&str> = choices.iter().map(String::as_str).collect();
        let kshetra = std::mem::take(&mut slf.inner);
        slf.inner = kshetra.choice(name, &choices);
        Ok(slf)
    }

    /// How many settings a grid with `points` values per float has.
    fn grid_size(&self, points: usize) -> PyResult<usize> {
        check_points(points)?;
        Ok(self.inner.grid_size(points))
    }
}

/// Picks every value at random. The same seed gives the same picks.
#[pyclass(name = "RandomSampler")]
struct RandomSampler {
    inner: CoreRandom,
}

#[pymethods]
impl RandomSampler {
    #[new]
    #[pyo3(signature = (seed=0))]
    fn new(seed: u64) -> Self {
        Self {
            inner: CoreRandom::new(seed),
        }
    }
}

/// Walks through every combination on a grid, one per call.
#[pyclass(name = "GridSampler")]
struct GridSampler {
    inner: CoreGrid,
}

#[pymethods]
impl GridSampler {
    #[new]
    fn new(points: usize) -> PyResult<Self> {
        check_points(points)?;
        Ok(Self {
            inner: CoreGrid::new(points),
        })
    }
}

/// Ask whichever sampler Python handed us.
fn sample_from(
    sampler: &Bound<'_, PyAny>,
    kshetra: &CoreKshetra,
    history: &[core_yatra::Paila],
) -> PyResult<Params> {
    if let Ok(mut s) = sampler.extract::<PyRefMut<'_, RandomSampler>>() {
        return Ok(s.inner.sample(kshetra, history));
    }
    if let Ok(mut s) = sampler.extract::<PyRefMut<'_, GridSampler>>() {
        return Ok(s.inner.sample(kshetra, history));
    }
    Err(PyValueError::new_err(
        "sampler must be a RandomSampler or a GridSampler",
    ))
}

/// A Yatra: one complete search, with every Paila in order.
#[pyclass(name = "Yatra")]
struct Yatra {
    inner: core_yatra::Yatra,
}

#[pymethods]
impl Yatra {
    #[new]
    #[pyo3(signature = (disha="minimize"))]
    fn new(disha: &str) -> PyResult<Self> {
        Ok(Self {
            inner: core_yatra::Yatra::new(parse_disha(disha)?),
        })
    }

    /// Get the next settings to try, as a dict.
    fn ask<'py>(
        &self,
        py: Python<'py>,
        kshetra: &Kshetra,
        sampler: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let params = sample_from(sampler, &kshetra.inner, &self.inner.paila)?;
        params_to_dict(py, &params)
    }

    /// Report the score `y` that those settings gave.
    fn tell(&mut self, params: &Bound<'_, PyDict>, y: f64) -> PyResult<()> {
        self.inner.tell(dict_to_params(params)?, y);
        Ok(())
    }

    /// Nepali name for `ask`.
    fn sodha<'py>(
        &self,
        py: Python<'py>,
        kshetra: &Kshetra,
        sampler: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyDict>> {
        self.ask(py, kshetra, sampler)
    }

    /// Nepali name for `tell`.
    fn bhana(&mut self, params: &Bound<'_, PyDict>, y: f64) -> PyResult<()> {
        self.tell(params, y)
    }

    /// The best Paila so far as `(params, y)`, or None if there is none.
    fn lakshya<'py>(&self, py: Python<'py>) -> PyResult<Option<(Bound<'py, PyDict>, f64)>> {
        match self.inner.lakshya() {
            Some(p) => Ok(Some((params_to_dict(py, &p.params)?, p.y))),
            None => Ok(None),
        }
    }

    /// Every Paila so far, as a list of `(params, y)`.
    fn history<'py>(&self, py: Python<'py>) -> PyResult<Vec<(Bound<'py, PyDict>, f64)>> {
        self.inner
            .paila
            .iter()
            .map(|p| Ok((params_to_dict(py, &p.params)?, p.y)))
            .collect()
    }

    fn __len__(&self) -> usize {
        self.inner.paila.len()
    }

    /// Save to a JSON file, replacing the file if it exists.
    fn save(&self, path: &str) -> PyResult<()> {
        self.inner
            .save(path)
            .map_err(|e| PyOSError::new_err(e.to_string()))
    }

    /// Load a Yatra from a JSON file written by `save`.
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        core_yatra::Yatra::load(path)
            .map(|inner| Self { inner })
            .map_err(|e| PyOSError::new_err(e.to_string()))
    }
}

/// Run `n` steps with `sampler`, scoring each with the Python `objective`.
fn run(
    py: Python<'_>,
    objective: &Bound<'_, PyAny>,
    n_paila: usize,
    disha: Disha,
    mut next: impl FnMut(&[core_yatra::Paila]) -> Params,
) -> PyResult<Yatra> {
    let mut yatra = core_yatra::Yatra::new(disha);
    for _ in 0..n_paila {
        let params = next(&yatra.paila);
        let y: f64 = objective
            .call1((params_to_dict(py, &params)?,))?
            .extract()?;
        yatra.tell(params, y);
    }
    Ok(Yatra { inner: yatra })
}

/// Take `n_paila` random steps, calling `objective(params)` for each.
#[pyfunction]
#[pyo3(signature = (kshetra, objective, n_paila, seed=0, disha="minimize"))]
fn random_search(
    py: Python<'_>,
    kshetra: &Kshetra,
    objective: &Bound<'_, PyAny>,
    n_paila: usize,
    seed: u64,
    disha: &str,
) -> PyResult<Yatra> {
    let mut sampler = CoreRandom::new(seed);
    run(py, objective, n_paila, parse_disha(disha)?, |history| {
        sampler.sample(&kshetra.inner, history)
    })
}

/// Try every combination on a grid, calling `objective(params)` for each.
#[pyfunction]
#[pyo3(signature = (kshetra, objective, points, disha="minimize"))]
fn grid_search(
    py: Python<'_>,
    kshetra: &Kshetra,
    objective: &Bound<'_, PyAny>,
    points: usize,
    disha: &str,
) -> PyResult<Yatra> {
    check_points(points)?;
    let mut sampler = CoreGrid::new(points);
    run(
        py,
        objective,
        kshetra.inner.grid_size(points),
        parse_disha(disha)?,
        |history| sampler.sample(&kshetra.inner, history),
    )
}

#[pymodule]
#[pyo3(name = "nephop")]
fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Kshetra>()?;
    m.add_class::<RandomSampler>()?;
    m.add_class::<GridSampler>()?;
    m.add_class::<Yatra>()?;
    m.add_function(wrap_pyfunction!(random_search, m)?)?;
    m.add_function(wrap_pyfunction!(grid_search, m)?)?;
    Ok(())
}

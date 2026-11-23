# NepHOp (नेपहप)

**Nepalese Hyperparameter Optimizer**: a fast optimizer written in Rust,
for Python and PyTorch.

> Status: very early. Not usable yet.

## Try it

    cargo run

Runs a demo that tunes a learning rate, batch size and optimizer on a
pretend loss function with grid search, random search and TPE, and prints the
best settings each one found.

## Use from Python

    cd python
    pip install maturin
    maturin develop

```python
from nephop import Kshetra, Yatra, RandomSampler, random_search

kshetra = (
    Kshetra()
    .log_float("lr", 1e-5, 1e-1)
    .int("batch", 8, 256)
    .choice("optimizer", ["adam", "sgd"])
)

# One call:
yatra = random_search(kshetra, lambda p: train(**p), n_paila=50, seed=42)
params, loss = yatra.lakshya()

# Or ask and tell:
yatra, sampler = Yatra("minimize"), RandomSampler(42)
p = yatra.ask(kshetra, sampler)
yatra.tell(p, train(**p))
```

Parameters are plain dicts. See `python/examples/pytorch_tune.py` for tuning
a small PyTorch model. There is also `grid_search` and
`GridSampler`, `yatra.save(path)` and `Yatra.load(path)`.

## Nepali names

| Name | Nepali | Meaning | In other tools |
|---|---|---|---|
| Yatra | यात्रा | journey | study |
| Paila | पाइला | step | trial |
| Kshetra | क्षेत्र | field | search space |
| Lakshya | लक्ष्य | goal | best trial |
| Disha | दिशा | direction | direction |
| Fedi | फेदी | foot of the hill | minimize |
| Chuchuro | चुचुरो | peak | maximize |
| sodha | सोध | ask | ask |
| bhana | भन | tell | tell |

`Direction`, `Minimize`, `Maximize`, `ask` and `tell` also work.


## Roadmap

- [x] Random search
- [x] Disha (minimize or maximize)
- [x] Search space with several parameters (Kshetra)
- [x] Categorical parameters
- [x] Ask/tell API and Sampler trait
- [x] Save and load a Yatra as JSON
- [x] Grid search over a Kshetra
- [x] TPE sampler (learns from past trials, models parameters together)
- [ ] TPE in the Python bindings
- [ ] SQLite storage
- [x] Python bindings (first slice: space, samplers, ask/tell, save/load)

## License

MIT
# NepHOp (नेपहप)

**Nepalese Hyperparameter Optimizer**: a fast optimizer written in Rust,
for Python and PyTorch.

> Status: very early. Not usable yet.

## Try it

    cargo run

Runs a demo that tunes a learning rate, batch size and optimizer on a
pretend loss function with grid search and random search, and prints the
best settings each one found.

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
- [ ] SQLite storage
- [ ] Python bindings

## License

MIT
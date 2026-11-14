# NepHOp (नेपहप)

**Nepalese Hyperparameter Optimizer**: a fast optimizer written in Rust,
for Python and PyTorch.

> Status: very early. Not usable yet.

## Try it

    cargo run

Runs a demo that tunes a learning rate, batch size and optimizer on a
pretend loss function and prints the best settings found.

## Roadmap

- [x] Random search
- [x] Direction (minimize or maximize)
- [x] Search space with several parameters (Kshetra)
- [x] Categorical parameters
- [ ] Ask/tell API and Sampler trait
- [ ] Grid search over a Kshetra
- [ ] Trial history and SQLite storage
- [ ] Python bindings

## License

MIT
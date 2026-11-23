import pytest

import nephop
from nephop import GridSampler, Kshetra, RandomSampler, Yatra, grid_search, random_search


def space():
    return (
        Kshetra()
        .log_float("lr", 1e-5, 1e-1)
        .int("batch", 8, 256)
        .choice("optimizer", ["adam", "sgd"])
    )


def loss(p):
    return (p["lr"] * 100 - 1) ** 2 + ((p["batch"] - 64) / 64) ** 2


def test_ask_returns_typed_dict_inside_ranges():
    yatra = Yatra()
    sampler = RandomSampler(1)
    for _ in range(50):
        p = yatra.ask(space(), sampler)
        assert isinstance(p["lr"], float) and 1e-5 <= p["lr"] <= 1e-1
        assert isinstance(p["batch"], int) and 8 <= p["batch"] <= 256
        assert p["optimizer"] in ("adam", "sgd")


def test_ask_tell_loop_and_lakshya():
    kshetra, yatra, sampler = space(), Yatra("minimize"), RandomSampler(3)
    assert yatra.lakshya() is None
    for _ in range(100):
        p = yatra.ask(kshetra, sampler)
        yatra.tell(p, loss(p))
    assert len(yatra) == 100
    params, y = yatra.lakshya()
    assert y == min(h[1] for h in yatra.history())
    assert params in [h[0] for h in yatra.history()]


def test_maximize_picks_the_highest():
    yatra = Yatra("maximize")
    for y in (1.0, 5.0, 2.0):
        yatra.tell({"x": y}, y)
    assert yatra.lakshya()[1] == 5.0


def test_nepali_names():
    yatra = Yatra("fedi")
    p = yatra.sodha(space(), RandomSampler(1))
    yatra.bhana(p, 1.0)
    assert len(yatra) == 1


def test_same_seed_same_result():
    a = random_search(space(), loss, 20, seed=7)
    b = random_search(space(), loss, 20, seed=7)
    assert a.history() == b.history()


def test_grid_search_covers_the_whole_grid():
    kshetra = space()
    yatra = grid_search(kshetra, loss, 5)
    assert len(yatra) == kshetra.grid_size(5) == 50
    assert len({(p["lr"], p["batch"], p["optimizer"]) for p, _ in yatra.history()}) == 50
    assert yatra.lakshya()[0]["lr"] == pytest.approx(0.01)


def test_grid_sampler_via_ask():
    kshetra = Kshetra().int("n", 1, 3)
    yatra, sampler = Yatra(), GridSampler(3)
    assert [yatra.ask(kshetra, sampler)["n"] for _ in range(3)] == [1, 2, 3]


def test_objective_errors_propagate():
    def boom(_):
        raise RuntimeError("objective failed")

    with pytest.raises(RuntimeError, match="objective failed"):
        random_search(space(), boom, 3)


def test_save_and_load_round_trip(tmp_path):
    yatra = random_search(space(), loss, 10, seed=2, disha="maximize")
    path = str(tmp_path / "yatra.json")
    yatra.save(path)
    loaded = Yatra.load(path)
    assert loaded.history() == yatra.history()
    assert loaded.lakshya() == yatra.lakshya()


def test_load_missing_file_is_oserror():
    with pytest.raises(OSError):
        Yatra.load("no_such_folder/no_such_file.json")


@pytest.mark.parametrize(
    "build",
    [
        lambda: Kshetra().float("x", 1.0, 1.0),
        lambda: Kshetra().float("x", 0.0, float("inf")),
        lambda: Kshetra().log_float("x", 0.0, 1.0),
        lambda: Kshetra().int("x", 5, 1),
        lambda: Kshetra().choice("x", []),
        lambda: Kshetra().float("x", 0, 1).int("x", 0, 1),
        lambda: Kshetra().grid_size(1),
        lambda: GridSampler(1),
        lambda: Yatra("sideways"),
    ],
)
def test_bad_input_is_value_error_not_a_panic(build):
    with pytest.raises(ValueError):
        build()


def test_tell_rejects_unsupported_value_types():
    with pytest.raises(ValueError):
        Yatra().tell({"x": [1, 2]}, 1.0)


def test_ask_rejects_unknown_sampler():
    with pytest.raises(ValueError):
        Yatra().ask(space(), object())

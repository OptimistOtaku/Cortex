from sietch.common.hlc import HLC, parse


def test_monotonic_under_frozen_clock():
    c = HLC("a", clock=lambda: 1000)
    ts = [c.now() for _ in range(5)]
    assert ts == sorted(ts) and len(set(ts)) == 5


def test_observe_moves_past_remote_even_if_local_clock_is_behind():
    slow = HLC("slow", clock=lambda: 1000)
    fast = HLC("fast", clock=lambda: 9_000_000)
    remote = fast.now()
    after = slow.observe(remote)
    assert after > remote
    assert slow.now() > after


def test_parse_roundtrip():
    c = HLC("n1", clock=lambda: 42)
    ms, counter, node = parse(c.now())
    assert (ms, counter, node) == (42, 0, "n1")

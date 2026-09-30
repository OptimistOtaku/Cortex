from sietch.common.signing import sign, verify


def cmd(**kw):
    c = {"id": 7, "node": "rover-1", "kind": "discard", "body": {"item": "abc"}, "hlc": "0000000000001.00000.ground", **kw}
    c["sig"] = sign(c)
    return c


def test_valid_command_verifies():
    assert verify(cmd())


def test_tampered_body_is_refused():
    c = cmd()
    c["body"] = {"item": "something-else"}
    assert not verify(c)


def test_retargeted_command_is_refused():
    c = cmd()
    c["node"] = "rover-2"
    assert not verify(c)


def test_forged_signature_is_refused():
    c = cmd()
    c["sig"] = "0" * 64
    assert not verify(c)

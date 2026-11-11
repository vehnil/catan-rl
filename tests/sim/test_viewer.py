import json
import threading
import urllib.error
import urllib.request

import pytest

from catan.agents.registry import make_agent
from catan.engine import RulesConfig
from catan.sim.replay import save_replay
from catan.sim.runner import play_game
from catan.viewer.server import make_server


@pytest.fixture
def server(tmp_path):
    specs = ("greedy", "random", "random", "random")
    rec = play_game([make_agent(s) for s in specs], 0, RulesConfig(max_turns_per_game=60), specs=specs)
    save_replay(rec, tmp_path / "data" / "replays" / "run1", "g0")
    (tmp_path / "secret.txt").write_text("nope")  # outside the data dir
    srv = make_server(tmp_path / "data", port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", rec
    srv.shutdown()
    srv.server_close()


def _get(url):
    with urllib.request.urlopen(url) as r:
        return r.status, r.read()


def test_page_and_api(server):
    base, rec = server
    status, body = _get(base + "/")
    assert status == 200 and b"<svg" in body
    _, body = _get(base + "/api/replays")
    replays = json.loads(body)
    assert [r["path"] for r in replays] == ["replays/run1/g0.json.gz"]
    assert replays[0]["agents"] == rec.agents
    _, body = _get(base + "/api/timeline?path=replays/run1/g0.json.gz")
    assert len(json.loads(body)["steps"]) == len(rec.actions) + 1


@pytest.mark.parametrize(
    ("path", "code"), [("../secret.txt", 403), ("", 403), ("replays/run1/missing.json.gz", 404)]
)
def test_timeline_rejects_bad_paths(server, path, code):
    base, _ = server
    with pytest.raises(urllib.error.HTTPError) as err:
        _get(f"{base}/api/timeline?path={path}")
    assert err.value.code == code

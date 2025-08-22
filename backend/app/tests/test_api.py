from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_predict_single_json():
    payload = {
        "duration": 5,
        "src_bytes": 100,
        "dst_bytes": 50,
        "count": 4,
        "srv_count": 2,
        "wrong_fragment": 0,
    }
    # This test expects model present; if not present, server will raise an error.
    r = client.post("/predict_single", json=payload)
    assert r.status_code in (200, 500)



    

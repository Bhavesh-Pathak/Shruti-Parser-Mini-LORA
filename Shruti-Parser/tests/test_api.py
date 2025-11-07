from fastapi.testclient import TestClient
import pytest
from app import app

client = TestClient(app)

ALLOWED = {"command", "query", "teaching", "data"}

def test_analyze_happy_path():
    payload = {
        "text": "When writing unit tests, prefer small, focused test cases. Use fixtures for shared setup and assert one behavior per test so failures are easy to diagnose."
    }
    resp = client.post('/analyze', json=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert 'intent' in data and data['intent'] in ALLOWED
    assert 'actions' in data and isinstance(data['actions'], list)
    assert 'summary' in data and isinstance(data['summary'], str) and len(data['summary'].strip()) > 0


def test_empty_input_returns_400():
    resp = client.post('/analyze', json={"text": ""})
    assert resp.status_code == 400


def test_extract_endpoint():
    payload = {"text": "To onboard the server: install system packages, create the 'app' user, clone the repository, install Python dependencies, configure nginx, migrate the database, and restart the service."}
    resp = client.post('/extract', json=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert 'actions' in data and isinstance(data['actions'], list)
    # Expect at least one multi-word action
    assert any(len(a.split()) > 1 for a in data['actions'])

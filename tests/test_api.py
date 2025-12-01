"""Tests for FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient

from alyra_ai_ml.api.main import app


@pytest.fixture
def client() -> TestClient:
    """Create a test client."""
    return TestClient(app)


class TestRootEndpoint:
    """Tests for root endpoint."""

    def test_root_returns_welcome(self, client: TestClient) -> None:
        """Test that root returns welcome message."""
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"message": "Welcome to Alyra AI/ML API"}


class TestHealthEndpoint:
    """Tests for health endpoint."""

    def test_health_returns_healthy(self, client: TestClient) -> None:
        """Test that health endpoint returns healthy status."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}


class TestPredictEndpoint:
    """Tests for predict endpoint."""

    def test_predict_not_implemented(self, client: TestClient) -> None:
        """Test that predict returns 501 until implemented."""
        sample_request = {
            "marital_status": 1,
            "application_mode": 1,
            "application_order": 1,
            "course": 9003,
            "daytime_evening_attendance": 1,
            "previous_qualification": 1,
            "previous_qualification_grade": 130.0,
            "nationality": 1,
            "mothers_qualification": 37,
            "fathers_qualification": 37,
            "mothers_occupation": 5,
            "fathers_occupation": 9,
            "admission_grade": 125.0,
            "displaced": 1,
            "educational_special_needs": 0,
            "debtor": 0,
            "tuition_fees_up_to_date": 1,
            "gender": 1,
            "scholarship_holder": 0,
            "age_at_enrollment": 19,
            "international": 0,
            "curricular_units_1st_sem_credited": 0,
            "curricular_units_1st_sem_enrolled": 6,
            "curricular_units_1st_sem_evaluations": 8,
            "curricular_units_1st_sem_approved": 6,
            "curricular_units_1st_sem_grade": 13.5,
            "curricular_units_1st_sem_without_evaluations": 0,
            "curricular_units_2nd_sem_credited": 0,
            "curricular_units_2nd_sem_enrolled": 6,
            "curricular_units_2nd_sem_evaluations": 8,
            "curricular_units_2nd_sem_approved": 6,
            "curricular_units_2nd_sem_grade": 13.0,
            "curricular_units_2nd_sem_without_evaluations": 0,
            "unemployment_rate": 10.8,
            "inflation_rate": 1.4,
            "gdp": 1.74,
        }
        response = client.post("/api/v1/predict", json=sample_request)
        assert response.status_code == 501

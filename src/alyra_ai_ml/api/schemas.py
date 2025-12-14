"""Pydantic schemas for API request/response validation."""

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Request schema for prediction endpoint."""

    marital_status: int = Field(..., ge=1, le=6, description="Marital status code")
    application_mode: int = Field(..., description="Application mode code")
    application_order: int = Field(..., ge=0, le=9, description="Application order")
    course: int = Field(..., description="Course code")
    daytime_evening_attendance: int = Field(..., ge=0, le=1, description="1=Daytime, 0=Evening")
    previous_qualification: int = Field(..., description="Previous qualification code")
    previous_qualification_grade: float = Field(
        ..., ge=0, le=200, description="Previous qualification grade"
    )
    nationality: int = Field(..., description="Nationality code")
    mothers_qualification: int = Field(..., description="Mother's qualification code")
    fathers_qualification: int = Field(..., description="Father's qualification code")
    mothers_occupation: int = Field(..., description="Mother's occupation code")
    fathers_occupation: int = Field(..., description="Father's occupation code")
    admission_grade: float = Field(..., ge=0, le=200, description="Admission grade")
    displaced: int = Field(..., ge=0, le=1, description="Displaced from residence")
    educational_special_needs: int = Field(..., ge=0, le=1, description="Educational special needs")
    debtor: int = Field(..., ge=0, le=1, description="Is debtor")
    tuition_fees_up_to_date: int = Field(..., ge=0, le=1, description="Tuition fees up to date")
    gender: int = Field(..., ge=0, le=1, description="0=Female, 1=Male")
    scholarship_holder: int = Field(..., ge=0, le=1, description="Is scholarship holder")
    age_at_enrollment: int = Field(..., ge=17, le=100, description="Age at enrollment")
    international: int = Field(..., ge=0, le=1, description="Is international student")
    curricular_units_1st_sem_credited: int = Field(..., ge=0, description="1st sem credited units")
    curricular_units_1st_sem_enrolled: int = Field(..., ge=0, description="1st sem enrolled units")
    curricular_units_1st_sem_evaluations: int = Field(..., ge=0, description="1st sem evaluations")
    curricular_units_1st_sem_approved: int = Field(..., ge=0, description="1st sem approved units")
    curricular_units_1st_sem_grade: float = Field(..., ge=0, le=20, description="1st sem grade")
    curricular_units_1st_sem_without_evaluations: int = Field(
        ..., ge=0, description="1st sem without evaluations"
    )
    curricular_units_2nd_sem_credited: int = Field(..., ge=0, description="2nd sem credited units")
    curricular_units_2nd_sem_enrolled: int = Field(..., ge=0, description="2nd sem enrolled units")
    curricular_units_2nd_sem_evaluations: int = Field(..., ge=0, description="2nd sem evaluations")
    curricular_units_2nd_sem_approved: int = Field(..., ge=0, description="2nd sem approved units")
    curricular_units_2nd_sem_grade: float = Field(..., ge=0, le=20, description="2nd sem grade")
    curricular_units_2nd_sem_without_evaluations: int = Field(
        ..., ge=0, description="2nd sem without evaluations"
    )
    unemployment_rate: float = Field(..., ge=0, le=100, description="Unemployment rate")
    inflation_rate: float = Field(..., ge=-10, le=20, description="Inflation rate")
    gdp: float = Field(..., ge=-10, le=10, description="GDP")

    model_config = {"extra": "forbid"}


class PredictionResponse(BaseModel):
    """Response schema for prediction endpoint."""

    prediction: str = Field(..., description="Predicted outcome: Dropout, Enrolled, or Graduate")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence")
    probabilities: dict[str, float] = Field(..., description="Probabilities for each class")

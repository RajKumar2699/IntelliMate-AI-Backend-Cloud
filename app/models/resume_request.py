from pydantic import BaseModel, EmailStr, Field
from typing import List, Optional


class ExperienceItem(BaseModel):
    company: str = Field(..., example="Askme Technologies")
    role: str = Field(..., example="iOS Developer")
    start_date: str = Field(..., example="2024-01")
    end_date: Optional[str] = Field(None, example="2026-07")
    description: str = Field(..., example="Built AI-integrated mobile apps using Swift and UIKit.")


class EducationItem(BaseModel):
    institution: str = Field(..., example="XYZ University")
    degree: str = Field(..., example="B.Tech")
    field_of_study: str = Field(..., example="Computer Science")
    start_date: str = Field(..., example="2019")
    end_date: Optional[str] = Field(None, example="2023")


class ProjectItem(BaseModel):
    title: str = Field(..., example="IntelliMate AI")
    description: str = Field(..., example="AI-powered app with chat and resume builder.")
    technologies: List[str] = Field(default_factory=list, example=["Swift", "UIKit", "FastAPI"])


class ResumeBaseRequest(BaseModel):
    full_name: str = Field(..., example="John Doe")
    email: EmailStr = Field(..., example="john@example.com")
    phone: str = Field(..., example="+91-9876543210")
    target_role: str = Field(..., example="Senior iOS Developer")
    years_of_experience: Optional[int] = Field(None, example=3)
    summary: Optional[str] = Field(None, example="iOS developer with strong UIKit experience.")
    skills: List[str] = Field(default_factory=list, example=["Swift", "UIKit", "MVVM"])
    experience: List[ExperienceItem] = Field(default_factory=list)
    education: List[EducationItem] = Field(default_factory=list)
    projects: List[ProjectItem] = Field(default_factory=list)


class GenerateSummaryRequest(BaseModel):
    full_name: str
    target_role: str
    years_of_experience: Optional[int] = None
    skills: List[str] = Field(default_factory=list)
    summary: Optional[str] = None


class ImproveExperienceRequest(BaseModel):
    target_role: str
    experience: List[ExperienceItem]


class ATSKeywordsRequest(BaseModel):
    target_role: str
    skills: List[str] = Field(default_factory=list)
    experience: List[ExperienceItem] = Field(default_factory=list)


class GenerateFullResumeRequest(ResumeBaseRequest):
    pass
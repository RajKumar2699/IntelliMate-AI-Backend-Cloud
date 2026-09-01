from pydantic import BaseModel, Field
from typing import List, Optional


class ResumeAnalysisResponse(BaseModel):
    extracted_text: str
    overall_score: int = Field(..., ge=0, le=100)
    ats_score: int = Field(..., ge=0, le=100)
    matched_keywords: List[str] = Field(default_factory=list)
    missing_keywords: List[str] = Field(default_factory=list)
    strengths: List[str] = Field(default_factory=list)
    improvement_points: List[str] = Field(default_factory=list)
    jd_alignment_points: List[str] = Field(default_factory=list)
    updated_resume_text: Optional[str] = None
    report_text: str
    pdf_download_url: Optional[str] = None
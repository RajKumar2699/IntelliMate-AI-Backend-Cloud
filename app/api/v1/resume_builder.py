# app/api/v1/resume_builder.py
import os
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.models.resume_response import ResumeAnalysisResponse
from app.services.resume_builder_service import ResumeBuilderService
from pydantic import BaseModel


router = APIRouter(prefix="/resume", tags=["Resume Analyzer"])

resume_builder_service = ResumeBuilderService()

class UpdatedResumeAnalysisRequest(BaseModel):
    updated_resume_text: str
    job_description: str

@router.post("/analyze", response_model=ResumeAnalysisResponse)
async def analyze_resume(
    file: UploadFile = File(...),
    job_description: str | None = Form(default=None),
):
    return await resume_builder_service.analyze_resume(
        file=file,
        job_description=job_description,
    )

@router.post("/analyze-updated", response_model=ResumeAnalysisResponse)
async def analyze_updated_resume(payload: UpdatedResumeAnalysisRequest):
    updated_resume_text = payload.updated_resume_text.strip()
    job_description = payload.job_description.strip()

    if not updated_resume_text:
        raise HTTPException(status_code=400, detail="updated_resume_text is required")

    if not job_description:
        raise HTTPException(status_code=400, detail="job_description is required")

    return await resume_builder_service.analyze_updated_resume(
        updated_resume_text=updated_resume_text,
        job_description=job_description,
    )


@router.get("/download-pdf/{filename}")
async def download_pdf(filename: str):
    file_path = os.path.join("output", filename)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="PDF file not found.")

    return FileResponse(
        path=file_path,
        media_type="application/pdf",
        filename=filename
    )
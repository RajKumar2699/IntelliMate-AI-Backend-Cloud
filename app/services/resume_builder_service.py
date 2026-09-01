import io
import os
import re
import uuid
from typing import List, Tuple

import pdfplumber
from docx import Document
from fastapi import HTTPException, UploadFile

from app.models.resume_response import ResumeAnalysisResponse
from app.services.openai_service import openai_service
from app.utils.pdf_generator import generate_resume_pdf


class ResumeBuilderService:
    ALLOWED_EXTENSIONS = (".pdf", ".docx", ".txt")

    async def analyze_resume(
        self,
        file: UploadFile,
        job_description: str | None = None,
    ) -> ResumeAnalysisResponse:
        filename = (file.filename or "").lower()

        if not filename.endswith(self.ALLOWED_EXTENSIONS):
            raise HTTPException(
                status_code=400,
                detail="Unsupported file format. Please upload PDF, DOCX, or TXT."
            )

        file_bytes = await file.read()
        if not file_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        resume_text = self._extract_text(filename, file_bytes).strip()
        if not resume_text:
            raise HTTPException(status_code=400, detail="Could not extract text from the uploaded resume.")

        jd_text = (job_description or "").strip()

        matched_keywords, missing_keywords = self._compare_with_jd(
            resume_text=resume_text,
            job_description=jd_text
        )

        strengths = self._generate_strengths(resume_text)
        improvement_points = self._generate_improvement_points(
            resume_text=resume_text,
            missing_keywords=missing_keywords,
            has_jd=bool(jd_text)
        )

        jd_alignment_points = self._generate_jd_alignment_points(
            resume_text=resume_text,
            job_description=jd_text,
            missing_keywords=missing_keywords
        )

        ats_score, overall_score = self._calculate_scores(
            resume_text=resume_text,
            matched_keywords=matched_keywords,
            missing_keywords=missing_keywords
        )

        updated_resume_text = resume_text
        pdf_download_url = None

        if jd_text:
            updated_resume_text = openai_service.improve_resume(
                resume_text=resume_text,
                job_description=jd_text
            )

            pdf_id = str(uuid.uuid4())
            pdf_filename = f"updated_resume_{pdf_id}.pdf"
            pdf_path = os.path.join("output", pdf_filename)
            generate_resume_pdf(updated_resume_text, pdf_path)

            pdf_download_url = f"/api/v1/resume/download-pdf/{pdf_filename}"

        report_text = self._generate_report_text(
            overall_score=overall_score,
            ats_score=ats_score,
            matched_keywords=matched_keywords,
            missing_keywords=missing_keywords,
            strengths=strengths,
            improvement_points=improvement_points,
            jd_alignment_points=jd_alignment_points,
            updated_resume_text=updated_resume_text
        )

        return ResumeAnalysisResponse(
            extracted_text=resume_text,
            overall_score=overall_score,
            ats_score=ats_score,
            matched_keywords=matched_keywords,
            missing_keywords=missing_keywords,
            strengths=strengths,
            improvement_points=improvement_points,
            jd_alignment_points=jd_alignment_points,
            updated_resume_text=updated_resume_text,
            report_text=report_text,
            pdf_download_url=pdf_download_url
        )

    async def analyze_updated_resume(
        self,
        updated_resume_text: str,
        job_description: str,
    ) -> ResumeAnalysisResponse:
        resume_text = updated_resume_text.strip()
        jd_text = (job_description or "").strip()

        if not resume_text:
            raise HTTPException(status_code=400, detail="Updated resume text is empty.")

        matched_keywords, missing_keywords = self._compare_with_jd(
            resume_text=resume_text,
            job_description=jd_text
        )

        strengths = self._generate_strengths(resume_text)
        improvement_points = self._generate_improvement_points(
            resume_text=resume_text,
            missing_keywords=missing_keywords,
            has_jd=bool(jd_text)
        )

        jd_alignment_points = self._generate_jd_alignment_points(
            resume_text=resume_text,
            job_description=jd_text,
            missing_keywords=missing_keywords
        )

        ats_score, overall_score = self._calculate_scores(
            resume_text=resume_text,
            matched_keywords=matched_keywords,
            missing_keywords=missing_keywords
        )

        report_text = self._generate_report_text(
            overall_score=overall_score,
            ats_score=ats_score,
            matched_keywords=matched_keywords,
            missing_keywords=missing_keywords,
            strengths=strengths,
            improvement_points=improvement_points,
            jd_alignment_points=jd_alignment_points,
            updated_resume_text=resume_text
        )

        return ResumeAnalysisResponse(
            extracted_text=resume_text,
            overall_score=overall_score,
            ats_score=ats_score,
            matched_keywords=matched_keywords,
            missing_keywords=missing_keywords,
            strengths=strengths,
            improvement_points=improvement_points,
            jd_alignment_points=jd_alignment_points,
            updated_resume_text=resume_text,
            report_text=report_text,
            pdf_download_url=None
        )

    def _extract_text(self, filename: str, file_bytes: bytes) -> str:
        if filename.endswith(".pdf"):
            return self._extract_text_from_pdf(file_bytes)
        if filename.endswith(".docx"):
            return self._extract_text_from_docx(file_bytes)
        if filename.endswith(".txt"):
            return self._extract_text_from_txt(file_bytes)
        return ""

    def _extract_text_from_pdf(self, file_bytes: bytes) -> str:
        text_parts = []
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                text_parts.append(page.extract_text() or "")
        return "\n".join(text_parts)

    def _extract_text_from_docx(self, file_bytes: bytes) -> str:
        document = Document(io.BytesIO(file_bytes))
        return "\n".join([para.text for para in document.paragraphs if para.text.strip()])

    def _extract_text_from_txt(self, file_bytes: bytes) -> str:
        return file_bytes.decode("utf-8", errors="ignore")

    def _normalize_words(self, text: str) -> List[str]:
        return re.findall(r"[a-zA-Z0-9\+\#\.]+", text.lower())

    def _extract_candidate_keywords(self, text: str) -> List[str]:
        common_keywords = [
            "swift", "uikit", "swiftui", "mvvm", "viper", "objective-c",
            "xcode", "ios", "rest", "api", "firebase", "push", "notifications",
            "core data", "gcd", "combine", "async", "await", "autolayout",
            "dependency injection", "unit testing", "xctest",
            "cocoapods", "spm", "app store", "analytics", "deep linking"
        ]
        lower_text = text.lower()
        found = [keyword for keyword in common_keywords if keyword in lower_text]
        return list(dict.fromkeys(found))

    def _compare_with_jd(self, resume_text: str, job_description: str) -> Tuple[List[str], List[str]]:
        resume_keywords = set(self._extract_candidate_keywords(resume_text))

        if not job_description.strip():
            return sorted(resume_keywords), []

        jd_keywords = set(self._extract_candidate_keywords(job_description))
        if not jd_keywords:
            jd_words = self._normalize_words(job_description)
            jd_keywords = {word for word in jd_words if len(word) > 3}

        matched = sorted(resume_keywords.intersection(jd_keywords))
        missing = sorted(jd_keywords.difference(resume_keywords))
        return matched[:25], missing[:25]

    def _calculate_scores(self, resume_text: str, matched_keywords: List[str], missing_keywords: List[str]) -> Tuple[int, int]:
        ats_score = 55
        lower_text = resume_text.lower()

        if "summary" in lower_text or "professional summary" in lower_text:
            ats_score += 8
        if "experience" in lower_text:
            ats_score += 8
        if "skills" in lower_text:
            ats_score += 8
        if "education" in lower_text:
            ats_score += 6

        ats_score += min(len(matched_keywords) * 3, 18)
        ats_score -= min(len(missing_keywords) * 2, 20)

        ats_score = max(0, min(100, ats_score))
        overall_score = max(0, min(100, ats_score + 5 if ats_score < 95 else ats_score))
        return ats_score, overall_score

    def _generate_strengths(self, resume_text: str) -> List[str]:
        lower_text = resume_text.lower()
        strengths = []

        if "swift" in lower_text:
            strengths.append("Strong relevance for iOS roles through Swift experience.")
        if "uikit" in lower_text or "swiftui" in lower_text:
            strengths.append("Resume shows hands-on iOS UI development exposure.")
        if "mvvm" in lower_text or "viper" in lower_text:
            strengths.append("Architecture-related keywords improve technical alignment.")
        if "api" in lower_text or "rest" in lower_text:
            strengths.append("Backend integration exposure is visible in the resume.")
        if "firebase" in lower_text or "analytics" in lower_text:
            strengths.append("Product-focused mobile development signals are present.")

        if not strengths:
            strengths.append("Resume contains technical experience, but it needs stronger role-specific keywords.")

        return strengths[:5]

    def _generate_improvement_points(self, resume_text: str, missing_keywords: List[str], has_jd: bool) -> List[str]:
        lower_text = resume_text.lower()
        points = []

        if "professional summary" not in lower_text and "summary" not in lower_text:
            points.append("Add a professional summary tailored to your target role.")
        if "skills" not in lower_text:
            points.append("Add a dedicated skills section with role-specific keywords.")
        if "education" not in lower_text:
            points.append("Add an education section to complete the resume structure.")
        if len(resume_text.split()) < 180:
            points.append("Add more project, achievement, and impact details to strengthen content depth.")
        points.append("Rewrite experience bullets with measurable impact and action-driven wording.")

        if has_jd and missing_keywords:
            points.append("Add the missing job-description keywords naturally in the summary, skills, and experience sections.")

        return list(dict.fromkeys(points))[:6]

    def _generate_jd_alignment_points(self, resume_text: str, job_description: str, missing_keywords: List[str]) -> List[str]:
        if not job_description.strip():
            return ["No job description provided, so JD-specific alignment was not evaluated."]

        if missing_keywords:
            return [
                "Your resume does not currently reflect several keywords required in the job description.",
                "Update the professional summary to mirror the target role and key technical stack from the JD.",
                "Add or strengthen project bullets that directly support the JD expectations.",
                "Mention tools, frameworks, and platform responsibilities using the same wording style as the JD."
            ]

        return ["Your resume already aligns well with the provided job description keywords."]

    def _generate_report_text(
        self,
        overall_score: int,
        ats_score: int,
        matched_keywords: List[str],
        missing_keywords: List[str],
        strengths: List[str],
        improvement_points: List[str],
        jd_alignment_points: List[str],
        updated_resume_text: str
    ) -> str:
        lines = [
            "RESUME ANALYSIS REPORT",
            "======================",
            "",
            f"Overall Score: {overall_score}/100",
            f"ATS Score: {ats_score}/100",
            "",
            "Matched Keywords:",
            *([f"- {x}" for x in matched_keywords] if matched_keywords else ["- No matched keywords detected."]),
            "",
            "Missing Keywords:",
            *([f"- {x}" for x in missing_keywords] if missing_keywords else ["- No major missing keywords detected."]),
            "",
            "Strengths:",
            *[f"- {x}" for x in strengths],
            "",
            "Improvement Points:",
            *[f"- {x}" for x in improvement_points],
            "",
            "JD Alignment Points:",
            *[f"- {x}" for x in jd_alignment_points],
            "",
            "UPDATED RESUME",
            "==============",
            updated_resume_text
        ]
        return "\n".join(lines)
"""Single aggregate router."""

from fastapi import APIRouter

from app.api import assessments, auth, coding, kb, learning, library, practice, projects, stats

router = APIRouter()
for module in (auth, library, kb, learning, practice, coding, projects, assessments, stats):
    router.include_router(module.router)

"""Single aggregate router."""

from fastapi import APIRouter

from app.api import assessments, auth, coding, kb, learning, library, practice, projects, stats
from app.game import api as game

router = APIRouter()
for module in (auth, library, kb, learning, practice, coding, projects, assessments, stats, game):
    router.include_router(module.router)

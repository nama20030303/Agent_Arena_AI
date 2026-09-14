"""Skill graph service: prerequisites, readiness, unlockability, UI payload."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Skill, SkillDependency, Topic, UserSkill


@dataclass
class Node:
    code: str
    name: str
    level: int
    category: str
    topic_code: str | None
    difficulty: int
    keywords: list[str]


class SkillGraph:
    """In-memory view of the curriculum graph (149 skills - cheap to build per request)."""

    def __init__(self, session: Session) -> None:
        self.session = session
        self.nodes: dict[str, Node] = {}
        self.topics: dict[str, Topic] = {}
        self.requires: dict[str, set[str]] = defaultdict(set)   # skill -> prerequisites
        self.unlocks: dict[str, set[str]] = defaultdict(set)    # skill -> dependents
        self.topic_skills: dict[str, list[str]] = defaultdict(list)
        self._load()

    def _load(self) -> None:
        topic_by_id: dict[int, Topic] = {}
        for topic in self.session.execute(select(Topic)).scalars():
            self.topics[topic.code] = topic
            topic_by_id[topic.id] = topic
        skill_by_id: dict[int, str] = {}
        for skill in self.session.execute(select(Skill)).scalars():
            topic = topic_by_id.get(skill.topic_id or -1)
            self.nodes[skill.code] = Node(
                code=skill.code,
                name=skill.name,
                level=skill.level,
                category=skill.category,
                topic_code=topic.code if topic else None,
                difficulty=skill.difficulty,
                keywords=list(skill.keywords or []),
            )
            skill_by_id[skill.id] = skill.code
            if topic is not None:
                self.topic_skills[topic.code].append(skill.code)
        for dep in self.session.execute(select(SkillDependency)).scalars():
            parent, child = skill_by_id.get(dep.skill_id), skill_by_id.get(dep.depends_on_id)
            if parent and child:
                self.requires[parent].add(child)
                self.unlocks[child].add(parent)

    # ------------------------------------------------------------- traversal
    def prerequisites(self, code: str, *, depth: int | None = None) -> list[str]:
        """Transitive prerequisite closure (what must be solid before this skill)."""
        out: list[str] = []
        seen: set[str] = {code}
        queue: deque[tuple[str, int]] = deque((c, 1) for c in sorted(self.requires.get(code, ())))
        while queue:
            node, level = queue.popleft()
            if node in seen:
                continue
            seen.add(node)
            out.append(node)
            if depth is None or level < depth:
                queue.extend((parent, level + 1) for parent in sorted(self.requires.get(node, ())))
        return out

    def dependents(self, code: str, *, depth: int = 2) -> list[str]:
        out: list[str] = []
        seen: set[str] = {code}
        queue: deque[tuple[str, int]] = deque((c, 1) for c in sorted(self.unlocks.get(code, ())))
        while queue:
            node, level = queue.popleft()
            if node in seen:
                continue
            seen.add(node)
            out.append(node)
            if level < depth:
                queue.extend((child, level + 1) for child in sorted(self.unlocks.get(node, ())))
        return out

    def topic_prerequisites(self, topic_code: str) -> list[str]:
        topic = self.topics.get(topic_code)
        if topic is not None and topic.prerequisites:
            return list(topic.prerequisites)
        return []

    def topic_prerequisite_skills(self, topic_code: str, *, depth: int = 2) -> list[tuple[str, int]]:
        """
        (skill, distance) for what must be solid *before* a topic: the prerequisite closure of
        the topic's own skills, excluding the skills the topic is about to teach.
        """
        own = set(self.topic_skills.get(topic_code, []))
        out: dict[str, int] = {}
        for skill_code in self.topic_skills.get(topic_code, []):
            for level, prereq in enumerate(self._prerequisites_with_depth(skill_code, depth=depth), start=1):
                if prereq in own:
                    continue
                out[prereq] = min(out.get(prereq, 99), level)
        return sorted(out.items(), key=lambda kv: (kv[1], kv[0]))

    def topic_required_skills(self, topic_code: str, *, depth: int = 2) -> list[tuple[str, int]]:
        """
        (skill_code, distance) needed for a topic: its own skills at distance 0 and the
        prerequisite closure up to `depth`. Distance is used to weight importance - a far
        ancestor should not block a topic as strongly as its direct skills.
        """
        out: dict[str, int] = {}
        for skill_code in self.topic_skills.get(topic_code, []):
            out.setdefault(skill_code, 0)
            for level, prereq in enumerate(self._prerequisites_with_depth(skill_code, depth=depth), start=1):
                if prereq not in self.topic_skills.get(topic_code, []):
                    out.setdefault(prereq, level)
        return sorted(out.items(), key=lambda kv: (kv[1], kv[0]))

    def _prerequisites_with_depth(self, code: str, *, depth: int) -> list[str]:
        out: list[str] = []
        seen = {code}
        frontier = sorted(self.requires.get(code, ()))
        for level in range(1, depth + 1):
            if not frontier:
                break
            nxt: list[str] = []
            for node in frontier:
                if node in seen:
                    continue
                seen.add(node)
                out.append(node)
                nxt.extend(sorted(self.requires.get(node, ())))
            frontier = nxt
        return out

    def topic_assessment(self, user_id: int, topic_code: str, *, scores: dict[str, tuple[float, float]] | None = None) -> dict[str, Any]:
        """
        Score the learner on a topic using *assessed* skills only (unknown != zero).
        `peak` is the best demonstrated dimension - skipping a topic requires demonstrated
        competence, while calling it "mastered" requires breadth.
        """
        scores = scores if scores is not None else self.user_scores(user_id)
        skills = self.topic_skills.get(topic_code, [])
        assessed = [scores[c] for c in skills if c in scores and scores[c][1] > 0]
        if not assessed:
            return {"avg": 0.0, "peak": 0.0, "assessed": 0, "coverage": 0.0, "confidence": 0.0}
        avg = sum(a[0] for a in assessed) / len(assessed)
        return {
            "avg": round(avg, 2),
            "peak": round(max(a[0] for a in assessed), 2),  # best skill-level knowledge
            "assessed": len(assessed),
            "coverage": round(len(assessed) / max(1, len(skills)), 3),
            "confidence": round(sum(a[1] for a in assessed) / len(assessed), 3),
        }

    def topic_dimension_scores(self, user_id: int, topic_code: str) -> dict[str, float]:
        """Best per-dimension score across the topic's skills (code/math/theory evidence)."""
        from app.models import Skill as _Skill, UserSkill as _US
        from app.services.user_knowledge import DIMENSIONS

        rows = self.session.execute(
            select(_US, _Skill).join(_Skill, _Skill.id == _US.skill_id).where(
                _US.user_id == user_id, _Skill.code.in_(self.topic_skills.get(topic_code, []) or ["__none__"])
            )
        ).all()
        best: dict[str, float] = {}
        for us, _skill in rows:
            for dim in DIMENSIONS:
                value = float(getattr(us, dim, 0.0) or 0.0)
                if value > best.get(dim, 0.0):
                    best[dim] = value
        return best

    # ------------------------------------------------------------- readiness
    def user_scores(self, user_id: int) -> dict[str, tuple[float, float]]:
        rows = self.session.execute(
            select(UserSkill, Skill).join(Skill, Skill.id == UserSkill.skill_id).where(UserSkill.user_id == user_id)
        ).all()
        return {skill.code: (us.knowledge_score or 0.0, us.confidence or 0.0) for us, skill in rows}

    def readiness(self, user_id: int, topic_code: str, *, scores: dict[str, tuple[float, float]] | None = None, prior_score: float | None = None) -> dict[str, Any]:
        """
        0-100 readiness = weighted prerequisite satisfaction (the topic's own skills are what
        the learner is about to learn, so they must not drag the estimate to zero).
        Distant ancestors count less: weight 1/(1+distance).
        """
        scores = scores if scores is not None else self.user_scores(user_id)
        required = self.topic_prerequisite_skills(topic_code)
        if not required:
            # no prerequisites in the graph -> nothing blocks this topic
            return {"readiness": 100.0, "unmet": [], "required": []}
        total_weight, acc = 0.0, 0.0
        unmet: list[dict[str, Any]] = []
        for code, distance in required:
            node = self.nodes.get(code)
            if node is None:
                continue
            importance = 1.0 / (1.0 + distance)
            weight = (0.6 + 0.4 * node.difficulty) * importance
            score, confidence = scores.get(code, (0.0, 0.0))
            if confidence <= 0 and prior_score is not None:
                # never-assessed prerequisite: assume partial knowledge proportional to the
                # learner's demonstrated level, so skipping the diagnostic cannot lock the map
                score, confidence = prior_score, 0.2
            effective = score * (0.5 + 0.5 * min(1.0, confidence * 1.6))
            total_weight += weight
            acc += weight * effective
            if effective < 55.0:
                unmet.append({"code": code, "name": node.name, "score": round(effective, 1), "level": node.level, "distance": distance})
        readiness = acc / total_weight if total_weight else 0.0
        unmet.sort(key=lambda d: (d["distance"], d["score"]))
        return {"readiness": round(readiness, 1), "unmet": unmet[:8], "required": [c for c, _d in required]}

    def next_topics_by_level(self, *, from_level: int, to_level: int = 10) -> list[Topic]:
        return [t for t in sorted(self.topics.values(), key=lambda t: (t.level, t.order_index)) if from_level <= t.level <= to_level]

    def skill_of_topic(self, topic_code: str) -> list[str]:
        return list(self.topic_skills.get(topic_code, []))

    def graph_payload(self, user_id: int | None = None) -> dict[str, Any]:
        scores = self.user_scores(user_id) if user_id else {}
        nodes = [
            {
                "id": n.code,
                "name": n.name,
                "level": n.level,
                "category": n.category,
                "topic": n.topic_code,
                "score": round(scores.get(n.code, (0, 0))[0], 1),
                "confidence": round(scores.get(n.code, (0, 0))[1], 2),
            }
            for n in sorted(self.nodes.values(), key=lambda x: (x.level, x.code))
        ]
        links = [
            {"source": child, "target": parent, "type": "requires"}
            for child, parents in self.requires.items()
            for parent in sorted(parents)
        ]
        return {"nodes": nodes, "links": links, "levels": sorted({n["level"] for n in nodes})}

    def shortest_path(self, from_code: str, to_code: str) -> list[str]:
        """Learning path between two skills (following 'unlocks' edges)."""
        if from_code == to_code:
            return [from_code]
        queue = deque([(from_code, [from_code])])
        seen = {from_code}
        while queue:
            node, path = queue.popleft()
            for nxt in sorted(self.unlocks.get(node, ())):
                if nxt in seen:
                    continue
                if nxt == to_code:
                    return path + [nxt]
                seen.add(nxt)
                queue.append((nxt, path + [nxt]))
        return []


_graph_cache: dict[int, SkillGraph] = {}


def get_graph(session: Session) -> SkillGraph:
    return SkillGraph(session)

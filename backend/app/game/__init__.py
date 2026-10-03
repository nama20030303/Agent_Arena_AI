"""
AI Knowledge RPG — game layer.

Built on top of the existing academy backend: shares auth (User/UserSession),
the AI gateway (app.ai.manager) and the SQLite persistence layer, but keeps its
own entities so the game economy stays server-authoritative and auditable.

Design rule (spec §74): the server is the only source of truth for XP, levels,
chests, inventory and achievements. The client never sends reward values.
"""

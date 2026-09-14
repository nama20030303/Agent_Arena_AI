/** Shapes returned by the FastAPI backend (kept intentionally loose where the API is dynamic). */

export type Level = number

export interface Profile {
  id: number
  username: string
  display_name: string
  background: string
  target_role: string
  goal: string
  daily_minutes: number
  level_index: number
  level_label: string
  onboarded: boolean
  diagnostic_done: boolean
  web_search_allowed?: boolean
  ai_enabled?: boolean
  reply_language?: string
  settings?: Record<string, unknown>
  created_at?: string
}

export interface SkillState {
  knowledge_score: number
  confidence: number
  attempts: number
  successes?: number
  success_rate?: number
  streak?: number
  failure_count?: number
  difficulty_level?: number
  mastery_state: string
  theory: number
  math: number
  coding: number
  problem_solving: number
  engineering: number
  last_review?: string | null
  next_review?: string | null
  xp?: number
  minutes_studied?: number
  last_error_type?: string
}

export interface TopicSummary {
  code: string
  name: string
  level: number
  domain: string
  summary: string
  keywords: string[]
  skills: string[]
  avg_score: number
  prerequisites: string[]
  est_hours: number
}

export interface TopicDetail extends TopicSummary {
  skills: Array<{
    code: string
    name: string
    category: string
    difficulty: number
    state: Partial<SkillState>
    prerequisites: string[]
  }>
  readiness: { readiness: number; unmet: Array<{ code: string; name: string; score: number; distance?: number }>; required: string[] }
  counts: { questions: number; practice: number; coding: number; documents: number }
  coding_tasks: Array<{ id: number; slug: string; title: string; difficulty: number; from_scratch: boolean; libraries: string[] }>
  documents: Array<{ id: number; title: string; author: string; chunks: number; matched: boolean }>
  library_hits: Array<{ document_id: number; document: string; chapter: string; page: number | null; score: number; snippet: string; citation: Citation }>
  suggested_modes: string[]
}

export interface Citation {
  index?: number
  document_id?: number
  chunk_id?: number
  document: string
  author?: string
  chapter?: string
  section?: string
  page?: number | null
  page_end?: number | null
  url?: string
  tier?: number
  label?: string
}

export interface QuestionOptions { id: number; text: string }

export interface Question {
  id: number
  code?: string
  type: string
  difficulty: number
  level?: number
  stem: string
  options: QuestionOptions[]
  starter_code?: string
  hints?: string[]
  topic?: string
  skills?: string[]
  generated_by?: string
  citation?: Citation
  correct_option?: number | null
  expected_answer?: string
  expected_points?: string[]
  explanation?: string
}

export interface Verdict {
  correct: boolean
  score: number
  dimensions: Record<string, number>
  error_type: string
  what_is_wrong?: string
  correction?: string
  feedback?: string
  missing_points?: string[]
  check_question?: { stem: string; expected_answer?: string } | null
  evaluated_by?: string
  sources?: Citation[]
  usage?: Record<string, unknown>
}

export interface AnswerResult {
  attempt_id: number
  question_id: number
  verdict: Verdict
  engine: string
  skill_updates: Array<{ skill_code: string; before: number; after: number; delta: number; attempts: number; confidence: number; mastery_state: string; difficulty_level: number }>
  learning_engine: { actions: Array<{ type: string; message?: string; skill?: string; prerequisite?: string; to?: number }>; plan_updated: boolean }
  retry_question?: Question | null
  xp: number
  sources: Citation[]
}

export interface PlanItem {
  id: string
  kind: 'review' | 'learn' | 'practice' | 'coding' | 'project' | 'reflection' | 'exam_prep'
  code: string
  title: string
  why: string
  minutes: number
  done: boolean
  payload?: Record<string, unknown>
}

export interface DailyPlan {
  date: string
  items: PlanItem[]
  rationale: string
  minutes_budget: number
  used_minutes: number
  completed: number
  total: number
  progress: number
  generated_by: string
  focus_topic?: string
  weak_skills?: string[]
  note?: string
}

export interface TeacherReply {
  conversation_id: number
  mode: string
  topic: string
  answer: string
  sources: Citation[]
  weak_matches?: Citation[]
  grounded: boolean
  kb_coverage: number
  web_note?: string
  conflicts: Array<{ term: string; a: { text: string; source: string }; b: { text: string; source: string }; note: string }>
  engine: string
  ai: { usage: Record<string, number | boolean>; message?: string }
  attachments: Array<Question | PracticeTask>
  pending: Record<string, unknown>
  suggestions: string[]
  message_id: number
}

export interface PracticeTask {
  id: number
  title: string
  topic: string
  level: string
  kind: string
  difficulty: number
  statement: string
  given_data?: string
  steps_required: string[]
  starter_code?: string
  est_minutes: number
  skills: string[]
  citation?: Citation
  generated_by?: string
  solved?: boolean
  solution?: string
  expected_answer?: string
}

export interface CodingTaskSummary {
  id: number
  slug: string
  title: string
  description: string
  difficulty: number
  difficulty_label: string
  level: string
  libraries: string[]
  skills: string[]
  topic: string
  from_scratch: boolean
  starter_code: string
  hint_count: number
  est_minutes: number
  xp: number
  test_count: number
  attempts?: number
  solved?: boolean
  last_attempt?: string | null
  solution?: string
  solution_explanation?: string
}

export interface TestResult { name: string; passed: boolean; message: string; runtime_ms: number }

export interface RunResult {
  ok: boolean
  total_tests: number
  passed_tests: number
  tests: TestResult[]
  stdout: string
  error: string
  runtime_ms: number
  violated?: string[]
  timed_out?: boolean
  mode?: string
  passed_all?: boolean
  solution_unlocked?: boolean
  attempt_id?: number
  solved?: boolean
  skill_updates?: AnswerResult['skill_updates']
  learning_engine?: AnswerResult['learning_engine']
}

export interface DocumentRow {
  id: number
  title: string
  filename: string
  doc_type: string
  author: string
  kind: string
  tier: number
  size_bytes: number
  size_human: string
  page_count: number
  word_count: number
  chunk_count: number
  status: 'uploading' | 'processing' | 'indexed' | 'error'
  error: string
  topics: string[]
  tags: string[]
  notes?: string
  url?: string
  created_at?: string
  indexed_at?: string
  progress?: { stage: string; percent: number; message: string }
}

export interface ChunkHit {
  document_id: number
  document: string
  author?: string
  chunk_id?: number | null
  chapter?: string
  section?: string
  page_start?: number | null
  snippet: string
  score: number
  tier?: number
  url?: string
  match_type?: string
}

export interface ProjectPayload {
  id: number
  slug: string
  title: string
  description: string
  target_level: string
  domain: string
  difficulty: number
  est_hours: number
  stack: string[]
  skills: string[]
  deliverables: string[]
  milestones: Array<{ title: string; description: string; acceptance_criteria: string[] }>
  rubric: Array<{ dimension: string; weight: number; levels?: Record<string, string> }>
  guidance?: string
  dataset_hint?: string
  skill_readiness?: number
  recommended?: boolean
  enrollment?: EnrollmentPayload | null
}

export interface Milestone {
  id: number
  title: string
  description: string
  acceptance_criteria: string[]
  order: number
  status: string
  review_notes?: string
  completed_at?: string | null
}

export interface EnrollmentPayload {
  id: number
  project_id: number
  project_title: string
  status: string
  progress: number
  milestones_done: number
  milestones_total: number
  minutes_spent: number
  milestones: Milestone[]
  log?: Array<{ id: number; kind: string; author: string; content: string; evaluation?: Record<string, unknown>; at?: string }>
}

export interface EvaluationReport {
  engine: string
  total: number
  grade: string
  dimensions: Array<{ dimension: string; score: number; evidence?: string; next_action?: string; comment?: string }>
  summary: string
  strengths: string[]
  gaps: string[]
  next_actions: string[]
  note?: string
  interview_questions?: string[]
}

export interface ExamInfo {
  id: number
  code: string
  title: string
  description: string
  target_level: string
  duration_minutes: number
  passing_score: number
  sections: Array<{ name: string; weight: number; types: string[]; levels: number[]; count?: number }>
}

export interface ExamReadiness {
  exam_id: number
  exam: string
  code: string
  readiness: number
  covered: number
  total: number
  bar: number
  weak: string[]
  advice: string
}

export interface ExamAttemptPayload {
  attempt_id: number
  exam: { id: number; code: string; title: string; duration_minutes: number; passing_score: number; sections: ExamInfo['sections'] }
  state: string
  items: Array<{ id: number; code?: string; section: string; weight: number; question: Question }>
  answers: Record<string, { answer?: string; selected_option?: number | null; seconds?: number }>
  elapsed_seconds: number
  item_count: number
  missing: string[]
  report: ExamReport
}

export interface ExamReport {
  score: number
  passed: boolean
  verdict: string
  section_scores: Record<string, { score: number; weight: number; items: number }>
  item_results: Array<{
    id: number
    section: string
    correct: boolean
    score: number
    graded_by: string
    feedback?: string
    error_type?: string
    answered?: boolean
    type?: string
  }>
  strengths: string[]
  gaps: string[]
  feedback: string
  duration_seconds: number
}

export interface InterviewPayload {
  id: number
  level: string
  focus: string
  state: string
  turns: Array<{ role: string; content: string; question_id?: number | null; at?: string; score?: number; feedback?: string }>
  asked_count: number
  scores: Record<string, number>
  report: {
    engine: string
    level: string
    focus: string
    turns: number
    recommendation: string
    narrative: string
    per_turn: Array<{ question_id: number | null; score: number; error_type: string; words: number }>
  }
  started_at?: string
  finished_at?: string | null
}

export interface ProgressPayload {
  ml_engineer_score: number
  level_coverage: number
  dimensions: Record<string, number>
  weighted_dimension_score: number
  skills_assessed: number
  skills_total: number
  xp: number
  minutes_studied: number
  strong_skills: Array<{ code: string; name: string; score: number; level: number }>
  weak_skills: Array<{ code: string; name: string; score: number; level: number; success_rate?: number }>
  needs_review: Array<{ code: string; name: string; due: string; score: number }>
  current_level: { index: number; label: string }
}

export interface VelocityPayload {
  window_days: number
  series: Array<{ date: string; questions: number; correct: number; coding: number; minutes: number; xp: number }>
  questions_per_week: number
  coding_tasks_per_week: number
  projects_per_month: number
  study_minutes_total: number
  minutes_per_week: number
  xp_total: number
  xp_per_week: number
  skills_improved: number
  answer_accuracy: number | null
  days_active: number
}

export interface RoadmapLevel {
  index: number
  title: string
  goal: string
  topics: TopicSummary[]
  completion: number
  avg_score: number
  topic_count: number
}

export interface RoadmapPayload {
  levels: RoadmapLevel[]
  current_level: { index: number; label: string }
  goal: string
  target_role: string
  weak_skills: string[]
  recommendation: {
    topic?: { code: string; name: string; level: number; readiness: number; reasons: string[] }
    remediation?: { skill: string; name: string; score: number; why: string }
    candidates?: Array<TopicSummary & { readiness?: number }>
    mode_suggestion?: string
  }
  skipped?: Array<{ code: string; name: string; reason: string; score?: number }>
}

export interface AiConfig {
  ai_provider: string
  ai_available: boolean
  embedding_provider: string
  embedding_model: string
  vector_backend: string
  max_requests_per_day: number
  max_tokens_per_request: number
  cost_currency: string
  cost_per_1k_input: number
  cost_per_1k_output: number
  yandex_model: string
  yandex_flavour: string
  yandex_api_key_present?: boolean
  yandex_folder_id_present?: boolean
  openai_api_key_present?: boolean
  retrieval_top_k: number
  code_execution_enabled: boolean
  web_ingestion_enabled: boolean
  budget?: AiBudget
  health?: Record<string, unknown>
}

export interface AiBudget {
  requests_today: number
  requests_this_month: number
  tokens_today: number
  tokens_this_month: number
  cost_today: number
  cost_this_month: number
  currency: string
  max_requests_per_day: number
  requests_remaining_today: number
  cache_saved_requests: number
  limit_reached?: boolean
}

export interface AiUsageStats {
  budget: AiBudget
  by_method: Array<{ method: string; calls: number; cost: number; avg_tokens?: number }>
  by_day: Array<{ date: string; requests: number; cache_hits: number; cost: number }>
  recent_problems: Array<{ at: string; method: string; status: string; error: string }>
  cache?: { entries: number; total_hits: number }
  window_days?: number
}

export interface DashboardPayload {
  ml_engineer_score: number
  dimensions: Record<string, number>
  current_level: { index: number; label: string }
  today: { date: string; items: PlanItem[]; completed: number; total: number; rationale?: string }
  weak_skills: Array<{ code: string; name: string; score: number }>
  strong_skills: Array<{ code: string; name: string; score: number }>
  due_for_review: { total: number; items: Array<{ code: string; type: string; overdue_hours: number; item_id: number }> }
  current_course: { topic: string; name: string; readiness?: number }
  current_project: EnrollmentPayload | null
  streak: { current: number; longest: number; days_studied: number }
  velocity: VelocityPayload
  library: { indexed_documents: number }
  ai: { configured: boolean; provider: string } & Partial<AiBudget>
  recommendation: RoadmapPayload['recommendation']
  diagnostic_done: boolean
}

export interface NoteRow {
  id: number
  kind: string
  title: string
  body: string
  topic: string
  excerpt?: string
  citation?: Citation
  tags: string[]
  pinned: boolean
  created?: string
  updated?: string
}

export interface JournalRow {
  id: number
  date: string
  content: string
  mood: number
  minutes: number
  word_count: number
  topics: Array<{ code: string; name: string; level?: number; knowledge_score?: number }>
  gaps: Array<{ code: string; name: string; why?: string }>
  strengths: Array<{ code: string; name: string }>
  risk_flags: string[]
  analysis?: { summary?: string; engine?: string }
}

export interface MemoryPayload {
  summary: { categories: Record<string, number>; total_rows: number; recent_attempts: number }
  strengths: Array<{ key: string; value: string; at?: string }>
  weaknesses: Array<{ key: string; value: string; at?: string }>
  preferences: Array<{ key: string; value: string }>
  topics: Array<{ key: string; value: string; payload?: Record<string, unknown> }>
  projects: Array<{ key: string; value: string }>
  exams: Array<{ key: string; value: string }>
  mistakes: Array<{ key: string; value: string }>
  history: Array<{ key: string; value: string }>
  facts: Array<{ key: string; value: string }>
  risks: Array<{ key: string; value: string }>
}

export interface ReviewQueueItem {
  code: string
  name?: string
  type: string
  item_id: number
  overdue_hours: number
  ease?: number
  lapses?: number
  question?: Question | null
}

export interface LibraryStats {
  documents: number
  chunks: number
  words: number
  pages: number
  by_kind: Record<string, number>
  by_status: Record<string, number>
  vector_store: { name: string; count: number; ok?: boolean; error?: string }
  lexical_docs: number
  progress?: Record<string, { stage: string; percent: number }>
}

export interface SandboxCapabilities {
  enabled: boolean
  isolation: string
  timeout_seconds: number
  memory_mb: number
  numpy: boolean
  pandas: boolean
  sklearn: boolean
  language: string
  posix_limits: boolean
}

export interface SkillNode {
  id: string
  name: string
  level: number
  category: string
  topic?: string | null
  score: number
  confidence: number
}

export interface SkillGraphPayload {
  nodes: SkillNode[]
  links: Array<{ source: string; target: string; type: string }>
  levels: number[]
}

export interface LearningState {
  level_index: number
  level_label: string
  strengths: string[]
  weaknesses: string[]
  weak_detail: Array<{ code: string; score: number; confidence: number }>
  missing_prerequisites: string[]
  recent_mistakes: Array<{ question: string; error: string }>
  preferences: string[]
  current_topic: string
  velocity: VelocityPayload
  goal: string
  daily_minutes: number
  recommendation?: RoadmapPayload['recommendation']
  due?: { due_total: number; urgent: ReviewQueueItem[] }
}

export interface QuestionStats {
  bank_size: number
  by_type: Record<string, number>
  by_source: Record<string, number>
  your_attempts: number
  your_correct: number
  your_accuracy: number | null
  grounded_in_library: number
}

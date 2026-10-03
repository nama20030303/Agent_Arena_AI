export interface PlayerPublicState {
  level: number;
  rank: string;
  xp: number;
  xp_into_level: number;
  xp_for_next: number;
  next_rank: string | null;
  knowledge_power: number;
  streak: number;
  best_streak: number;
  streak_shields: number;
  chests_unopened: number;
  answers_total: number;
  answers_correct: number;
  bosses_defeated: number;
  onboarded: boolean;
  interests: string[];
  goal: string;
  intensity: string;
  selected_title: string;
  selected_frame: string;
}

export interface Fx {
  xp_gained?: number;
  xp_reasons?: { reason: string; amount: number }[];
  level_up?: { from: number; to: number; rank: string };
  chests_granted?: { id: number; source: string; world: string }[];
  achievements?: { code: string; name: string; glyph: string; xp: number }[];
  quests_completed?: { code: string; title: string }[];
  streak?: { streak: number; kind: string };
}

export interface Objective {
  node: string;
  name: string;
  short: string;
  world: string;
  world_name: string;
  color: string;
  progress: number;
  is_boss: boolean;
  status: string;
}

export interface Quest {
  code: string;
  title: string;
  target: number;
  progress: number;
  completed: boolean;
  xp: number;
  chest: boolean;
}

export interface ChestRef {
  id: number;
  source: string;
  world: string;
}

export interface WorldPower {
  code: string;
  name: string;
  color: string;
  glyph: string;
  power: number;
  mastered: number;
  total: number;
}

export interface GameState {
  state: PlayerPublicState;
  objective: Objective | null;
  quests: Quest[];
  chests: ChestRef[];
  worlds: WorldPower[];
  display_name: string;
  username: string;
}

export interface MapWorld {
  code: string;
  name: string;
  tagline: string;
  color: string;
  glyph: string;
}

export interface MapNode {
  code: string;
  world: string;
  name: string;
  short: string;
  difficulty: number;
  is_boss: boolean;
  x: number;
  y: number;
  status: "locked" | "available" | "active" | "mastered";
  progress: number;
}

export interface MapData {
  worlds: MapWorld[];
  nodes: MapNode[];
  edges: { from: string; to: string }[];
}

export interface NodeQuestion {
  id: number;
  type: "mcq" | "input" | "open";
  prompt: string;
  options: string[];
  difficulty: number;
  hints_total: number;
  hints_used: number;
  solved: boolean;
  attempts: number;
}

export interface NodeDetail {
  code: string;
  name: string;
  short: string;
  theory: string;
  difficulty: number;
  xp_reward: number;
  is_boss: boolean;
  world: { code: string; name: string; color: string; glyph: string };
  status: string;
  progress: number;
  questions: NodeQuestion[];
  prereqs: { code: string; name: string }[];
}

export interface AnswerResult {
  correct: boolean;
  already_solved: boolean;
  explanation: string;
  fx: Fx;
  node: { code?: string; progress?: number; mastered?: boolean; unlocked?: string[] };
  state: PlayerPublicState;
}

export interface ItemInfo {
  code: string;
  name: string;
  description: string;
  rarity: string;
  kind: string;
  world: string;
  glyph: string;
  count?: number;
}

export interface ChestOpenResult {
  rarity: string;
  item: ItemInfo;
  duplicate: boolean;
  xp_bonus: number;
  fx: Fx;
  state: PlayerPublicState;
}

export interface Achievement {
  code: string;
  name: string;
  description: string;
  glyph: string;
  xp: number;
  unlocked: boolean;
  unlocked_at: string | null;
  secret: boolean;
}

export interface TimelineEntry {
  type: string;
  label: string;
  detail: string;
  at: string;
}

export interface ProfileData {
  state: PlayerPublicState;
  display_name: string;
  username: string;
  worlds: WorldPower[];
  achievements: Achievement[];
  timeline: TimelineEntry[];
}

export interface ChatMessage {
  role: "user" | "teacher";
  text: string;
  mode: string;
  node: string;
  via_ai: boolean;
  at?: string;
}

export interface LeaderboardEntry {
  username: string;
  display_name: string;
  xp: number;
  level: number;
  rank: string;
  knowledge_power: number;
  me: boolean;
}

export const RARITY_RU: Record<string, string> = {
  common: "Обычный",
  uncommon: "Необычный",
  rare: "Редкий",
  epic: "Эпический",
  legendary: "Легендарный",
  mythic: "Мифический",
};

export const KIND_RU: Record<string, string> = {
  collectible: "Артефакт",
  title: "Титул",
  frame: "Рамка",
  theme: "Тема",
  badge: "Значок",
};

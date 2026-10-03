interface Props {
  level: number;
  progress: number; // 0..1 inside current level
  size?: number;
  color?: string;
}

export default function LevelRing({ level, progress, size = 64, color = "#7dd3fc" }: Props) {
  const stroke = Math.max(3, size / 18);
  const r = (size - stroke) / 2;
  const c = 2 * Math.PI * r;
  const clamped = Math.max(0, Math.min(1, progress));
  return (
    <div className="level-ring" style={{ width: size, height: size }}>
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
        <circle cx={size / 2} cy={size / 2} r={r} stroke="rgba(255,255,255,0.1)" strokeWidth={stroke} fill="none" />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={r}
          stroke={color}
          strokeWidth={stroke}
          fill="none"
          strokeLinecap="round"
          strokeDasharray={c}
          strokeDashoffset={c * (1 - clamped)}
          transform={`rotate(-90 ${size / 2} ${size / 2})`}
          style={{ transition: "stroke-dashoffset 0.8s cubic-bezier(.22,1,.36,1)", filter: `drop-shadow(0 0 ${stroke}px ${color})` }}
        />
      </svg>
      <div className="level-ring-value" style={{ fontSize: size * 0.32 }}>
        {level}
      </div>
    </div>
  );
}

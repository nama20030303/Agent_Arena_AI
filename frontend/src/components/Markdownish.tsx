import React from "react";

/** Minimal safe renderer for seeded theory text: paragraphs, ``` code blocks, - lists, `inline code`. */
export default function Markdownish({ text }: { text: string }) {
  const blocks: React.ReactNode[] = [];
  const parts = (text || "").split(/```/);
  parts.forEach((part, i) => {
    if (i % 2 === 1) {
      blocks.push(
        <pre key={`code-${i}`} className="code-block">
          <code>{part.replace(/^\w*\n/, "").trimEnd()}</code>
        </pre>
      );
      return;
    }
    part
      .split(/\n\s*\n/)
      .map((p) => p.trim())
      .filter(Boolean)
      .forEach((para, j) => {
        const lines = para.split("\n");
        const isList = lines.every((l) => /^[-•*]\s+/.test(l.trim()));
        if (isList) {
          blocks.push(
            <ul key={`l-${i}-${j}`}>
              {lines.map((l, k) => (
                <li key={k}>{inline(l.replace(/^[-•*]\s+/, ""))}</li>
              ))}
            </ul>
          );
        } else {
          blocks.push(<p key={`p-${i}-${j}`}>{inline(para)}</p>);
        }
      });
  });
  return <div className="mdish">{blocks}</div>;
}

function inline(text: string): React.ReactNode[] {
  const out: React.ReactNode[] = [];
  const segments = text.split(/(`[^`]+`|\*\*[^*]+\*\*)/g);
  segments.forEach((seg, i) => {
    if (seg.startsWith("`") && seg.endsWith("`")) {
      out.push(<code key={i}>{seg.slice(1, -1)}</code>);
    } else if (seg.startsWith("**") && seg.endsWith("**")) {
      out.push(<strong key={i}>{seg.slice(2, -2)}</strong>);
    } else if (seg) {
      out.push(<React.Fragment key={i}>{seg}</React.Fragment>);
    }
  });
  return out;
}

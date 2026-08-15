/**
 * RoutingTimeline — Feature Round 3, Feature 3.
 *
 * Chronological replay panel showing every routing decision made
 * in this session: message excerpt → target node → confidence → reasoning.
 */
"use client";

import React, { useEffect, useState } from "react";
import { getRoutingLog, RoutingLogEntry } from "../../lib/api-client";
import { ArrowRight, Clock, Cpu, X } from "lucide-react";

interface RoutingTimelineProps {
  sessionId: string;
  open: boolean;
  onClose: () => void;
}

export function RoutingTimeline({ sessionId, open, onClose }: RoutingTimelineProps) {
  const [entries, setEntries] = useState<RoutingLogEntry[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!open || !sessionId) return;
    setLoading(true);
    getRoutingLog(sessionId)
      .then(setEntries)
      .catch((e) => console.error("Failed to load routing log", e))
      .finally(() => setLoading(false));
  }, [open, sessionId]);

  if (!open) return null;

  const confColor = (c: number) => {
    const pct = Math.round(c * 100);
    if (pct >= 85) return "bg-emerald-500/15 text-emerald-400 border-emerald-500/30";
    if (pct >= 60) return "bg-amber-500/15 text-amber-400 border-amber-500/30";
    return "bg-rose-500/15 text-rose-400 border-rose-500/30";
  };

  return (
    <div className="fixed inset-0 z-50 flex justify-end" onClick={onClose}>
      <div className="absolute inset-0 bg-black/40 backdrop-blur-sm" />
      <div
        className="relative w-full max-w-md bg-slate-950 border-l border-slate-800 shadow-2xl flex flex-col animate-in slide-in-from-right duration-200"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-slate-800">
          <div className="flex items-center gap-2 text-sm font-semibold text-slate-200">
            <Clock size={15} className="text-primary" />
            Routing Decision Timeline
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded text-slate-400 hover:text-slate-200 hover:bg-slate-800 transition-colors"
          >
            <X size={16} />
          </button>
        </div>

        {/* Entries */}
        <div className="flex-1 overflow-y-auto p-4 space-y-3">
          {loading ? (
            <div className="flex items-center justify-center h-32 text-sm text-slate-500">
              <Cpu size={14} className="animate-spin mr-2" />
              Loading timeline…
            </div>
          ) : entries.length === 0 ? (
            <div className="text-center text-sm text-slate-500 py-12">
              <Clock size={24} className="mx-auto mb-3 opacity-40" />
              <p>No routing decisions yet.</p>
              <p className="text-xs text-slate-600 mt-1">
                Send some messages across different topics to see the timeline.
              </p>
            </div>
          ) : (
            entries.map((entry, idx) => {
              const pct = Math.round((entry.confidence ?? 0.9) * 100);
              const ts = entry.timestamp
                ? new Date(entry.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })
                : "";
              return (
                <div
                  key={idx}
                  className="group rounded-xl border border-slate-800 bg-slate-900/60 p-3 hover:border-slate-700 transition-colors"
                >
                  {/* Timestamp + model */}
                  <div className="flex items-center justify-between text-[10px] text-slate-500 mb-2">
                    <span className="flex items-center gap-1">
                      <Clock size={10} />
                      {ts}
                    </span>
                    {entry.router_model && (
                      <span className="text-[9px] bg-slate-800 rounded px-1.5 py-0.5 font-mono text-slate-400">
                        {entry.router_model.split("/").pop()?.replace(":free", "")}
                      </span>
                    )}
                  </div>

                  {/* Message → Node */}
                  <div className="flex items-start gap-2 mb-2">
                    <div className="flex-1 min-w-0">
                      <p className="text-xs text-slate-300 line-clamp-2 leading-relaxed">
                        &ldquo;{entry.message_excerpt}&rdquo;
                      </p>
                    </div>
                    <ArrowRight size={13} className="text-slate-600 mt-0.5 flex-shrink-0" />
                    <div className="flex-shrink-0">
                      <span className="text-xs font-medium text-primary bg-primary/10 rounded px-1.5 py-0.5 border border-primary/20">
                        {entry.node_title}
                      </span>
                    </div>
                  </div>

                  {/* Confidence + reasoning */}
                  <div className="flex items-center gap-2">
                    <span
                      className={`text-[10px] font-bold px-1.5 py-0.5 rounded border ${confColor(entry.confidence)}`}
                    >
                      {pct}%
                    </span>
                    {entry.latency_ms != null && entry.latency_ms > 0 && (
                      <span className="text-[10px] text-slate-500 font-mono">
                        {entry.latency_ms}ms
                      </span>
                    )}
                    <span className="text-[10px] text-slate-500 truncate flex-1">
                      {entry.reasoning}
                    </span>
                  </div>
                </div>
              );
            })
          )}
        </div>

        {/* Footer */}
        <div className="border-t border-slate-800 px-5 py-3 text-[10px] text-slate-600 text-center">
          {entries.length} routing decision{entries.length !== 1 ? "s" : ""} recorded this session
        </div>
      </div>
    </div>
  );
}

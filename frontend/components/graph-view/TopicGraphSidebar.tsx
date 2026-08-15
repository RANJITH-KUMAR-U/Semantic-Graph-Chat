/**
 * TopicGraphSidebar — single-column vertical tree with SVG connecting lines,
 * topic search bar (Feature 5), topic merge suggestion chips (Feature 3),
 * relatedness lines (Feature R3-F2), and routing timeline toggle (Feature R3-F3).
 */
"use client";

import React, { useRef, useEffect, useState } from "react";
import { TopicNode, searchMessages, SearchResult, mergeNodes } from "../../lib/api-client";
import { RoutingTimeline } from "./RoutingTimeline";
import {
  Network,
  GitBranch,
  Sparkles,
  MessageSquare,
  Zap,
  Clock,
  Search,
  X,
  Merge,
  History,
  FileText,
} from "lucide-react";

interface TopicGraphSidebarProps {
  sessionId: string | null;
  nodes: TopicNode[];
  activeNodeId: string | null;
  onForceRoute: (nodeId: string) => void;
  onSelectSearchResult?: (nodeId: string, messageId: string) => void;
  onNodesUpdated?: () => void;
}

const TOPIC_HUES = [239, 262, 197, 158, 43, 340, 24, 283, 174, 56];

function getHue(idx: number): number {
  return TOPIC_HUES[idx % TOPIC_HUES.length];
}

function relativeTime(iso: string): string {
  const diff = Date.now() - new Date(iso).getTime();
  const m = Math.floor(diff / 60000);
  if (m < 1) return "just now";
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h ago`;
  return `${Math.floor(h / 24)}d ago`;
}

export function TopicGraphSidebar({
  sessionId,
  nodes,
  activeNodeId,
  onForceRoute,
  onSelectSearchResult,
  onNodesUpdated,
}: TopicGraphSidebarProps) {
  const activeRef = useRef<HTMLButtonElement | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [dismissedMerges, setDismissedMerges] = useState<Record<string, boolean>>({});
  // Feature R3-F3: routing timeline panel
  const [showTimeline, setShowTimeline] = useState(false);

  // Refs for computing relatedness line positions
  const nodeCardRefs = useRef<Record<string, HTMLButtonElement | null>>({});
  const graphContainerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (activeRef.current) {
      activeRef.current.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
  }, [activeNodeId]);

  // Feature 5: Search messages across topics
  useEffect(() => {
    if (!searchQuery.trim() || !sessionId) {
      setSearchResults([]);
      setIsSearching(false);
      return;
    }
    const timer = setTimeout(async () => {
      setIsSearching(true);
      try {
        const results = await searchMessages(sessionId, searchQuery.trim());
        setSearchResults(results);
      } catch (e) {
        console.error("Search failed", e);
      } finally {
        setIsSearching(false);
      }
    }, 250);

    return () => clearTimeout(timer);
  }, [searchQuery, sessionId]);

  // Feature 3: One-click topic merge
  const handleMerge = async (sourceId: string, targetId: string) => {
    if (!sessionId) return;
    try {
      await mergeNodes(sessionId, sourceId, targetId);
      onNodesUpdated?.();
    } catch (e) {
      console.error("Merge failed", e);
    }
  };

  const handleDismissMerge = (nodeId: string) => {
    setDismissedMerges((prev) => ({ ...prev, [nodeId]: true }));
  };

  // Feature R3-F2: Build unique relatedness pairs for SVG lines
  const relatednessPairs: Array<{ from: string; to: string }> = [];
  const seenPairs = new Set<string>();
  for (const node of nodes) {
    if (node.related_node_ids) {
      for (const relId of node.related_node_ids) {
        const pairKey = [node.node_id, relId].sort().join("--");
        if (!seenPairs.has(pairKey) && nodes.some((n) => n.node_id === relId)) {
          seenPairs.add(pairKey);
          relatednessPairs.push({ from: node.node_id, to: relId });
        }
      }
    }
  }

  return (
    <div className="ts-root">
      {/* ─── Header ─────────────────────────────────────── */}
      <div className="ts-header">
        <div className="ts-header__left">
          <Network size={15} className="ts-header__icon" />
          <span className="ts-header__title">Topic Graph</span>
        </div>
        <div className="flex items-center gap-1.5">
          {/* Feature R3-F3: Timeline toggle */}
          <button
            onClick={() => setShowTimeline(true)}
            className="p-1 rounded text-slate-500 hover:text-primary hover:bg-primary/10 transition-colors"
            title="View routing decision timeline"
          >
            <History size={14} />
          </button>
          <span className="ts-header__badge" title="Total active topic nodes in this session">
            {nodes.length} {nodes.length === 1 ? "node" : "nodes"}
          </span>
        </div>
      </div>

      {/* Feature R3-F3: Routing Timeline Panel */}
      {sessionId && (
        <RoutingTimeline
          sessionId={sessionId}
          open={showTimeline}
          onClose={() => setShowTimeline(false)}
        />
      )}

      {/* Feature 5: Topic Search Input */}
      <div className="px-3 py-2 border-b border-border bg-muted/20">
        <div className="relative flex items-center">
          <Search size={13} className="absolute left-2.5 text-muted-foreground" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search topic messages..."
            className="w-full bg-slate-950 border border-slate-800 rounded-lg pl-8 pr-7 py-1 text-xs text-foreground placeholder:text-muted-foreground focus:outline-none focus:border-primary/50"
          />
          {searchQuery && (
            <button
              onClick={() => setSearchQuery("")}
              className="absolute right-2 text-muted-foreground hover:text-foreground"
            >
              <X size={13} />
            </button>
          )}
        </div>
      </div>

      {/* Search results overlay */}
      {searchQuery.trim() ? (
        <div className="flex-1 overflow-y-auto p-2 space-y-1">
          <div className="text-[10px] font-semibold uppercase text-muted-foreground px-2 py-1">
            {isSearching ? "Searching..." : `Results (${searchResults.length})`}
          </div>
          {searchResults.length === 0 && !isSearching ? (
            <p className="text-xs text-muted-foreground p-3 text-center">No matching messages found.</p>
          ) : (
            searchResults.map((res) => (
              <button
                key={res.message_id}
                onClick={() => onSelectSearchResult?.(res.node_id, res.message_id)}
                className="w-full text-left p-2 rounded-lg bg-card hover:bg-muted/50 border border-border transition-colors group"
              >
                <div className="flex items-center justify-between text-[11px] font-medium text-primary mb-1">
                  <span className="truncate">{res.node_title}</span>
                  <span className="text-[9px] uppercase text-muted-foreground">{res.role}</span>
                </div>
                <p className="text-xs text-slate-300 line-clamp-2 leading-relaxed">{res.content}</p>
              </button>
            ))
          )}
        </div>
      ) : (
        /* ─── Regular Graph View ──────────────────────────── */
        <div className="ts-graph" ref={graphContainerRef}>
          {nodes.length === 0 ? (
            <div className="ts-empty">
              <div className="ts-empty__orb">
                <Network size={28} />
              </div>
              <p className="ts-empty__title">No topics yet</p>
              <p className="ts-empty__sub">
                Send your first message — the AI will automatically create a topic
                node and start routing.
              </p>
              <div className="ts-empty__hints">
                <span className="ts-hint-chip">Try &quot;explain binary trees&quot;</span>
                <span className="ts-hint-chip">Try &quot;write a React hook&quot;</span>
                <span className="ts-hint-chip">Try &quot;what is dark matter?&quot;</span>
              </div>
            </div>
          ) : (
            <div className="ts-tree">
              {/* Session root hub */}
              <div className="ts-root-hub" title="Root session context container">
                <Sparkles size={13} className="ts-root-hub__icon" />
                <span>Session Root</span>
              </div>

              <div className="ts-connector ts-connector--root" />

              {nodes.map((n, idx) => {
                const isActive = n.node_id === activeNodeId;
                const hue = getHue(idx);
                const isLast = idx === nodes.length - 1;
                const targetNode = nodes.find((x) => x.node_id === n.possible_duplicate_of);
                const showMergeSuggestion =
                  n.possible_duplicate_of && targetNode && !dismissedMerges[n.node_id];

                return (
                  <div key={n.node_id} className="ts-node-wrap">
                    <div className="ts-rail">
                      <div
                        className="ts-rail__dot"
                        style={{
                          background: `hsl(${hue} 80% ${isActive ? "65%" : "45%"})`,
                          boxShadow: isActive ? `0 0 8px hsl(${hue} 80% 65% / 0.6)` : "none",
                        }}
                      />
                      {!isLast && <div className="ts-rail__line" />}
                    </div>

                    <div className="flex-1 min-w-0">
                      <button
                        ref={(el) => {
                          nodeCardRefs.current[n.node_id] = el;
                          if (isActive) {
                            (activeRef as React.MutableRefObject<HTMLButtonElement | null>).current = el;
                          }
                        }}
                        className={`ts-card${isActive ? " ts-card--active" : ""}`}
                        style={{ "--card-hue": hue } as React.CSSProperties}
                        onClick={() => onForceRoute(n.node_id)}
                        title={`Topic Node: ${n.title}`}
                      >
                        <div
                          className="ts-card__accent"
                          style={{ background: `hsl(${hue} 80% ${isActive ? "60%" : "40%"})` }}
                        />

                        <div
                          className="ts-card__icon"
                          style={{
                            background: `hsl(${hue} 80% ${isActive ? "60%" : "35%"} / ${isActive ? "0.25" : "0.15"})`,
                            color: `hsl(${hue} 80% ${isActive ? "70%" : "55%"})`,
                          }}
                        >
                          <GitBranch size={14} />
                        </div>

                        <div className="ts-card__body">
                          <div className="ts-card__title-row">
                            <span className="ts-card__title">{n.title || "Untitled Topic"}</span>
                            {isActive && (
                              <span
                                className="ts-card__active-dot"
                                style={{ background: `hsl(${hue} 80% 60%)` }}
                                title="Active context node"
                              />
                            )}
                          </div>
                          <div className="ts-card__meta">
                            {n.message_count !== undefined && n.message_count > 0 && (
                              <span className="ts-card__meta-item" title="Total messages stored in this node">
                                <MessageSquare size={10} />
                                {n.message_count} msgs
                              </span>
                            )}
                            {n.document_chunk_count !== undefined && n.document_chunk_count > 0 && (
                              <span className="ts-card__meta-item text-blue-400 font-semibold" title={`${n.document_chunk_count} document chunk(s) indexed for RAG`}>
                                <FileText size={10} />
                                {n.document_chunk_count} docs
                              </span>
                            )}
                            {n.last_active_at && (
                              <span className="ts-card__meta-item" title="Time of last message in this topic">
                                <Clock size={10} />
                                {relativeTime(n.last_active_at)}
                              </span>
                            )}
                            {/* Feature R3-F2: related nodes count hint */}
                            {n.related_node_ids && n.related_node_ids.length > 0 && (
                              <span
                                className="ts-card__meta-item text-violet-400"
                                title={`Related to ${n.related_node_ids.length} other topic(s)`}
                              >
                                <Network size={10} />
                                {n.related_node_ids.length} related
                              </span>
                            )}
                          </div>
                        </div>

                        {isActive && (
                          <span title="Active routing context">
                            <Zap
                              size={11}
                              className="ts-card__zap"
                              style={{ color: `hsl(${hue} 80% 65%)` }}
                            />
                          </span>
                        )}

                        {/* Bug 3 fix: Removed unlabeled ordinal badge — it carried no semantic meaning. */}
                      </button>

                      {/* Feature 3: Topic merge suggestion chip */}
                      {showMergeSuggestion && (
                        <div className="mt-1 mb-2 p-2 bg-amber-500/10 border border-amber-500/30 rounded-lg text-xs space-y-1.5 animate-in fade-in slide-in-from-top-1 duration-150">
                          <div className="flex items-center gap-1 font-medium text-amber-400">
                            <Merge size={12} />
                            <span>Similar to &quot;{targetNode.title}&quot; — merge?</span>
                          </div>
                          <div className="flex items-center gap-2 pt-0.5">
                            <button
                              onClick={() => handleMerge(n.node_id, targetNode.node_id)}
                              className="px-2 py-0.5 bg-amber-500 text-slate-950 font-semibold text-[10px] rounded hover:bg-amber-400 transition-colors"
                            >
                              Merge topics
                            </button>
                            <button
                              onClick={() => handleDismissMerge(n.node_id)}
                              className="text-[10px] text-amber-400/80 hover:text-amber-200"
                            >
                              Keep separate
                            </button>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                );
              })}

              {/* Feature R3-F2: Relatedness lines between nodes */}
              {relatednessPairs.length > 0 && (
                <div className="mt-3 px-2 space-y-1">
                  {relatednessPairs.map(({ from, to }) => {
                    const fromNode = nodes.find((n) => n.node_id === from);
                    const toNode = nodes.find((n) => n.node_id === to);
                    if (!fromNode || !toNode) return null;
                    return (
                      <div
                        key={`${from}--${to}`}
                        className="flex items-center gap-1.5 text-[10px] text-violet-400/70 py-0.5"
                        title={`"${fromNode.title}" and "${toNode.title}" are topically related`}
                      >
                        <div className="flex-1 border-t border-dashed border-violet-500/30" />
                        <Network size={9} className="text-violet-400/50 flex-shrink-0" />
                        <span className="truncate max-w-[70px]">{fromNode.title}</span>
                        <span className="text-violet-500/40">⟷</span>
                        <span className="truncate max-w-[70px]">{toNode.title}</span>
                        <div className="flex-1 border-t border-dashed border-violet-500/30" />
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

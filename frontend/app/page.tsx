"use client";

import React, { useEffect, useState, useCallback, useRef } from "react";
import { TopicGraphSidebar } from "../components/graph-view/TopicGraphSidebar";
import { ChatWindow } from "../components/chat/ChatWindow";
import { RouterModelSelector } from "../components/ui/RouterModelSelector";
import { useChatSession } from "../hooks/useChatSession";
import { createSession, reassignMessage, getSessionRecap, exportSession, SessionRecap } from "../lib/api-client";
import { Bot, AlertCircle, ChevronDown, ChevronUp, Layers, Zap, Info, Loader2, Download } from "lucide-react";

export default function Home() {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [summaryExpanded, setSummaryExpanded] = useState(false);
  const [showTokenDetails, setShowTokenDetails] = useState(false);
  const [recapBanner, setRecapBanner] = useState<{ show: boolean; activeTitle: string; count: number } | null>(null);
  const [targetMessageId, setTargetMessageId] = useState<string | null>(null);

  // 1b: Resizable sidebar
  const SIDEBAR_MIN = 200;
  const SIDEBAR_MAX = 520;
  const CHAT_MIN = 380;
  const [sidebarWidth, setSidebarWidth] = useState(300);
  const isDragging = useRef(false);
  const dragStartX = useRef(0);
  const dragStartWidth = useRef(0);

  const handleResizeMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    isDragging.current = true;
    dragStartX.current = e.clientX;
    dragStartWidth.current = sidebarWidth;
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
  }, [sidebarWidth]);

  useEffect(() => {
    const onMove = (e: MouseEvent) => {
      if (!isDragging.current) return;
      const delta = e.clientX - dragStartX.current;
      const containerWidth = window.innerWidth;
      const newWidth = Math.min(
        SIDEBAR_MAX,
        Math.max(SIDEBAR_MIN, dragStartWidth.current + delta)
      );
      // Enforce min chat width
      const chatWidth = containerWidth - newWidth - 4; // 4px divider
      if (chatWidth < CHAT_MIN) return;
      setSidebarWidth(newWidth);
    };
    const onUp = () => {
      if (!isDragging.current) return;
      isDragging.current = false;
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
    };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    return () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
  }, []);

  useEffect(() => {
    const stored = localStorage.getItem("chat_session_id");
    if (stored) {
      setSessionId(stored);
    } else {
      createSession()
        .then((data) => {
          setSessionId(data.session_id);
          localStorage.setItem("chat_session_id", data.session_id);
        })
        .catch(console.error);
    }
  }, []);

  const {
    nodes,
    activeNodeId,
    messages,
    globalSummary,
    isSummarizing,
    tokensUsed,
    baselineTokens,
    isRouting,
    isGenerating,
    error,
    sendMessage,
    stopGeneration,
    selectNode,
    fetchNodes,
    // Feature 5: router model state
    routerModel,
    setRouterModel,
  } = useChatSession(sessionId);

  // Feature 6: Check for session recap on return load
  useEffect(() => {
    if (!sessionId) return;
    getSessionRecap(sessionId)
      .then((recap: SessionRecap) => {
        if (recap.has_history && recap.node_count > 0 && recap.active_node_title) {
          setRecapBanner({
            show: true,
            activeTitle: recap.active_node_title,
            count: recap.node_count,
          });
        }
      })
      .catch(console.error);
  }, [sessionId]);

  const handleSelectNode = async (nodeId: string) => {
    if (!sessionId) return;
    try {
      await selectNode(nodeId);
    } catch (e) {
      console.error("Failed to select node", e);
    }
  };

  // Feature 2: Handle manual message reassignment
  const handleReassignMessage = useCallback(
    async (messageId: string, targetNodeId?: string, newTitle?: string) => {
      if (!sessionId) return;
      try {
        const res = await reassignMessage(sessionId, messageId, targetNodeId, newTitle);
        await fetchNodes();
        if (res.target_node_id) {
          await selectNode(res.target_node_id);
        }
      } catch (e) {
        console.error("Failed to reassign message", e);
      }
    },
    [sessionId, fetchNodes, selectNode]
  );

  // Feature 5: Handle jumping from search result
  const handleSelectSearchResult = async (nodeId: string, messageId: string) => {
    await handleSelectNode(nodeId);
    setTargetMessageId(messageId);
    setTimeout(() => setTargetMessageId(null), 3000);
  };

  // Feature 1: Handle jump-to-node from cross-ref chip
  const handleJumpToNode = useCallback(
    async (nodeId: string) => {
      await handleSelectNode(nodeId);
    },
    [sessionId, selectNode]
  );

  // Feature 4: Handle session export
  const handleExport = useCallback(async () => {
    if (!sessionId) return;
    try {
      await exportSession(sessionId);
    } catch (e) {
      console.error("Failed to export session", e);
    }
  }, [sessionId]);

  // Feature 4: Calculate Token Savings %
  const savingsPct =
    baselineTokens > 0
      ? Math.max(0, Math.round(((baselineTokens - tokensUsed) / baselineTokens) * 100))
      : 0;

  const activeNodeTitle =
    nodes.find((n) => n.node_id === activeNodeId)?.title ?? null;

  if (!sessionId) {
    return (
      <div className="flex h-screen items-center justify-center bg-background">
        <div className="flex flex-col items-center gap-4 text-muted-foreground">
          <div className="app-logo-spin">
            <Layers size={32} />
          </div>
          <span className="text-sm font-medium">Initializing Semantic Session…</span>
        </div>
      </div>
    );
  }

  return (
    <div className="app-shell">
      {/* ── Top header ─────────────────────────────────── */}
      <header className="app-header">
        <div className="app-header__brand">
          <div className="app-header__logo">
            <Bot size={17} />
          </div>
          <div>
            <h1 className="app-header__title">Semantic Graph Chat</h1>
            <p className="app-header__sub">Isolated AI memory per topic</p>
          </div>
        </div>

        {/* Feature 0: Global summary button with live updates & loading state */}
        <button
          className="app-header__summary-btn"
          onClick={() => setSummaryExpanded((v) => !v)}
          title={summaryExpanded ? "Collapse summary" : "Expand summary"}
        >
          <span className="app-header__summary-label flex items-center gap-1">
            Global Context
            {isSummarizing && <Loader2 size={10} className="spin text-primary ml-0.5" />}
          </span>
          <span className="app-header__summary-text">
            {isSummarizing
              ? "Summarizing session activity across topics…"
              : globalSummary || "No context established yet"}
          </span>
          {summaryExpanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
        </button>

        {/* Feature R3-F5: Router Model Playground Selector */}
        <RouterModelSelector
          selectedModel={routerModel}
          onSelectModel={setRouterModel}
        />

        {/* Feature 4: Token Savings Badge */}
        <div className="relative">
          <button
            onClick={() => setShowTokenDetails((v) => !v)}
            className="flex items-center gap-1.5 px-2.5 py-1 bg-emerald-500/10 border border-emerald-500/30 rounded-lg text-emerald-400 text-xs font-semibold hover:bg-emerald-500/20 transition-colors"
            title="Click for token savings breakdown"
          >
            <Zap size={13} />
            <span>TOKENS SAVED {savingsPct}%</span>
          </button>

          {showTokenDetails && (
            <div className="absolute right-0 top-full mt-2 z-50 w-72 p-3 bg-slate-900 border border-slate-700 rounded-xl shadow-2xl text-xs space-y-2 animate-in fade-in zoom-in-95 duration-150">
              <div className="font-bold text-slate-200 flex items-center justify-between">
                <span>Token Efficiency Dashboard</span>
                <span className="text-emerald-400 font-extrabold">{savingsPct}% Saved</span>
              </div>
              <div className="space-y-1.5 text-slate-300 text-[11px] pt-1">
                <div className="flex justify-between border-b border-slate-800 pb-1">
                  <span className="text-slate-400">Actual Tokens Used:</span>
                  <span className="font-mono font-semibold text-emerald-400">{tokensUsed.toLocaleString()}</span>
                </div>
                <div className="flex justify-between border-b border-slate-800 pb-1">
                  <span className="text-slate-400">Linear Baseline Estimate:</span>
                  <span className="font-mono font-semibold text-rose-400">{baselineTokens.toLocaleString()}</span>
                </div>
                <p className="text-[10px] text-slate-400 pt-1 leading-normal">
                  Semantic Graph Chat isolates memory per topic. Linear chat sends all previous turns on every message.
                </p>
              </div>
            </div>
          )}
        </div>

        {/* Feature R3-F4: Export button */}
        <button
          onClick={handleExport}
          className="flex items-center gap-1.5 px-2.5 py-1 bg-slate-800 border border-slate-700 rounded-lg text-slate-300 text-xs font-semibold hover:bg-slate-700 hover:text-white transition-colors"
          title="Export session as Markdown"
        >
          <Download size={13} />
          <span>Export</span>
        </button>

        <div className="app-header__session">
          <span className="app-header__session-label">Session</span>
          <span className="app-header__session-id">{sessionId.split("-")[0]}</span>
        </div>
      </header>

      {/* Expanded global summary */}
      {summaryExpanded && (
        <div className="app-summary-panel">
          <p className="app-summary-panel__text">
            {isSummarizing ? "Refreshing global context summary…" : globalSummary || "No global context summary available yet. Send messages across topics to generate."}
          </p>
        </div>
      )}

      {/* Error bar */}
      {error && (
        <div className="app-error">
          <AlertCircle size={14} />
          <span>{error}</span>
        </div>
      )}

      {/* ── Main layout ────────────────────────────────── */}
      <main className="app-main">
        {/* Sidebar */}
        <aside className="app-sidebar" style={{ width: sidebarWidth, minWidth: SIDEBAR_MIN, maxWidth: SIDEBAR_MAX }}>
          <TopicGraphSidebar
            sessionId={sessionId}
            nodes={nodes}
            activeNodeId={activeNodeId}
            onForceRoute={handleSelectNode}
            onSelectSearchResult={handleSelectSearchResult}
            onNodesUpdated={fetchNodes}
          />
        </aside>

        {/* Drag handle */}
        <div
          className="app-resize-handle"
          onMouseDown={handleResizeMouseDown}
          title="Drag to resize sidebar"
        />

        {/* Chat area */}
        <div className="app-chat">
          <ChatWindow
            messages={messages}
            isRouting={isRouting}
            isGenerating={isGenerating}
            activeNodeTitle={activeNodeTitle}
            availableNodes={nodes}
            recapBanner={recapBanner}
            onDismissRecap={() => setRecapBanner(null)}
            onSendMessage={sendMessage}
            onStopGeneration={stopGeneration}
            onReassignMessage={handleReassignMessage}
            onJumpToNode={handleJumpToNode}
            targetMessageId={targetMessageId}
            sessionId={sessionId}
            onUploadComplete={fetchNodes}
          />
        </div>
      </main>
    </div>
  );
}

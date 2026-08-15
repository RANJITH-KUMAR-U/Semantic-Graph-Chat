/**
 * ChatWindow — full-featured chat panel.
 */
"use client";

import React, { useRef, useEffect, useState, useCallback } from "react";
import { MessageBubble } from "./MessageBubble";
import { Message } from "../../hooks/useChatSession";
import { TopicNode } from "../../lib/api-client";
import {
  Send,
  Square,
  Loader2,
  GitBranch,
  Lightbulb,
  Cpu,
  Globe,
  Code2,
  BookOpen,
  FlaskConical,
  X,
} from "lucide-react";

import { FileUpload } from "./FileUpload";

const SUGGESTED_PROMPTS = [
  { icon: Code2,        text: "Explain how binary search trees work", category: "CS" },
  { icon: Cpu,          text: "What is transformer architecture in AI?", category: "AI" },
  { icon: Globe,        text: "Describe the formation of black holes", category: "Space" },
  { icon: BookOpen,     text: "Summarize the key ideas of stoicism", category: "Philosophy" },
  { icon: FlaskConical, text: "How does CRISPR gene editing work?", category: "Biology" },
];

interface ChatWindowProps {
  messages: Message[];
  isRouting: boolean;
  isGenerating?: boolean;
  activeNodeTitle?: string | null;
  availableNodes?: TopicNode[];
  recapBanner?: { show: boolean; activeTitle: string; count: number } | null;
  onDismissRecap?: () => void;
  onSendMessage: (msg: string) => void;
  onStopGeneration?: () => void;
  onReassignMessage?: (messageId: string, targetNodeId?: string, newTitle?: string) => void;
  onJumpToNode?: (nodeId: string) => void;
  targetMessageId?: string | null;
  sessionId?: string | null;
  onUploadComplete?: () => void;
}

export function ChatWindow({
  messages,
  isRouting,
  isGenerating = false,
  activeNodeTitle,
  availableNodes = [],
  recapBanner,
  onDismissRecap,
  onSendMessage,
  onStopGeneration,
  onReassignMessage,
  onJumpToNode,
  targetMessageId,
  sessionId = null,
  onUploadComplete,
}: ChatWindowProps) {
  const [input, setInput] = useState("");
  const endRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const isBusy = isRouting || isGenerating;
  const charLimit = 4000;

  // Auto-scroll to target message or bottom
  useEffect(() => {
    if (targetMessageId) {
      const el = document.getElementById(`msg-${targetMessageId}`);
      if (el) {
        el.scrollIntoView({ behavior: "smooth", block: "center" });
        return;
      }
    }
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isRouting, isGenerating, targetMessageId]);

  // Auto-grow textarea
  useEffect(() => {
    const ta = textareaRef.current;
    if (!ta) return;
    ta.style.height = "auto";
    const maxH = parseInt(getComputedStyle(ta).lineHeight) * 6 + 32;
    ta.style.height = Math.min(ta.scrollHeight, maxH) + "px";
  }, [input]);

  const handleSubmit = useCallback(
    (e: React.FormEvent) => {
      e.preventDefault();
      const trimmed = input.trim();
      if (!trimmed || isBusy) return;
      onSendMessage(trimmed);
      setInput("");
      if (textareaRef.current) textareaRef.current.style.height = "auto";
    },
    [input, isBusy, onSendMessage]
  );

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        handleSubmit(e as unknown as React.FormEvent);
      }
    },
    [handleSubmit]
  );

  const handleSuggest = (text: string) => {
    if (isBusy) return;
    onSendMessage(text);
  };

  const streamingMsgId =
    isGenerating && messages.length > 0 && messages[messages.length - 1].role === "assistant"
      ? messages[messages.length - 1].id
      : null;

  return (
    <div className="cw-root">
      {/* Feature 6: Session Recap Banner on Return */}
      {recapBanner && recapBanner.show && (
        <div className="bg-primary/10 border-b border-primary/20 px-5 py-2.5 flex items-center justify-between text-xs text-foreground animate-in fade-in slide-in-from-top-1 duration-200">
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-primary animate-pulse" />
            <span>
              Welcome back — you were last discussing{" "}
              <strong className="text-primary font-semibold">"{recapBanner.activeTitle}"</strong>, with{" "}
              <strong>{recapBanner.count}</strong> thread{recapBanner.count === 1 ? "" : "s"} open.
            </span>
          </div>
          <button
            onClick={onDismissRecap}
            className="text-muted-foreground hover:text-foreground p-1 rounded transition-colors"
            title="Dismiss recap"
          >
            <X size={14} />
          </button>
        </div>
      )}

      {/* ── Active node breadcrumb bar ──────────────────── */}
      {activeNodeTitle && (
        <div className="cw-node-bar">
          <GitBranch size={13} className="cw-node-bar__icon" />
          <span className="cw-node-bar__label">Active thread:</span>
          <span className="cw-node-bar__name">{activeNodeTitle}</span>
          {isGenerating && (
            <span className="cw-node-bar__generating">
              <span className="cw-node-bar__dot" />
              Generating
            </span>
          )}
        </div>
      )}

      {/* ── Messages area ───────────────────────────────── */}
      <div className="cw-messages">
        {messages.length === 0 && !isRouting ? (
          /* Empty / Welcome state */
          <div className="cw-empty">
            <div className="cw-empty__logo">
              <Cpu size={32} />
            </div>
            <h2 className="cw-empty__title">Semantic Graph Chat</h2>
            <p className="cw-empty__sub">
              Each topic you discuss gets its own isolated AI context. The router
              automatically detects your intent and branches to the right node — no context
              pollution, no forgotten details.
            </p>

            <div className="cw-empty__how">
              <div className="cw-empty__how-item">
                <span className="cw-empty__how-num">1</span>
                <span>Type any question — the AI classifies the topic</span>
              </div>
              <div className="cw-empty__how-item">
                <span className="cw-empty__how-num">2</span>
                <span>A new node appears in the graph sidebar</span>
              </div>
              <div className="cw-empty__how-item">
                <span className="cw-empty__how-num">3</span>
                <span>Switch topics freely — each node remembers its own history</span>
              </div>
            </div>

            <p className="cw-empty__try">
              <Lightbulb size={13} /> Try one of these to get started:
            </p>
            <div className="cw-suggestions">
              {SUGGESTED_PROMPTS.map((p) => (
                <button
                  key={p.text}
                  className="cw-suggest-btn"
                  onClick={() => handleSuggest(p.text)}
                >
                  <span className="cw-suggest-btn__category">{p.category}</span>
                  <p.icon size={14} className="cw-suggest-btn__icon" />
                  <span className="cw-suggest-btn__text">{p.text}</span>
                </button>
              ))}
            </div>
          </div>
        ) : (
          <>
            {messages.map((msg) => (
              <MessageBubble
                key={msg.id}
                message={msg}
                isStreaming={msg.id === streamingMsgId}
                nodeTitle={activeNodeTitle ?? undefined}
                availableNodes={availableNodes}
                onReassignMessage={onReassignMessage}
                onJumpToNode={onJumpToNode}
              />
            ))}

            {/* Routing indicator */}
            {isRouting && (
              <div className="cw-routing">
                <div className="cw-routing__spinner">
                  <Loader2 size={14} className="spin" />
                </div>
                <div className="cw-routing__text">
                  <span className="cw-routing__label">Semantic router</span>
                  <span className="cw-routing__sub">
                    Analysing intent and selecting the right topic node…
                  </span>
                </div>
              </div>
            )}
          </>
        )}
        <div ref={endRef} />
      </div>

      {/* ── Input area ──────────────────────────────────── */}
      <div className="cw-input-area">
        <form className="cw-form" onSubmit={handleSubmit}>
          <div className="cw-textarea-wrap">
            <textarea
              ref={textareaRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={
                isBusy
                  ? "Waiting for response…"
                  : "Ask anything… (Enter to send, Shift+Enter for newline)"
              }
              disabled={isBusy}
              rows={1}
              maxLength={charLimit}
              className="cw-textarea"
            />

            {/* Char count warning */}
            {input.length > charLimit * 0.8 && (
              <span className="cw-char-count">
                {input.length}/{charLimit}
              </span>
            )}

            {/* Action buttons (FileUpload + Send/Stop) */}
            <div className="cw-input-actions">
              <FileUpload
                sessionId={sessionId}
                onUploadComplete={onUploadComplete}
                disabled={isBusy}
              />

              {isBusy ? (
                <button
                  type="button"
                  className="cw-stop-btn"
                  onClick={onStopGeneration}
                  title="Stop generation"
                >
                  <Square size={14} className="fill-current" />
                  <span>Stop</span>
                </button>
              ) : (
                <button
                  type="submit"
                  className="cw-send-btn"
                  disabled={!input.trim()}
                  title="Send message (Enter)"
                >
                  <Send size={15} />
                </button>
              )}
            </div>
          </div>

          <p className="cw-footer-hint">
            Responses are scoped to each topic node. Switch nodes in the sidebar to
            change context.
          </p>
        </form>
      </div>
    </div>
  );
}

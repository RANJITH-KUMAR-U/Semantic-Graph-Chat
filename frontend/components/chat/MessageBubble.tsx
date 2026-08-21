/**
 * MessageBubble — rich per-message component.
 *
 * Features:
 * - Hover-reveal action toolbar (Copy, Thumbs Up/Down, Regenerate, Move to topic)
 * - Routing confidence badge (green ≥85%, amber 60-85%, red <60%)
 * - Reasoning popover on hover/click of topic chip or confidence badge
 * - Manual re-route dropdown (Move message to another topic or new topic)
 * - Copy with ✓ confirmation flash
 * - Streaming cursor on in-progress messages
 * - Markdown with code-block copy buttons
 * - Relative timestamp
 */
"use client";

import React, { useState, useCallback, useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Message } from "../../hooks/useChatSession";
import { TopicNode } from "../../lib/api-client";
import {
  Bot,
  User,
  Copy,
  Check,
  ThumbsUp,
  ThumbsDown,
  RotateCcw,
  Tag,
  MoreHorizontal,
  FolderInput,
  Plus,
  HelpCircle,
  Link2,
  FileText,
  Paperclip,
  Cpu,
} from "lucide-react";

interface MessageBubbleProps {
  message: Message;
  isStreaming?: boolean;
  nodeTitle?: string;
  availableNodes?: TopicNode[];
  onRegenerate?: (messageId: string) => void;
  onReassignMessage?: (messageId: string, targetNodeId?: string, newTitle?: string) => void;
  onJumpToNode?: (nodeId: string) => void;
}

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      const el = document.createElement("textarea");
      el.value = text;
      document.body.appendChild(el);
      el.select();
      document.execCommand("copy");
      document.body.removeChild(el);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  }, [text]);

  return (
    <button className="mb-copy-btn" onClick={handleCopy} title="Copy code">
      {copied ? <Check size={13} /> : <Copy size={13} />}
    </button>
  );
}

function CodeBlock({
  children,
  className,
  ...props
}: React.HTMLAttributes<HTMLElement> & { children?: React.ReactNode }) {
  const code = typeof children === "string" ? children : String(children ?? "");
  const lang = (className || "").replace("language-", "");
  return (
    <div className="mb-code-wrap">
      <div className="mb-code-header">
        <span className="mb-code-lang">{lang || "code"}</span>
        <CopyButton text={code.trimEnd()} />
      </div>
      <pre className="mb-code-pre">
        <code {...props} className={className}>
          {children}
        </code>
      </pre>
    </div>
  );
}

export function MessageBubble({
  message,
  isStreaming = false,
  nodeTitle,
  availableNodes = [],
  onRegenerate,
  onReassignMessage,
  onJumpToNode,
}: MessageBubbleProps) {
  const isUser = message.role === "user";
  const [thumbState, setThumbState] = useState<"up" | "down" | null>(null);
  const [copied, setCopied] = useState(false);
  const [showReasoning, setShowReasoning] = useState(false);
  const [showMenu, setShowMenu] = useState(false);
  const [newTopicInput, setNewTopicInput] = useState(false);
  const [customTitle, setCustomTitle] = useState("");

  const menuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    function handleClickOutside(e: MouseEvent) {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setShowMenu(false);
        setNewTopicInput(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const handleCopy = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(message.content);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {/* ignore */}
  }, [message.content]);

  const confidence = message.confidence ?? 0.90;
  const confPct = Math.round(confidence * 100);

  // Confidence color pill
  const confColorClass =
    confPct >= 85
      ? "bg-emerald-500/15 text-emerald-400 border-emerald-500/30"
      : confPct >= 60
      ? "bg-amber-500/15 text-amber-400 border-amber-500/30"
      : "bg-rose-500/15 text-rose-400 border-rose-500/30";

  const timestamp = message.createdAt
    ? new Date(message.createdAt).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
    : null;

  const handleReassign = (targetNodeId: string) => {
    onReassignMessage?.(message.id, targetNodeId);
    setShowMenu(false);
  };

  const handleCreateNewAndReassign = (e: React.FormEvent) => {
    e.preventDefault();
    if (!customTitle.trim()) return;
    onReassignMessage?.(message.id, undefined, customTitle.trim());
    setCustomTitle("");
    setNewTopicInput(false);
    setShowMenu(false);
  };

  return (
    <div id={`msg-${message.id}`} className={`mb-root${isUser ? " mb-root--user" : " mb-root--ai"}`}>
      {/* Avatar */}
      <div className="mb-avatar-col">
        <div className={`mb-avatar${isUser ? " mb-avatar--user" : " mb-avatar--ai"}`}>
          {isUser ? <User size={15} /> : <Bot size={15} />}
        </div>
      </div>

      {/* Content column */}
      <div className="mb-content">
        {/* Role label + node chip + confidence badge + time */}
        <div className="mb-meta-row">
          <span className="mb-role">{isUser ? "You" : "Assistant"}</span>

          {nodeTitle && (
            <div className="relative inline-block">
              <span
                className="mb-node-chip cursor-pointer hover:border-primary/50 transition-colors"
                onClick={() => setShowReasoning((v) => !v)}
                onMouseEnter={() => setShowReasoning(true)}
                onMouseLeave={() => setShowReasoning(false)}
                title={nodeTitle}
              >
                <Tag size={9} />
                <span className="mb-node-chip__text">{nodeTitle}</span>
              </span>
            </div>
          )}

          {/* Feature 1: Confidence badge */}
          <div className="relative inline-block">
            <span
              className={`text-[10px] font-semibold px-1.5 py-0.5 rounded border ${confColorClass} cursor-pointer flex items-center gap-1 transition-opacity hover:opacity-80`}
              onClick={() => setShowReasoning((v) => !v)}
              onMouseEnter={() => setShowReasoning(true)}
              onMouseLeave={() => setShowReasoning(false)}
              title="Semantic Router Confidence"
            >
              {confPct}%
            </span>

            {/* Reasoning popover */}
            {showReasoning && message.reasoning && (
              <div className="absolute left-0 bottom-full mb-1 z-30 w-64 p-2 bg-slate-900 border border-slate-700 text-slate-200 text-xs rounded-lg shadow-xl backdrop-blur animate-in fade-in zoom-in-95 duration-150 pointer-events-none">
                <div className="font-semibold text-primary mb-0.5 flex items-center gap-1">
                  <HelpCircle size={11} /> Router Reasoning ({confPct}% match)
                </div>
                <p className="text-[11px] text-slate-300 leading-normal">{message.reasoning}</p>
                {message.routerModelUsed && (
                  <div className="mt-1 pt-1 border-t border-slate-800 text-[10px] text-slate-400 flex items-center justify-between">
                    <span>Model: {message.routerModelUsed.split("/").pop()?.replace(":free", "")}</span>
                    {message.routerLatencyMs != null && <span>{message.routerLatencyMs}ms</span>}
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Feature 5: Router Model Badge */}
          {message.routerModelUsed && (
            <span
              className="text-[10px] font-mono px-1.5 py-0.5 rounded border border-violet-500/30 bg-violet-500/10 text-violet-300 flex items-center gap-1"
              title={`Router Model: ${message.routerModelUsed} (Latency: ${message.routerLatencyMs ?? 0}ms)`}
            >
              <Cpu size={10} />
              <span>{message.routerModelUsed.split("/").pop()?.replace(":free", "")}</span>
              {message.routerLatencyMs != null && (
                <span className="text-violet-400/70">· {message.routerLatencyMs}ms</span>
              )}
            </span>
          )}

          {timestamp && <span className="mb-time">{timestamp}</span>}

          {/* Feature 1: Cross-node reference chip */}
          {message.referencedNodeId && message.referencedNodeTitle && (
            <button
              onClick={() => onJumpToNode?.(message.referencedNodeId!)}
              className="inline-flex items-center gap-1 text-[10px] font-medium px-1.5 py-0.5 rounded border bg-sky-500/10 text-sky-400 border-sky-500/30 hover:bg-sky-500/20 transition-colors cursor-pointer"
              title={`This response drew on context from "${message.referencedNodeTitle}". Click to jump to that topic.`}
            >
              <Link2 size={9} />
              referenced: {message.referencedNodeTitle}
            </button>
          )}
        </div>

        {/* Message body */}
        <div className="mb-body">
          {isUser ? (
            <div className="mb-user-text">{message.content}</div>
          ) : (
            <div className="mb-ai-text">
              <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                components={{
                  code({ node, className, children, ...props }) {
                    const isBlock = className?.startsWith("language-");
                    if (isBlock) {
                      return (
                        <CodeBlock className={className} {...props}>
                          {children}
                        </CodeBlock>
                      );
                    }
                    return (
                      <code className="mb-inline-code" {...props}>
                        {children}
                      </code>
                    );
                  },
                  p({ children }) {
                    return <p className="mb-paragraph">{children}</p>;
                  },
                  ul({ children }) {
                    return <ul className="mb-list">{children}</ul>;
                  },
                  ol({ children }) {
                    return <ol className="mb-list mb-list--ordered">{children}</ol>;
                  },
                  li({ children }) {
                    return <li className="mb-list-item">{children}</li>;
                  },
                  h1({ children }) { return <h1 className="mb-h1">{children}</h1>; },
                  h2({ children }) { return <h2 className="mb-h2">{children}</h2>; },
                  h3({ children }) { return <h3 className="mb-h3">{children}</h3>; },
                  blockquote({ children }) {
                    return <blockquote className="mb-blockquote">{children}</blockquote>;
                  },
                  strong({ children }) {
                    return <strong className="mb-strong">{children}</strong>;
                  },
                  a({ href, children }) {
                    return (
                      <a href={href} className="mb-link" target="_blank" rel="noopener noreferrer">
                        {children}
                      </a>
                    );
                  },
                }}
              >
                {message.content}
              </ReactMarkdown>
              {isStreaming && <span className="mb-cursor">▋</span>}
            </div>
          )}

          {/* Source Citations for Document RAG */}
          {!isUser && message.sourceCitations && message.sourceCitations.length > 0 && (
            <div className="mb-sources-bar flex flex-wrap items-center gap-1.5 mt-2 pt-2 border-t border-slate-800/80 text-[11px] text-slate-400">
              <span className="font-medium text-slate-400 flex items-center gap-1 mr-1">
                <Paperclip size={11} className="text-blue-400" /> Sources:
              </span>
              {Array.from(new Set(message.sourceCitations.map((c) => c.source_filename))).map((fname) => (
                <span
                  key={fname}
                  className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md bg-slate-800/90 border border-slate-700/60 text-slate-300 hover:border-slate-600 transition-colors"
                  title={`Source document: ${fname}`}
                >
                  <FileText size={11} className="text-slate-400" />
                  <span className="font-mono text-[10.5px] truncate max-w-[150px]">{fname}</span>
                </span>
              ))}
            </div>
          )}
        </div>

        {/* Action toolbar */}
        <div className="mb-actions">
          {/* Copy */}
          <button
            className={`mb-action-btn${copied ? " mb-action-btn--active" : ""}`}
            onClick={handleCopy}
            title="Copy message"
          >
            {copied ? <Check size={13} /> : <Copy size={13} />}
            <span>{copied ? "Copied!" : "Copy"}</span>
          </button>

          {/* AI-only actions */}
          {!isUser && (
            <>
              <div className="mb-action-sep" />
              <button
                className={`mb-action-btn${thumbState === "up" ? " mb-action-btn--active" : ""}`}
                onClick={() => setThumbState(thumbState === "up" ? null : "up")}
                title="Good response"
              >
                <ThumbsUp size={13} />
              </button>
              <button
                className={`mb-action-btn${thumbState === "down" ? " mb-action-btn--active" : ""}`}
                onClick={() => setThumbState(thumbState === "down" ? null : "down")}
                title="Bad response"
              >
                <ThumbsDown size={13} />
              </button>
              {onRegenerate && (
                <>
                  <div className="mb-action-sep" />
                  <button
                    className="mb-action-btn"
                    onClick={() => onRegenerate(message.id)}
                    title="Regenerate response"
                  >
                    <RotateCcw size={13} />
                    <span>Retry</span>
                  </button>
                </>
              )}
            </>
          )}

          {/* Feature 2: Manual Re-route menu (⋯) */}
          {onReassignMessage && (
            <div className="relative ml-auto" ref={menuRef}>
              <button
                className="mb-action-btn"
                onClick={() => setShowMenu((v) => !v)}
                title="More actions / Reassign topic"
              >
                <MoreHorizontal size={14} />
              </button>

              {showMenu && (
                <div className="absolute right-0 bottom-full mb-1 z-40 w-52 bg-slate-900 border border-slate-700 rounded-lg shadow-2xl p-1.5 text-xs animate-in fade-in zoom-in-95 duration-150">
                  <div className="font-semibold text-slate-400 px-2 py-1 flex items-center gap-1">
                    <FolderInput size={12} /> Move to topic
                  </div>
                  <div className="my-1 border-t border-slate-800" />
                  <div className="max-h-40 overflow-y-auto space-y-0.5">
                    {availableNodes.map((node) => (
                      <button
                        key={node.node_id}
                        onClick={() => handleReassign(node.node_id)}
                        className="w-full text-left px-2 py-1 rounded text-slate-300 hover:bg-slate-800 hover:text-white flex items-center justify-between transition-colors truncate"
                      >
                        <span className="truncate">{node.title}</span>
                        {node.title === nodeTitle && (
                          <span className="text-[10px] text-primary font-medium ml-1">Current</span>
                        )}
                      </button>
                    ))}
                  </div>

                  <div className="my-1 border-t border-slate-800" />

                  {newTopicInput ? (
                    <form onSubmit={handleCreateNewAndReassign} className="p-1 space-y-1">
                      <input
                        type="text"
                        value={customTitle}
                        onChange={(e) => setCustomTitle(e.target.value)}
                        placeholder="New topic title..."
                        className="w-full bg-slate-950 border border-slate-700 rounded px-2 py-1 text-xs text-slate-200 focus:outline-none focus:border-primary"
                        autoFocus
                      />
                      <div className="flex gap-1 justify-end">
                        <button
                          type="button"
                          onClick={() => setNewTopicInput(false)}
                          className="px-2 py-0.5 text-[10px] text-slate-400 hover:text-slate-200"
                        >
                          Cancel
                        </button>
                        <button
                          type="submit"
                          className="px-2 py-0.5 text-[10px] bg-primary text-white font-medium rounded hover:bg-primary/90"
                        >
                          Create & Move
                        </button>
                      </div>
                    </form>
                  ) : (
                    <button
                      onClick={() => setNewTopicInput(true)}
                      className="w-full text-left px-2 py-1 rounded text-primary hover:bg-primary/10 flex items-center gap-1 font-medium transition-colors"
                    >
                      <Plus size={12} /> New topic…
                    </button>
                  )}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

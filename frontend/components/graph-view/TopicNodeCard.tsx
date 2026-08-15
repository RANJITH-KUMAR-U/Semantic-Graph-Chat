import React from "react";
import { Handle, Position, NodeProps } from "reactflow";
import { cn } from "../../lib/utils";
import { MessageSquare, GitBranch, Sparkles } from "lucide-react";

export function TopicNodeCard({ data }: NodeProps) {
  const { title, isActive, isRoot, messageCount, onSelect } = data;

  if (isRoot) {
    return (
      <div className="px-4 py-2.5 rounded-full shadow-lg border-2 border-primary/50 bg-primary/10 backdrop-blur-md text-primary font-bold text-xs flex items-center gap-2 select-none">
        <Sparkles className="w-4 h-4 animate-spin text-primary" style={{ animationDuration: "6s" }} />
        <span>Session Start</span>
        <Handle type="source" position={Position.Bottom} className="!bg-primary !w-2.5 !h-2.5" />
      </div>
    );
  }

  return (
    <div
      onClick={() => onSelect?.()}
      className={cn(
        "px-4 py-3 shadow-xl rounded-xl min-w-[210px] max-w-[260px] border-2 transition-all cursor-pointer select-none group",
        isActive
          ? "bg-card border-primary shadow-primary/25 ring-4 ring-primary/15 scale-105"
          : "bg-card/90 border-border hover:border-primary/60 hover:shadow-md"
      )}
    >
      <Handle type="target" position={Position.Top} className="!bg-muted-foreground !border-background !w-2.5 !h-2.5" />
      <Handle type="target" position={Position.Left} className="!bg-muted-foreground !border-background !w-2.5 !h-2.5" />
      
      <div className="flex items-start gap-3">
        <div
          className={cn(
            "w-9 h-9 rounded-lg flex items-center justify-center shrink-0 transition-colors mt-0.5",
            isActive
              ? "bg-primary text-primary-foreground shadow-md shadow-primary/30"
              : "bg-muted text-muted-foreground group-hover:bg-primary/20 group-hover:text-primary"
          )}
        >
          <GitBranch size={18} />
        </div>
        <div className="flex flex-col flex-1 min-w-0">
          <div className="flex items-center justify-between gap-1">
            <span className="font-semibold text-xs text-foreground truncate">{title || "Untitled Topic"}</span>
            {isActive && (
              <span className="flex h-2 w-2 relative shrink-0">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
              </span>
            )}
          </div>
          <div className="flex items-center justify-between mt-1 text-[11px]">
            <span
              className={cn(
                "font-medium",
                isActive ? "text-primary font-semibold" : "text-muted-foreground group-hover:text-foreground"
              )}
            >
              {isActive ? "Active Thread" : "Click to View"}
            </span>
            {messageCount !== undefined && messageCount > 0 && (
              <span className="flex items-center gap-1 text-[10px] bg-muted px-1.5 py-0.5 rounded text-muted-foreground">
                <MessageSquare size={10} />
                {messageCount}
              </span>
            )}
          </div>
        </div>
      </div>

      <Handle type="source" position={Position.Bottom} className="!bg-muted-foreground !border-background !w-2.5 !h-2.5" />
      <Handle type="source" position={Position.Right} className="!bg-muted-foreground !border-background !w-2.5 !h-2.5" />
    </div>
  );
}

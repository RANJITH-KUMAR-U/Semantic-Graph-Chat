/**
 * RouterModelSelector — Feature Round 3, Feature 5.
 *
 * Dropdown selector in the header that lets the user pick which router
 * model is used for semantic routing. The selection is forwarded on each
 * WS message so confidence/latency can be compared across models.
 */
"use client";

import React, { useEffect, useState, useRef } from "react";
import { getAvailableRouterModels, RouterModelInfo } from "../../lib/api-client";
import { Cpu, ChevronDown, Check } from "lucide-react";

interface RouterModelSelectorProps {
  selectedModel: string | null;
  onSelectModel: (modelId: string | null) => void;
}

export function RouterModelSelector({ selectedModel, onSelectModel }: RouterModelSelectorProps) {
  const [models, setModels] = useState<RouterModelInfo[]>([]);
  const [open, setOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    getAvailableRouterModels()
      .then(setModels)
      .catch((e) => console.error("Failed to load router models", e));
  }, []);

  useEffect(() => {
    function handleClickOutside(e: MouseEvent) {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  if (models.length < 2) return null; // No point showing selector with only 1 model

  const defaultModel = models.find((m) => m.is_default);
  const activeModel = models.find((m) => m.model_id === selectedModel) || defaultModel;
  const displayName = activeModel?.display_name || "Default";

  return (
    <div className="relative" ref={menuRef}>
      <button
        onClick={() => setOpen((v) => !v)}
        className="flex items-center gap-1.5 px-2.5 py-1 bg-violet-500/10 border border-violet-500/30 rounded-lg text-violet-400 text-xs font-semibold hover:bg-violet-500/20 transition-colors"
        title="Select router model for comparison"
      >
        <Cpu size={12} />
        <span className="max-w-[120px] truncate">Router: {displayName}</span>
        <ChevronDown size={12} />
      </button>

      {open && (
        <div className="absolute right-0 top-full mt-2 z-50 w-64 bg-slate-900 border border-slate-700 rounded-xl shadow-2xl p-1.5 text-xs animate-in fade-in zoom-in-95 duration-150">
          <div className="font-semibold text-slate-400 px-2 py-1.5 flex items-center gap-1.5">
            <Cpu size={12} />
            Router Model Playground
          </div>
          <div className="my-1 border-t border-slate-800" />
          <div className="space-y-0.5">
            {models.map((model) => {
              const isActive = model.model_id === (selectedModel || defaultModel?.model_id);
              return (
                <button
                  key={model.model_id}
                  onClick={() => {
                    onSelectModel(model.is_default ? null : model.model_id);
                    setOpen(false);
                  }}
                  className={`w-full text-left px-2.5 py-1.5 rounded-lg flex items-center justify-between transition-colors ${
                    isActive
                      ? "bg-violet-500/15 text-violet-300 border border-violet-500/20"
                      : "text-slate-300 hover:bg-slate-800 hover:text-white"
                  }`}
                >
                  <div className="flex flex-col min-w-0">
                    <span className="font-medium truncate">{model.display_name}</span>
                    <span className="text-[10px] text-slate-500 font-mono truncate">
                      {model.model_id}
                    </span>
                  </div>
                  <div className="flex items-center gap-1.5 flex-shrink-0 ml-2">
                    {model.is_default && (
                      <span className="text-[9px] bg-slate-800 text-slate-400 rounded px-1.5 py-0.5">
                        Default
                      </span>
                    )}
                    {isActive && <Check size={13} className="text-violet-400" />}
                  </div>
                </button>
              );
            })}
          </div>
          <div className="mt-1.5 border-t border-slate-800 pt-1.5 px-2">
            <p className="text-[10px] text-slate-600 leading-normal">
              Switching models changes how messages are routed. Compare confidence scores and latency across models.
            </p>
          </div>
        </div>
      )}
    </div>
  );
}

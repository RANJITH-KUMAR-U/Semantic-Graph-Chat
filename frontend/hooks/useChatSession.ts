import { useState, useEffect, useCallback, useRef } from "react";
import { ChatWebSocket, WSMessage } from "../lib/websocket-client";
import { TopicNode, listNodes, getNodeMessages, getSessionSummary } from "../lib/api-client";

export interface SourceCitation {
  source_filename: string;
  chunk_id: string;
  relevance_score: number;
  file_path?: string;
}

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  createdAt?: string;
  confidence?: number;
  reasoning?: string;
  // Feature Round 3: cross-node reference (Feature 1)
  referencedNodeId?: string;
  referencedNodeTitle?: string;
  // Feature Round 3: router model metadata (Feature 5)
  routerModelUsed?: string;
  routerLatencyMs?: number;
  // Document upload RAG: source citations
  sourceCitations?: SourceCitation[];
}

export function useChatSession(sessionId: string | null) {
  const [nodes, setNodes] = useState<TopicNode[]>([]);
  const [activeNodeId, setActiveNodeId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [globalSummary, setGlobalSummary] = useState<string>("");
  const [isSummarizing, setIsSummarizing] = useState<boolean>(false);
  const [isRouting, setIsRouting] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [tokensUsed, setTokensUsed] = useState<number>(0);
  const [baselineTokens, setBaselineTokens] = useState<number>(0);
  // Feature 5: selected router model
  const [routerModel, setRouterModel] = useState<string | null>(null);

  const wsRef = useRef<ChatWebSocket | null>(null);
  const currentAssistantMsgIdRef = useRef<string | null>(null);
  const currentRoutingMetaRef = useRef<{
    confidence?: number;
    reasoning?: string;
    routerModelUsed?: string;
    routerLatencyMs?: number;
  }>({});

  const fetchSummary = useCallback(async () => {
    if (!sessionId) return;
    try {
      const summaryData = await getSessionSummary(sessionId);
      if (summaryData.global_summary) {
        setGlobalSummary(summaryData.global_summary);
      }
    } catch (e) {
      console.error("Failed to fetch global summary", e);
    }
  }, [sessionId]);

  const fetchNodes = useCallback(async () => {
    if (!sessionId) return;
    try {
      const currentNodes = await listNodes(sessionId);
      setNodes(currentNodes);
      fetchSummary();
    } catch (e) {
      console.error("Failed to list nodes", e);
    }
  }, [sessionId, fetchSummary]);

  const selectNode = useCallback(async (nodeId: string) => {
    if (!sessionId) return;
    setActiveNodeId(nodeId);
    try {
      const history = await getNodeMessages(sessionId, nodeId);
      setMessages(
        history.map((m: any, idx: number) => ({
          id: m.message_id || `${nodeId}_${idx}`,
          role: m.role,
          content: m.content,
          confidence: m.confidence ?? 0.90,
          reasoning: m.reasoning,
        }))
      );
    } catch (e) {
      console.error("Failed to fetch node history", e);
    }
  }, [sessionId]);

  const connectWs = useCallback(() => {
    if (!sessionId) return;

    fetchNodes();

    const ws = new ChatWebSocket(sessionId);
    wsRef.current = ws;

    ws.connect((msg: WSMessage) => {
      switch (msg.type) {
        case "connected":
          console.log("Session connected:", msg.session_id);
          break;

        case "summary_status":
          if (typeof msg.is_summarizing === "boolean") {
            setIsSummarizing(msg.is_summarizing);
          }
          if (msg.global_summary) {
            setGlobalSummary(msg.global_summary);
            setIsSummarizing(false);
          }
          break;

        case "routing":
          currentRoutingMetaRef.current = {
            confidence: msg.confidence,
            reasoning: msg.reasoning,
            routerModelUsed: msg.router_model_used,
            routerLatencyMs: msg.router_latency_ms,
          };
          // Bug 1 fix: Do NOT optimistically insert a 0-message ghost node.
          // Only update activeNodeId optimistically so the UI can show the
          // "routing to: <title>" indicator. The actual node card must only
          // appear after the turn is persisted (fetchNodes on "done").
          if (msg.node_id) {
            setActiveNodeId(msg.node_id);
          }
          break;

        case "token":
          setIsRouting(false);
          setIsGenerating(true);
          if (!msg.content) break;

          setMessages((prev) => {
            const lastIdx = prev.length - 1;
            const last = prev[lastIdx];
            if (last && last.role === "assistant" && last.id === currentAssistantMsgIdRef.current) {
              const updated = [...prev];
              updated[lastIdx] = {
                ...last,
                content: last.content + msg.content,
              };
              return updated;
            } else {
              const newMsgId = Date.now().toString();
              currentAssistantMsgIdRef.current = newMsgId;
              return [
                ...prev,
                {
                  id: newMsgId,
                  role: "assistant",
                  content: msg.content || "",
                  createdAt: new Date().toISOString(),
                  confidence: currentRoutingMetaRef.current.confidence ?? 0.90,
                  reasoning: currentRoutingMetaRef.current.reasoning,
                  routerModelUsed: currentRoutingMetaRef.current.routerModelUsed,
                  routerLatencyMs: currentRoutingMetaRef.current.routerLatencyMs,
                },
              ];
            }
          });
          break;

        case "done":
          setIsRouting(false);
          setIsGenerating(false);
          // Bug 4 fix: use != null so that legitimate 0 values aren't discarded.
          if (msg.tokens_used != null) setTokensUsed(msg.tokens_used);
          if (msg.baseline_tokens != null) setBaselineTokens(msg.baseline_tokens);

          // Feature 1: attach cross-node reference info to the completed assistant message
          if ((msg.referenced_node_id || msg.source_citations) && currentAssistantMsgIdRef.current) {
            const refId = msg.referenced_node_id;
            const refTitle = msg.referenced_node_title;
            const citations = msg.source_citations;
            setMessages((prev) => {
              const lastIdx = prev.length - 1;
              const last = prev[lastIdx];
              if (last && last.id === currentAssistantMsgIdRef.current) {
                const updated = [...prev];
                updated[lastIdx] = {
                  ...last,
                  ...(refId ? { referencedNodeId: refId, referencedNodeTitle: refTitle } : {}),
                  ...(citations ? { sourceCitations: citations } : {}),
                };
                return updated;
              }
              return prev;
            });
          }

          currentAssistantMsgIdRef.current = null;
          // Refresh sidebar nodes and sync active node message history from backend
          fetchNodes();
          if (msg.node_id) {
            selectNode(msg.node_id);
          }
          break;

        case "summary_status":
          if (msg.global_summary) {
            setGlobalSummary(msg.global_summary);
            fetchNodes();
          }
          break;

        case "error":
          setIsRouting(false);
          setIsGenerating(false);
          setError(msg.content || "An unexpected error occurred");
          currentAssistantMsgIdRef.current = null;
          break;
      }
    });
  }, [sessionId, fetchNodes]);

  useEffect(() => {
    if (!sessionId) return;
    connectWs();

    return () => {
      wsRef.current?.disconnect();
    };
  }, [sessionId, connectWs]);

  const sendMessage = useCallback((content: string, forceNodeId?: string) => {
    if (!wsRef.current) return;
    setIsRouting(true);
    setIsGenerating(false);
    setError(null);
    currentAssistantMsgIdRef.current = null;
    setMessages((prev) => [
      ...prev,
      { id: Date.now().toString(), role: "user", content, createdAt: new Date().toISOString() },
    ]);
    // Feature 5: forward routerModel selection
    wsRef.current.sendMessage(content, forceNodeId, routerModel || undefined);
  }, [routerModel]);

  const stopGeneration = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.disconnect();
    }
    setIsRouting(false);
    setIsGenerating(false);
    currentAssistantMsgIdRef.current = null;

    if (sessionId) {
      setTimeout(() => {
        connectWs();
      }, 200);
    }
  }, [sessionId, connectWs]);

  return {
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
    setMessages,
    setNodes,
    // Feature 5: router model state
    routerModel,
    setRouterModel,
  };
}

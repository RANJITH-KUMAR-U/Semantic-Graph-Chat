function getWsBaseUrl(): string {
  if (process.env.NEXT_PUBLIC_WS_URL) {
    return process.env.NEXT_PUBLIC_WS_URL;
  }
  if (process.env.NEXT_PUBLIC_API_BASE_URL) {
    const raw = process.env.NEXT_PUBLIC_API_BASE_URL.trim().replace(/\/+$/, "");
    const cleanHost = raw.replace(/^https?:\/\//, "").replace(/^wss?:\/\//, "");
    const isLocalhost = cleanHost.includes("localhost") || cleanHost.includes("127.0.0.1");
    const protocol = isLocalhost ? "ws" : "wss";
    return `${protocol}://${cleanHost}/ws/chat`;
  }
  return "ws://localhost:8000/ws/chat";
}

const WS_BASE_URL = getWsBaseUrl();

export interface WSMessage {
  type: "connected" | "routing" | "token" | "done" | "error" | "summary_status";
  content?: string;
  node_id?: string;
  node_title?: string;
  parent_node_id?: string;
  node_depth?: number;
  reasoning?: string;
  confidence?: number;
  global_summary?: string;
  is_summarizing?: boolean;
  tokens_used?: number;
  baseline_tokens?: number;
  session_id?: string;
  // Feature Round 3: cross-node reference (Feature 1)
  referenced_node_id?: string;
  referenced_node_title?: string;
  // Feature Round 3: router model metadata (Feature 5)
  router_model_used?: string;
  router_latency_ms?: number;
  // Document upload RAG: source citations
  source_citations?: Array<{
    source_filename: string;
    chunk_id: string;
    relevance_score: number;
    file_path?: string;
  }>;
}

type MessageHandler = (event: WSMessage) => void;

export class ChatWebSocket {
  private ws: WebSocket | null = null;
  private onMessage: MessageHandler | null = null;

  constructor(private sessionId: string) {}

  connect(onMessage: MessageHandler) {
    this.onMessage = onMessage;
    this.ws = new WebSocket(`${WS_BASE_URL}/${this.sessionId}`);

    this.ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data) as WSMessage;
        this.onMessage?.(data);
      } catch (e) {
        console.error("Failed to parse websocket message", e);
      }
    };

    this.ws.onerror = (e) => console.error("WebSocket error", e);
    this.ws.onclose = () => console.log("WebSocket closed");
  }

  /** Feature 5: added optional routerModel param */
  sendMessage(content: string, forceNodeId?: string, routerModel?: string) {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({
        content,
        force_node_id: forceNodeId || null,
        router_model: routerModel || null,
      }));
    } else {
      console.error("WebSocket is not connected");
    }
  }

  disconnect() {
    this.ws?.close();
    this.ws = null;
  }
}

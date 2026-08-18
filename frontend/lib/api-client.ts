function getApiBaseUrl(): string {
  const envUrl = process.env.NEXT_PUBLIC_API_BASE_URL;
  if (!envUrl) return "http://localhost:8000";
  const trimmed = envUrl.trim().replace(/\/+$/, "");
  if (trimmed.startsWith("http://") || trimmed.startsWith("https://")) {
    return trimmed;
  }
  return `https://${trimmed}`;
}

const API_BASE_URL = getApiBaseUrl();

export interface SessionData {
  session_id: string;
  created_at: string;
  global_summary: string;
}

/** Flat node as returned by the REST API */
export interface TopicNode {
  node_id: string;
  session_id?: string;
  title: string;
  created_at: string;
  last_active_at?: string;
  message_count?: number;
  parent_node_id?: string | null;
  depth?: number;          // 0 = root topic, 1 = sub-topic
  possible_duplicate_of?: string | null;
  // Feature Round 3: relatedness graph
  related_node_ids?: string[];
  // Document upload: chunk count
  document_chunk_count?: number;
}

export interface UploadStatus {
  upload_id: string;
  filename: string;
  status: "queued" | "chunking" | "routing" | "indexed" | "failed";
  total_chunks: number;
  error?: string;
  node_assignments?: Record<string, { title: string; chunk_count: number }>;
}

/** Tree-shaped node for UI rendering */
export interface TopicTreeNode extends TopicNode {
  children: TopicTreeNode[];
}

export interface SearchResult {
  message_id: string;
  node_id: string;
  node_title: string;
  role: "user" | "assistant";
  content: string;
  created_at?: string;
}

export interface SessionRecap {
  has_history: boolean;
  active_node_id?: string | null;
  active_node_title?: string | null;
  node_count: number;
  global_summary: string;
  session_tokens_used?: number;
  session_baseline_tokens?: number;
}

// Feature Round 3, Feature 3: Routing log entry
export interface RoutingLogEntry {
  timestamp: string;
  message_excerpt: string;
  decision: string;
  target_node_id: string;
  node_title: string;
  confidence: number;
  reasoning: string;
  router_model?: string;
  latency_ms?: number;
}

// Feature Round 3, Feature 5: Router model info
export interface RouterModelInfo {
  model_id: string;
  display_name: string;
  is_default: boolean;
}

/**
 * Convert a flat list of TopicNodes into a nested tree.
 */
export function buildNodeTree(flatNodes: TopicNode[]): TopicTreeNode[] {
  const nodeMap = new Map<string, TopicTreeNode>();

  for (const n of flatNodes) {
    nodeMap.set(n.node_id, { ...n, children: [] });
  }

  const roots: TopicTreeNode[] = [];

  for (const n of flatNodes) {
    const treeNode = nodeMap.get(n.node_id)!;
    if (n.parent_node_id && nodeMap.has(n.parent_node_id)) {
      nodeMap.get(n.parent_node_id)!.children.push(treeNode);
    } else {
      roots.push(treeNode);
    }
  }

  return roots;
}

export async function createSession(sessionId?: string): Promise<SessionData> {
  const response = await fetch(`${API_BASE_URL}/api/sessions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(sessionId ? { session_id: sessionId } : {}),
  });
  if (!response.ok) throw new Error("Failed to create session");
  return response.json();
}

export async function listNodes(sessionId: string): Promise<TopicNode[]> {
  const response = await fetch(`${API_BASE_URL}/api/sessions/${sessionId}/nodes`);
  if (response.status === 404) return [];
  if (!response.ok) throw new Error("Failed to fetch nodes");
  return response.json();
}

export async function getNodeMessages(sessionId: string, nodeId: string) {
  const response = await fetch(`${API_BASE_URL}/api/nodes/${nodeId}/messages?session_id=${sessionId}`);
  if (response.status === 404) return [];
  if (!response.ok) throw new Error("Failed to fetch node messages");
  return response.json();
}

export async function forceRoute(sessionId: string, nodeId: string): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/api/nodes/${nodeId}/force-route`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId }),
  });
  if (!response.ok) throw new Error("Failed to force route");
}

export async function getSessionSummary(sessionId: string): Promise<{ global_summary: string }> {
  const response = await fetch(`${API_BASE_URL}/api/sessions/${sessionId}/summary`);
  if (response.status === 404) {
    return { global_summary: "" };
  }
  if (!response.ok) throw new Error("Failed to fetch summary");
  return response.json();
}

export async function reassignMessage(
  sessionId: string,
  messageId: string,
  targetNodeId?: string,
  newTopicTitle?: string
): Promise<{ success: boolean; target_node_id: string; target_node_title: string }> {
  const response = await fetch(`${API_BASE_URL}/api/messages/${messageId}/reassign`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      session_id: sessionId,
      target_node_id: targetNodeId || null,
      new_topic_title: newTopicTitle || null,
    }),
  });
  if (!response.ok) throw new Error("Failed to reassign message");
  return response.json();
}

export async function mergeNodes(
  sessionId: string,
  sourceNodeId: string,
  targetNodeId: string
): Promise<{ success: boolean; target_node_id: string }> {
  const response = await fetch(`${API_BASE_URL}/api/nodes/merge`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      session_id: sessionId,
      source_node_id: sourceNodeId,
      target_node_id: targetNodeId,
    }),
  });
  if (!response.ok) throw new Error("Failed to merge nodes");
  return response.json();
}

export async function searchMessages(sessionId: string, query: string): Promise<SearchResult[]> {
  if (!query.trim()) return [];
  const response = await fetch(`${API_BASE_URL}/api/sessions/${sessionId}/search?q=${encodeURIComponent(query)}`);
  if (!response.ok) throw new Error("Failed to search messages");
  return response.json();
}

export async function getSessionRecap(sessionId: string): Promise<SessionRecap> {
  const response = await fetch(`${API_BASE_URL}/api/sessions/${sessionId}/recap`);
  if (!response.ok) throw new Error("Failed to get session recap");
  return response.json();
}

// ── Feature Round 3 API functions ────────────────────────────────────────

/** Feature 3: Get routing decision timeline log */
export async function getRoutingLog(sessionId: string): Promise<RoutingLogEntry[]> {
  const response = await fetch(`${API_BASE_URL}/api/sessions/${sessionId}/routing-log`);
  if (!response.ok) throw new Error("Failed to fetch routing log");
  return response.json();
}

/** Feature 4: Export session as Markdown — triggers browser download */
export async function exportSession(sessionId: string): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/api/sessions/${sessionId}/export`);
  if (!response.ok) throw new Error("Failed to export session");
  const blob = await response.blob();
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `session-${sessionId.slice(0, 8)}.md`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  window.URL.revokeObjectURL(url);
}

/** Feature 5: Get list of available router models */
export async function getAvailableRouterModels(): Promise<RouterModelInfo[]> {
  const response = await fetch(`${API_BASE_URL}/api/config/available-router-models`);
  if (!response.ok) throw new Error("Failed to fetch router models");
  return response.json();
}

/** Upload document file (.pdf, .docx, .txt, .md, .zip) */
export async function uploadFile(
  sessionId: string,
  file: File
): Promise<UploadStatus> {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_BASE_URL}/api/sessions/${sessionId}/upload`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || `Upload failed with status ${response.status}`);
  }

  return response.json();
}


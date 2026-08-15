"use client";

import React, { useRef, useState, useCallback } from "react";
import { Paperclip, FileText, CheckCircle2, AlertCircle, Loader2, X } from "lucide-react";
import { uploadFile, UploadStatus } from "../../lib/api-client";

interface FileUploadProps {
  sessionId: string | null;
  onUploadComplete?: () => void;
  disabled?: boolean;
}

export function FileUpload({ sessionId, onUploadComplete, disabled = false }: FileUploadProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<UploadStatus | null>(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const handleFileSelected = async (file: File) => {
    if (!sessionId) {
      setErrorMessage("No active session for file upload.");
      return;
    }

    setErrorMessage(null);
    setIsUploading(true);
    setUploadStatus({
      upload_id: "temp",
      filename: file.name,
      status: "chunking",
      total_chunks: 0,
    });

    try {
      // Step 1: Chunking & Routing on Backend
      setUploadStatus((prev) => (prev ? { ...prev, status: "routing" } : null));
      const res = await uploadFile(sessionId, file);
      setUploadStatus(res);
      if (res.status === "indexed") {
        onUploadComplete?.();
      }
    } catch (err: any) {
      console.error("File upload error:", err);
      setErrorMessage(err.message || "Failed to upload document");
      setUploadStatus((prev) => (prev ? { ...prev, status: "failed", error: err.message } : null));
    } finally {
      setIsUploading(false);
    }
  };

  const onInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files[0]) {
      handleFileSelected(files[0]);
    }
    // Reset file input value so re-uploading same file triggers change
    if (e.target) {
      e.target.value = "";
    }
  };

  const handleTriggerClick = () => {
    if (disabled || isUploading) return;
    fileInputRef.current?.click();
  };

  const dismissStatus = () => {
    setUploadStatus(null);
    setErrorMessage(null);
  };

  return (
    <div className="file-upload-container">
      <input
        ref={fileInputRef}
        type="file"
        accept=".pdf,.docx,.txt,.md,.zip"
        onChange={onInputChange}
        style={{ display: "none" }}
      />

      {/* Upload button icon */}
      <button
        type="button"
        className={`file-upload-btn ${isUploading ? "file-upload-btn--busy" : ""}`}
        onClick={handleTriggerClick}
        disabled={disabled || isUploading || !sessionId}
        title="Upload Document / PDF / Code ZIP (.pdf, .docx, .txt, .md, .zip)"
      >
        {isUploading ? (
          <Loader2 size={16} className="spin text-blue-400" />
        ) : (
          <Paperclip size={16} />
        )}
      </button>

      {/* Upload Status Card overlay / pill */}
      {(uploadStatus || errorMessage) && (
        <div className="upload-status-pill">
          <div className="upload-status-pill__info">
            <FileText size={14} className="upload-status-pill__icon" />
            <span className="upload-status-pill__name" title={uploadStatus?.filename}>
              {uploadStatus?.filename || "File"}
            </span>

            {isUploading && (
              <span className="upload-status-pill__badge upload-status-pill__badge--busy">
                <Loader2 size={12} className="spin mr-1 inline" />
                {uploadStatus?.status === "chunking" ? "Chunking…" : "Routing to topics…"}
              </span>
            )}

            {!isUploading && uploadStatus?.status === "indexed" && (
              <span className="upload-status-pill__badge upload-status-pill__badge--success">
                <CheckCircle2 size={12} className="mr-1 inline" />
                Indexed ({uploadStatus.total_chunks} chunks
                {uploadStatus.node_assignments && Object.keys(uploadStatus.node_assignments).length > 0
                  ? ` across ${Object.keys(uploadStatus.node_assignments).length} topic${Object.keys(uploadStatus.node_assignments).length > 1 ? "s" : ""}`
                  : ""}
                )
              </span>
            )}

            {(errorMessage || uploadStatus?.status === "failed") && (
              <span className="upload-status-pill__badge upload-status-pill__badge--error">
                <AlertCircle size={12} className="mr-1 inline" />
                {errorMessage || uploadStatus?.error || "Upload failed"}
              </span>
            )}
          </div>

          <button
            type="button"
            className="upload-status-pill__close"
            onClick={dismissStatus}
            title="Dismiss status"
          >
            <X size={12} />
          </button>
        </div>
      )}
    </div>
  );
}

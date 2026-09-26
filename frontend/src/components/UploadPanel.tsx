import { useId, useRef, useState } from "react";
import { validateFile } from "../api/analyze";

export function PdfIcon() {
  return (
    <svg
      width="30"
      height="36"
      viewBox="0 0 30 36"
      fill="none"
      aria-hidden="true"
    >
      <path d="M5 1h13l7 7v26H5z" stroke="currentColor" strokeWidth="1.5" />
      <path
        d="M18 1v8h7M10 16h10M10 21h10M10 26h7"
        stroke="currentColor"
        strokeWidth="1.5"
      />
    </svg>
  );
}
function sizeLabel(bytes: number) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}
interface Props {
  title: string;
  number: string;
  file: File | null;
  disabled: boolean;
  onChange: (file: File | null) => void;
}
export default function UploadPanel({
  title,
  number,
  file,
  disabled,
  onChange,
}: Props) {
  const id = useId();
  const input = useRef<HTMLInputElement>(null);
  const [error, setError] = useState("");
  const [dragging, setDragging] = useState(false);
  function select(files: FileList | null) {
    if (disabled || !files?.length) return;
    if (files.length !== 1) {
      setError("Choose one PDF for this document.");
      return;
    }
    const message = validateFile(files[0]);
    setError(message ?? "");
    if (!message) onChange(files[0]);
  }
  return (
    <section className="upload-panel" aria-labelledby={`${id}-title`}>
      <div className="upload-heading">
        <span className="step-number">{number}</span>
        <div>
          <h2 id={`${id}-title`}>{title}</h2>
        </div>
      </div>
      <div
        className={`dropzone ${dragging ? "dragging" : ""} ${file ? "selected" : ""}`}
        onDragOver={(event) => {
          event.preventDefault();
          if (!disabled) setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(event) => {
          event.preventDefault();
          setDragging(false);
          select(event.dataTransfer.files);
        }}
      >
        <span className="document-icon">
          <PdfIcon />
        </span>
        {file ? (
          <>
            <strong className="filename">{file.name}</strong>
            <span className="muted">PDF document · {sizeLabel(file.size)}</span>
            <div className="file-actions">
              <button
                type="button"
                className="text-button"
                disabled={disabled}
                onClick={() => input.current?.click()}
                aria-label={`Change ${title}`}
              >
                Change file
              </button>
              <span aria-hidden="true">·</span>
              <button
                type="button"
                className="text-button"
                disabled={disabled}
                onClick={() => {
                  onChange(null);
                  setError("");
                }}
                aria-label={`Remove ${title}`}
              >
                Remove
              </button>
            </div>
          </>
        ) : (
          <>
            <strong>Drop your PDF here</strong>
            <button
              type="button"
              className="secondary"
              disabled={disabled}
              onClick={() => input.current?.click()}
            >
              Choose{" "}
              {title === "Patient chart" ? "patient chart" : "policy PDF"}
            </button>
          </>
        )}
        <label className="sr-only" htmlFor={id}>
          {title}
        </label>
        <input
          ref={input}
          id={id}
          type="file"
          accept=".pdf,application/pdf"
          disabled={disabled}
          aria-label={title}
          aria-describedby={error ? `${id}-error` : undefined}
          onChange={(event) => {
            select(event.target.files);
            event.target.value = "";
          }}
          hidden
        />
      </div>
      <div className="upload-foot">PDF only</div>
      {error && (
        <p className="field-error" role="alert" id={`${id}-error`}>
          {error}
        </p>
      )}
    </section>
  );
}

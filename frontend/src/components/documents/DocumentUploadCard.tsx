import React, { useState, useRef } from 'react';
import { UploadCloud, FileText, CheckCircle2, AlertCircle, Loader2 } from 'lucide-react';
import { DocumentItem } from '../../types';
import { apiService, ApiError } from '../../services/api';
import { useApp } from '../../context/AppContext';

interface DocumentUploadCardProps {
  onUploaded?: (doc: DocumentItem) => void;
  className?: string;
}

export const DocumentUploadCard: React.FC<DocumentUploadCardProps> = ({
  onUploaded,
  className = '',
}) => {
  const { refreshDocuments, selectedCompany } = useApp();

  const [ticker, setTicker] = useState(selectedCompany?.ticker || '');
  const [docType, setDocType] =
    useState<DocumentItem['doc_type']>('10-K');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      validateAndSetFile(e.dataTransfer.files[0]);
    }
  };

  const validateAndSetFile = (file: File) => {
    setErrorMessage(null);
    setSuccessMessage(null);

    if (!file.name.toLowerCase().endsWith('.pdf')) {
      setErrorMessage('Validation failure: Only PDF files (.pdf) are supported in Research Vault.');
      setSelectedFile(null);
      return;
    }

    if (file.size > 20 * 1024 * 1024) {
      setErrorMessage('Validation failure: File size exceeds the 20MB maximum threshold.');
      setSelectedFile(null);
      return;
    }

    setSelectedFile(file);
  };

  const handleUpload = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedFile) {
      setErrorMessage('Please select a PDF document to upload.');
      return;
    }
    if (!ticker.trim()) {
      setErrorMessage('Stock ticker symbol is required to index the document.');
      return;
    }

    setIsUploading(true);
    setUploadProgress(15);
    setErrorMessage(null);

    try {
      // Simulate progress progression
      const interval = setInterval(() => {
        setUploadProgress((prev) => {
          if (prev >= 85) {
            clearInterval(interval);
            return 85;
          }
          return prev + 25;
        });
      }, 70);

      const newDoc = await apiService.uploadDocument(selectedFile, ticker, docType);
      clearInterval(interval);
      setUploadProgress(100);

      setSuccessMessage(
        `Successfully indexed "${newDoc.filename}" for ${newDoc.ticker} (${newDoc.chunk_count} vector chunks generated).`,
      );
      setSelectedFile(null);
      if (fileInputRef.current) fileInputRef.current.value = '';
      await refreshDocuments();
      if (onUploaded) onUploaded(newDoc);
    } catch (err: unknown) {
      const msg = err instanceof ApiError ? err.message : err instanceof Error ? err.message : 'Upload failed';
      setErrorMessage(msg);
    } finally {
      setIsUploading(false);
      setTimeout(() => setUploadProgress(0), 1000);
    }
  };

  return (
    <form
      data-testid="document-upload-card"
      onSubmit={handleUpload}
      className={`bg-white rounded-xl border border-slate-200 shadow-sm p-6 ${className}`}
    >
      <div className="pb-4 border-b border-slate-100">
        <h3 className="text-base font-bold text-slate-900">
          Upload Research Filing (SEC / Transcripts)
        </h3>
        <p className="text-xs text-slate-500 mt-0.5">
          Ingest 10-K, 10-Q, or earnings call PDFs into the local ChromaDB vector vault for grounded citation.
        </p>
      </div>

      <div className="space-y-4 pt-4">
        {/* Metadata row */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          <div>
            <label
              htmlFor="upload-ticker"
              className="block text-xs font-semibold uppercase text-slate-700 mb-1"
            >
              Associated Stock Ticker
            </label>
            <input
              id="upload-ticker"
              type="text"
              value={ticker}
              onChange={(e) => setTicker(e.target.value.toUpperCase())}
              placeholder="e.g. AAPL, NVDA, INFY"
              className="w-full px-3 py-2 text-sm bg-white border border-slate-300 rounded-lg text-slate-900 focus:ring-2 focus:ring-emerald-500 focus:outline-none"
              required
            />
          </div>

          <div>
            <label
              htmlFor="upload-doctype"
              className="block text-xs font-semibold uppercase text-slate-700 mb-1"
            >
              Document Category
            </label>
            <select
              id="upload-doctype"
              value={docType}
              onChange={(e) =>
                setDocType(e.target.value as DocumentItem['doc_type'])
              }
              className="w-full px-3 py-2 text-sm bg-white border border-slate-300 rounded-lg text-slate-900 focus:ring-2 focus:ring-emerald-500 focus:outline-none"
            >
              <option value="10-K">Form 10-K (Annual Report)</option>
              <option value="10-Q">Form 10-Q (Quarterly Report)</option>
              <option value="8-K">Form 8-K (Material Event)</option>
              <option value="Earnings Transcript">Earnings Call Transcript</option>
              <option value="Investor Presentation">Investor Presentation</option>
            </select>
          </div>
        </div>

        {/* Drag and Drop Zone */}
        <div
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
          className={`border-2 border-dashed rounded-xl p-6 text-center transition-all ${
            dragActive
              ? 'border-emerald-500 bg-emerald-50/40'
              : 'border-slate-300 hover:border-slate-400 bg-slate-50/50'
          }`}
        >
          <input
            ref={fileInputRef}
            type="file"
            id="file-upload-input"
            accept=".pdf"
            onChange={(e) => {
              if (e.target.files && e.target.files[0]) {
                validateAndSetFile(e.target.files[0]);
              }
            }}
            className="hidden"
          />

          <div className="flex flex-col items-center justify-center">
            <UploadCloud className="w-8 h-8 text-slate-400 mb-2" />
            {selectedFile ? (
              <div className="text-center">
                <div className="flex items-center justify-center text-xs font-bold text-slate-900 space-x-1.5">
                  <FileText className="w-4 h-4 text-emerald-600" />
                  <span>{selectedFile.name}</span>
                </div>
                <span className="text-[11px] text-slate-500 mt-0.5 block">
                  {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                </span>
                <button
                  type="button"
                  onClick={() => {
                    setSelectedFile(null);
                    if (fileInputRef.current) fileInputRef.current.value = '';
                  }}
                  className="mt-2 text-xs text-rose-600 hover:underline"
                >
                  Remove
                </button>
              </div>
            ) : (
              <div>
                <p className="text-xs text-slate-700 font-medium">
                  Drag and drop PDF filing here, or{' '}
                  <button
                    type="button"
                    onClick={() => fileInputRef.current?.click()}
                    className="text-emerald-700 font-bold hover:underline"
                  >
                    browse files
                  </button>
                </p>
                <p className="text-[11px] text-slate-400 mt-1">
                  PDF format only • Maximum file size: 20MB
                </p>
              </div>
            )}
          </div>
        </div>

        {/* Progress Bar during upload */}
        {isUploading && (
          <div className="space-y-1.5">
            <div className="flex justify-between text-xs font-semibold text-slate-700">
              <span className="flex items-center">
                <Loader2 className="w-3.5 h-3.5 mr-1.5 animate-spin text-emerald-600" />
                Chunking & Generating Embeddings...
              </span>
              <span>{uploadProgress}%</span>
            </div>
            <div className="w-full bg-slate-100 rounded-full h-2 overflow-hidden">
              <div
                className="bg-emerald-500 h-full transition-all duration-300"
                style={{ width: `${uploadProgress}%` }}
              />
            </div>
          </div>
        )}

        {/* Error Feedback */}
        {errorMessage && (
          <div className="p-3 rounded-lg bg-rose-50 border border-rose-200 text-xs text-rose-700 flex items-start space-x-2">
            <AlertCircle className="w-4 h-4 text-rose-600 flex-shrink-0 mt-0.5" />
            <span>{errorMessage}</span>
          </div>
        )}

        {/* Success Feedback */}
        {successMessage && (
          <div className="p-3 rounded-lg bg-emerald-50 border border-emerald-200 text-xs text-emerald-800 flex items-start space-x-2">
            <CheckCircle2 className="w-4 h-4 text-emerald-600 flex-shrink-0 mt-0.5" />
            <span>{successMessage}</span>
          </div>
        )}

        {/* Submit */}
        <div className="flex justify-end pt-2">
          <button
            type="submit"
            disabled={!selectedFile || isUploading}
            className="px-4 py-2 rounded-lg bg-slate-900 hover:bg-slate-800 disabled:opacity-40 text-white text-xs font-semibold transition-colors shadow-sm"
          >
            {isUploading ? 'Indexing Document...' : 'Upload & Process Vector Embeddings'}
          </button>
        </div>
      </div>
    </form>
  );
};

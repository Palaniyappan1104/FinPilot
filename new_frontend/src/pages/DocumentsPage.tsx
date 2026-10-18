import React from 'react';
import {
  FolderLock,
  UploadCloud,
  FileText,
  Search,
  ShieldCheck,
  Info,
  CheckCircle2,
} from 'lucide-react';

export const DocumentsPage: React.FC = () => {
  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 sm:p-8 flex flex-col sm:flex-row sm:items-center justify-between gap-4 shadow-xl">
        <div className="space-y-1">
          <div className="flex items-center space-x-2 text-purple-400">
            <FolderLock className="w-5 h-5" />
            <span className="text-xs font-bold uppercase tracking-wider">Research Vault & Filings</span>
          </div>
          <h1 className="text-2xl sm:text-3xl font-black text-white tracking-tight">
            Institutional Document Vault
          </h1>
          <p className="text-xs text-slate-400 max-w-2xl">
            Upload proprietary 10-K, 10-Q, earnings call transcripts, and equity research PDFs for vector indexing and deterministic citation grounding.
          </p>
        </div>

        <div className="px-3.5 py-2 rounded-xl bg-purple-950/60 border border-purple-800 text-purple-300 text-xs font-semibold flex items-center space-x-2 self-start sm:self-auto">
          <ShieldCheck className="w-4 h-4 text-purple-400" />
          <span>Local Vector Storage Ready</span>
        </div>
      </div>

      {/* Info Notice about RAG in Demo API */}
      <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 text-xs text-slate-300 flex items-start space-x-3">
        <Info className="w-4 h-4 text-emerald-400 flex-shrink-0 mt-0.5" />
        <div className="space-y-1">
          <span className="font-bold text-white">Research Vault Architecture Notice:</span>
          <p className="text-slate-400 leading-relaxed">
            The lightweight standalone demo backend (<code className="text-emerald-400">demo.py</code>) routes directly through Groq and live yfinance market data without vector database dependencies. Full ChromaDB chunking and embedding retrieval pipelines are decoupled and ready for enterprise LangGraph multi-agent deployment.
          </p>
        </div>
      </div>

      {/* Upload Dropzone Placeholder */}
      <div className="bg-slate-900 border border-dashed border-slate-800 hover:border-slate-700 rounded-2xl p-12 text-center space-y-4 transition-colors">
        <div className="w-14 h-14 rounded-2xl bg-slate-950 border border-slate-800 flex items-center justify-center text-purple-400 mx-auto">
          <UploadCloud className="w-7 h-7" />
        </div>

        <div className="space-y-1 max-w-sm mx-auto">
          <h3 className="text-sm font-bold text-white">Upload Financial Research Filing</h3>
          <p className="text-xs text-slate-400">
            Drag & drop PDF files (10-K, 10-Q, 8-K, Transcript), or browse from your workstation.
          </p>
        </div>

        <button
          type="button"
          disabled
          className="px-5 py-2 rounded-xl bg-slate-800 text-slate-400 text-xs font-semibold cursor-not-allowed border border-slate-700"
        >
          Select Document (Vault Standby)
        </button>
      </div>

      {/* Vault Capabilities Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="p-5 rounded-2xl bg-slate-900 border border-slate-800 space-y-2">
          <div className="flex items-center space-x-2 text-purple-400 text-xs font-bold uppercase tracking-wider">
            <CheckCircle2 className="w-4 h-4" />
            <span>Deterministic Parsing</span>
          </div>
          <p className="text-xs text-slate-400 leading-relaxed">
            Extracts financial tables, auditor notes, and executive risk factors with sub-second page accuracy.
          </p>
        </div>

        <div className="p-5 rounded-2xl bg-slate-900 border border-slate-800 space-y-2">
          <div className="flex items-center space-x-2 text-purple-400 text-xs font-bold uppercase tracking-wider">
            <CheckCircle2 className="w-4 h-4" />
            <span>Semantic Chunking</span>
          </div>
          <p className="text-xs text-slate-400 leading-relaxed">
            Splits dense financial statements by section headers (Item 1A, Item 7) to preserve balance-sheet context.
          </p>
        </div>

        <div className="p-5 rounded-2xl bg-slate-900 border border-slate-800 space-y-2">
          <div className="flex items-center space-x-2 text-purple-400 text-xs font-bold uppercase tracking-wider">
            <CheckCircle2 className="w-4 h-4" />
            <span>Auditable Citations</span>
          </div>
          <p className="text-xs text-slate-400 leading-relaxed">
            Every synthesized conclusion traces back to exact page numbers and paragraphs in the filed source PDF.
          </p>
        </div>
      </div>
    </div>
  );
};

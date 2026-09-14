import React from 'react';
import { FolderLock } from 'lucide-react';
import { useApp } from '../context/AppContext';
import { DocumentUploadCard } from '../components/documents/DocumentUploadCard';
import { DocumentListTable } from '../components/documents/DocumentListTable';
import { ResearchQAInterface } from '../components/documents/ResearchQAInterface';

export const DocumentsPage: React.FC = () => {
  const { documents } = useApp();

  return (
    <div data-testid="documents-page" className="space-y-6">
      {/* Page Header */}
      <div className="pb-4 border-b border-slate-200">
        <div className="flex items-center space-x-2">
          <FolderLock className="w-5 h-5 text-emerald-600" />
          <h2 className="text-xl font-bold text-slate-900">
            Research Filing Vault & Grounding Center
          </h2>
        </div>
        <p className="text-xs text-slate-500 mt-0.5">
          Ingest SEC filings and transcripts, review indexed vector chunks in ChromaDB, and run grounded document Q&A queries.
        </p>
      </div>

      {/* Grid: Upload on top / left, Q&A on right */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <DocumentUploadCard />
        <ResearchQAInterface />
      </div>

      {/* Table of Indexed Documents */}
      <DocumentListTable documents={documents} />
    </div>
  );
};

import { describe, it, expect } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { DocumentUploadCard } from '../components/documents/DocumentUploadCard';
import { DocumentListTable } from '../components/documents/DocumentListTable';
import { ResearchQAInterface } from '../components/documents/ResearchQAInterface';
import { AppContextProvider } from '../context/AppContext';
import { MOCK_DOCUMENTS } from '../services/mockData';

describe('Research Document Upload & Grounding (Phase 16.8)', () => {
  it('validates file upload format and rejects non-PDF documents', async () => {
    render(
      <AppContextProvider>
        <DocumentUploadCard />
      </AppContextProvider>,
    );

    expect(screen.getByTestId('document-upload-card')).toBeInTheDocument();

    const fileInput = document.getElementById(
      'file-upload-input',
    ) as HTMLInputElement;

    // Create a dummy txt file
    const invalidFile = new File(['dummy content'], 'report.txt', {
      type: 'text/plain',
    });

    fireEvent.change(fileInput, { target: { files: [invalidFile] } });

    await waitFor(() => {
      expect(
        screen.getByText(
          /Only PDF files \(\.pdf\) are supported in Research Vault/i,
        ),
      ).toBeInTheDocument();
    });
  });

  it('renders indexed documents table with metadata and chunk counts', () => {
    render(<DocumentListTable documents={MOCK_DOCUMENTS} />);

    expect(screen.getByTestId('document-list-table')).toBeInTheDocument();
    expect(screen.getByText('AAPL_2026_Q3_10Q.pdf')).toBeInTheDocument();
    expect(screen.getByText('142')).toBeInTheDocument();
    expect(screen.getByText('NVDA_10K_Annual_Filing.pdf')).toBeInTheDocument();
  });

  it('allows Q&A query against documents and renders source citations', async () => {
    render(<ResearchQAInterface />);

    expect(screen.getByTestId('research-qa-interface')).toBeInTheDocument();

    const searchInput = screen.getByPlaceholderText(
      /What does management state regarding Services gross margins/i,
    );
    fireEvent.change(searchInput, {
      target: { value: 'What did management say about AI capex?' },
    });

    const submitBtn = screen.getByText('Search Filings');
    fireEvent.click(submitBtn);

    await waitFor(() => {
      expect(screen.getByTestId('research-qa-result')).toBeInTheDocument();
      expect(screen.getByText(/Direct Filing Citations/i)).toBeInTheDocument();
      expect(screen.getByText(/AAPL_2026_Q3_10Q.pdf/i)).toBeInTheDocument();
    });
  });
});

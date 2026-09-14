import React, { useState, useEffect, useRef } from 'react';
import { Search, ArrowUpRight, ArrowDownRight } from 'lucide-react';
import { CompanyInfo } from '../../types';
import { apiService } from '../../services/api';

interface CompanySearchProps {
  onSelectCompany: (company: CompanyInfo) => void;
  selectedTicker?: string;
  className?: string;
}

export const CompanySearch: React.FC<CompanySearchProps> = ({
  onSelectCompany,
  selectedTicker,
  className = '',
}) => {
  const [searchTerm, setSearchTerm] = useState(selectedTicker || '');
  const [results, setResults] = useState<CompanyInfo[]>([]);
  const [isOpen, setIsOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let active = true;
    const fetchResults = async () => {
      setIsLoading(true);
      try {
        const companies = await apiService.searchCompanies(searchTerm);
        if (active) {
          setResults(companies);
        }
      } finally {
        if (active) setIsLoading(false);
      }
    };

    fetchResults();
    return () => {
      active = false;
    };
  }, [searchTerm]);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (
        dropdownRef.current &&
        !dropdownRef.current.contains(event.target as Node)
      ) {
        setIsOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const handleSelect = (company: CompanyInfo) => {
    setSearchTerm(company.ticker);
    setIsOpen(false);
    onSelectCompany(company);
  };

  return (
    <div className={`relative ${className}`} ref={dropdownRef}>
      <label
        htmlFor="company-search-input"
        className="block text-xs font-semibold uppercase tracking-wider text-slate-700 mb-1.5"
      >
        Target Equity / Company Search
      </label>
      <div className="relative">
        <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none text-slate-400">
          <Search className="w-4 h-4" />
        </div>
        <input
          id="company-search-input"
          type="text"
          value={searchTerm}
          onChange={(e) => {
            setSearchTerm(e.target.value);
            setIsOpen(true);
          }}
          onFocus={() => setIsOpen(true)}
          placeholder="Search by ticker (e.g. AAPL, NVDA, INFY) or name..."
          className="w-full pl-9 pr-4 py-2 text-sm bg-white border border-slate-300 rounded-lg text-slate-900 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500 shadow-sm"
          autoComplete="off"
        />
        {isLoading && (
          <div className="absolute inset-y-0 right-0 pr-3 flex items-center text-xs text-slate-400">
            Searching...
          </div>
        )}
      </div>

      {/* Autocomplete Dropdown */}
      {isOpen && results.length > 0 && (
        <div
          data-testid="company-search-dropdown"
          className="absolute z-30 w-full mt-1.5 bg-white border border-slate-200 rounded-xl shadow-lg max-h-60 overflow-y-auto divide-y divide-slate-100"
        >
          {results.map((comp) => (
            <button
              key={comp.ticker}
              type="button"
              onClick={() => handleSelect(comp)}
              className="w-full px-4 py-2.5 text-left hover:bg-slate-50 flex items-center justify-between transition-colors focus:bg-slate-50 focus:outline-none"
            >
              <div className="flex items-center space-x-3">
                <div className="w-8 h-8 rounded-lg bg-slate-100 text-slate-700 font-bold flex items-center justify-center text-xs">
                  {comp.ticker.slice(0, 2)}
                </div>
                <div>
                  <div className="text-sm font-semibold text-slate-900 flex items-center">
                    {comp.ticker}
                    <span className="ml-2 text-xs font-normal text-slate-500 truncate">
                      {comp.name}
                    </span>
                  </div>
                  <div className="text-[11px] text-slate-400">
                    {comp.sector} • {comp.exchange}
                  </div>
                </div>
              </div>

              <div className="text-right">
                {comp.price > 0 ? (
                  <>
                    <div className="text-sm font-bold text-slate-900">
                      {comp.currency === 'INR' ? '₹' : '$'}
                      {comp.price.toLocaleString()}
                    </div>
                    <div
                      className={`text-xs font-semibold flex items-center justify-end ${
                        comp.change >= 0 ? 'text-emerald-600' : 'text-rose-600'
                      }`}
                    >
                      {comp.change >= 0 ? (
                        <ArrowUpRight className="w-3 h-3 mr-0.5" />
                      ) : (
                        <ArrowDownRight className="w-3 h-3 mr-0.5" />
                      )}
                      {comp.changePercent}%
                    </div>
                  </>
                ) : (
                  <span className="text-[11px] px-2 py-0.5 rounded bg-slate-100 text-slate-600 font-medium">
                    Analyze
                  </span>
                )}
              </div>
            </button>
          ))}
        </div>
      )}
    </div>
  );
};

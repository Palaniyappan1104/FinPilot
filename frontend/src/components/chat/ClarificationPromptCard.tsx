import React, { useState } from 'react';
import { HelpCircle, ArrowRight } from 'lucide-react';

interface ClarificationPromptCardProps {
  questions: string[];
  onSubmitAnswers: (answers: Record<string, string | number>) => void;
  className?: string;
}

export const ClarificationPromptCard: React.FC<ClarificationPromptCardProps> = ({
  questions,
  onSubmitAnswers,
  className = '',
}) => {
  const [answers, setAnswers] = useState<Record<string, string>>({
    time_horizon: '3-5 years',
    risk_tolerance: 'moderate',
    capital_amount: '50000',
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmitAnswers(answers);
  };

  return (
    <div
      data-testid="clarification-prompt-card"
      className={`bg-amber-50/70 border border-amber-200 rounded-xl p-4 shadow-sm ${className}`}
    >
      <div className="flex items-start space-x-2.5">
        <HelpCircle className="w-5 h-5 text-amber-600 flex-shrink-0 mt-0.5" />
        <div className="flex-1">
          <h4 className="text-xs font-bold uppercase tracking-wider text-amber-900">
            Clarification Required for Personalized Assessment
          </h4>
          <p className="text-xs text-amber-800 mt-0.5">
            The Conversation Agent requires additional investor constraints before launching the multi-agent analysis pipeline:
          </p>

          <ul className="mt-2 space-y-1 text-xs text-amber-900 list-disc list-inside">
            {questions.map((q, idx) => (
              <li key={idx} className="font-medium">
                {q}
              </li>
            ))}
          </ul>

          {/* Inline Clarification Input Form */}
          <form onSubmit={handleSubmit} className="mt-3.5 space-y-3 bg-white p-3 rounded-lg border border-amber-200/80">
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
              <div>
                <label
                  htmlFor="clarif-horizon"
                  className="block font-semibold text-slate-700 mb-1"
                >
                  Time Horizon
                </label>
                <select
                  id="clarif-horizon"
                  value={answers.time_horizon}
                  onChange={(e) =>
                    setAnswers({ ...answers, time_horizon: e.target.value })
                  }
                  className="w-full px-2.5 py-1.5 border border-slate-300 rounded text-slate-800 focus:ring-1 focus:ring-amber-500 focus:outline-none"
                >
                  <option value="<1 year">Short-Term (&lt; 1 yr)</option>
                  <option value="1-3 years">Medium-Term (1–3 yrs)</option>
                  <option value="3-5 years">Long-Term (3–5 yrs)</option>
                  <option value="5+ years">Ultra Long-Term (5+ yrs)</option>
                </select>
              </div>

              <div>
                <label
                  htmlFor="clarif-risk"
                  className="block font-semibold text-slate-700 mb-1"
                >
                  Risk Tolerance
                </label>
                <select
                  id="clarif-risk"
                  value={answers.risk_tolerance}
                  onChange={(e) =>
                    setAnswers({ ...answers, risk_tolerance: e.target.value })
                  }
                  className="w-full px-2.5 py-1.5 border border-slate-300 rounded text-slate-800 focus:ring-1 focus:ring-amber-500 focus:outline-none"
                >
                  <option value="conservative">Conservative</option>
                  <option value="moderate">Moderate</option>
                  <option value="aggressive">Aggressive</option>
                </select>
              </div>
            </div>

            <div className="flex justify-end pt-2">
              <button
                type="submit"
                className="inline-flex items-center px-3.5 py-1.5 rounded-lg bg-amber-600 hover:bg-amber-700 text-white font-semibold text-xs transition-colors shadow-sm"
              >
                <span>Submit Clarifications & Resume</span>
                <ArrowRight className="w-3.5 h-3.5 ml-1.5" />
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
};

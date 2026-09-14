import React, { useState, useRef, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Send, Bot, User, Sparkles, ArrowRight, CheckCircle2 } from 'lucide-react';
import { useApp } from '../../context/AppContext';
import { ClarificationPromptCard } from './ClarificationPromptCard';
import { mockApi } from '../../services/mockApi';

interface ChatWindowProps {
  className?: string;
}

export const ChatWindow: React.FC<ChatWindowProps> = ({ className = '' }) => {
  const navigate = useNavigate();
  const {
    chatMessages,
    addChatMessage,
    profile,
    updateProfile,
    setActiveAnalysis,
  } = useApp();

  const [inputValue, setInputValue] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView?.({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [chatMessages]);

  const handleSendMessage = async (textToSend?: string) => {
    const query = (textToSend || inputValue).trim();
    if (!query || isSubmitting) return;

    setInputValue('');
    setIsSubmitting(true);

    // 1. Add user message
    const userMsgId = `msg-${Date.now()}`;
    addChatMessage({
      id: userMsgId,
      role: 'user',
      content: query,
      timestamp: new Date().toISOString(),
    });

    try {
      // 2. Call mockApi
      const result = await mockApi.sendChatMessage(query, profile);
      addChatMessage(result.message);

      if (result.analysisId) {
        setActiveAnalysis({
          analysisId: result.analysisId,
          ticker: profile.ticker || 'AAPL',
          companyName: profile.target_company || 'Apple Inc.',
          status: 'running',
          progressPercent: 35,
          progressStage: 'Specialist Agents executing concurrent research',
          specialistStatuses: {
            technical: 'running',
            fundamental: 'running',
            news: 'running',
            research: 'pending',
            risk: 'pending',
          },
          reportId: result.reportId,
        });
      }
    } catch {
      addChatMessage({
        id: `err-${Date.now()}`,
        role: 'system',
        content: 'Failed to process conversational query. Please try again.',
        timestamp: new Date().toISOString(),
      });
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleClarificationSubmit = async (
    answers: Record<string, string | number>,
  ) => {
    updateProfile({
      time_horizon: String(answers.time_horizon),
      risk_tolerance: String(answers.risk_tolerance),
      capital_amount: Number(answers.capital_amount) || profile.capital_amount,
    });

    setIsSubmitting(true);
    try {
      const result = await mockApi.submitClarification(
        answers,
        profile.ticker || 'AAPL',
      );

      addChatMessage({
        id: `msg-${Date.now()}`,
        role: 'assistant',
        content:
          'Clarification inputs captured successfully. Resuming the multi-agent pipeline: ' +
          'CIO allocating tasks to Technical, Fundamental, News, Research, and Risk specialists.',
        timestamp: new Date().toISOString(),
        analysisId: result.analysisId,
        reportId: result.reportId,
        suggestedActions: ['Monitor Analysis Progress', 'View Final Report'],
      });

      setActiveAnalysis({
        analysisId: result.analysisId,
        ticker: profile.ticker || 'AAPL',
        companyName: profile.target_company || 'Apple Inc.',
        status: 'running',
        progressPercent: 50,
        progressStage: 'Specialist Agents executing concurrent research',
        specialistStatuses: {
          technical: 'completed',
          fundamental: 'running',
          news: 'running',
          research: 'running',
          risk: 'pending',
        },
        reportId: result.reportId,
      });
    } finally {
      setIsSubmitting(false);
    }
  };

  const examplePrompts = [
    'Should I invest ₹1,00,000 in Infosys for 5 years?',
    'Analyze Apple (AAPL) valuation and growth catalysts',
    'Evaluate NVIDIA downside risks before earnings',
  ];

  return (
    <div
      data-testid="chat-window"
      className={`bg-white rounded-xl border border-slate-200 shadow-sm flex flex-col h-[560px] overflow-hidden ${className}`}
    >
      {/* Header */}
      <div className="px-5 py-3.5 border-b border-slate-100 flex items-center justify-between bg-slate-50/50">
        <div className="flex items-center space-x-2">
          <div className="w-6 h-6 rounded-md bg-emerald-600 flex items-center justify-center text-white">
            <Bot className="w-3.5 h-3.5" />
          </div>
          <div>
            <h3 className="text-xs font-bold text-slate-900">
              FinPilot Conversation Agent
            </h3>
            <p className="text-[10px] text-slate-400">
              Query intake, profile elicitation, and specialist routing
            </p>
          </div>
        </div>

        <span className="text-[10px] font-semibold px-2 py-0.5 rounded-full bg-emerald-50 text-emerald-700 border border-emerald-200">
          Agent Ready
        </span>
      </div>

      {/* Messages Scroll Area */}
      <div className="flex-1 p-5 overflow-y-auto space-y-4">
        {chatMessages.map((msg) => (
          <div
            key={msg.id}
            className={`flex ${
              msg.role === 'user' ? 'justify-end' : 'justify-start'
            }`}
          >
            <div
              className={`flex items-start max-w-[85%] space-x-2.5 ${
                msg.role === 'user' ? 'flex-row-reverse space-x-reverse' : 'flex-row'
              }`}
            >
              <div
                className={`w-7 h-7 rounded-full flex items-center justify-center flex-shrink-0 text-xs ${
                  msg.role === 'user'
                    ? 'bg-slate-900 text-white'
                    : msg.role === 'system'
                      ? 'bg-rose-100 text-rose-700'
                      : 'bg-emerald-100 text-emerald-800'
                }`}
              >
                {msg.role === 'user' ? (
                  <User className="w-3.5 h-3.5" />
                ) : (
                  <Bot className="w-3.5 h-3.5" />
                )}
              </div>

              <div className="space-y-2">
                <div
                  className={`p-3.5 rounded-xl text-xs leading-relaxed ${
                    msg.role === 'user'
                      ? 'bg-slate-900 text-white rounded-tr-none'
                      : 'bg-slate-100 text-slate-800 rounded-tl-none border border-slate-200/60'
                  }`}
                >
                  <p className="whitespace-pre-line">{msg.content}</p>
                </div>

                {/* Clarification prompt card if present */}
                {msg.clarificationQuestions &&
                  msg.clarificationQuestions.length > 0 && (
                    <ClarificationPromptCard
                      questions={msg.clarificationQuestions}
                      onSubmitAnswers={handleClarificationSubmit}
                    />
                  )}

                {/* Suggested CTA Buttons */}
                {msg.reportId && (
                  <div className="flex items-center space-x-2 pt-1">
                    <button
                      type="button"
                      onClick={() => navigate(`/reports/${msg.reportId}`)}
                      className="inline-flex items-center px-3 py-1 text-xs font-semibold rounded-lg bg-emerald-600 text-white hover:bg-emerald-700 shadow-sm transition-colors"
                    >
                      <CheckCircle2 className="w-3.5 h-3.5 mr-1" />
                      View Final Report
                    </button>
                    {msg.analysisId && (
                      <button
                        type="button"
                        onClick={() =>
                          navigate(`/analysis/${msg.analysisId}/progress`)
                        }
                        className="inline-flex items-center px-3 py-1 text-xs font-semibold rounded-lg bg-white border border-slate-300 text-slate-700 hover:bg-slate-50 transition-colors"
                      >
                        Monitor Pipeline
                        <ArrowRight className="w-3.5 h-3.5 ml-1" />
                      </button>
                    )}
                  </div>
                )}
              </div>
            </div>
          </div>
        ))}
        {isSubmitting && (
          <div className="flex items-center space-x-2 text-xs text-slate-400 py-1">
            <Bot className="w-4 h-4 text-emerald-600 animate-spin" />
            <span>Agent synthesizing response...</span>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Suggested Prompt Pills */}
      <div className="px-4 py-2 bg-slate-50/80 border-t border-slate-100 flex items-center space-x-2 overflow-x-auto no-scrollbar text-xs">
        <Sparkles className="w-3.5 h-3.5 text-amber-500 flex-shrink-0" />
        <span className="text-[10px] font-semibold text-slate-400 flex-shrink-0">
          Suggestions:
        </span>
        {examplePrompts.map((p, i) => (
          <button
            key={i}
            type="button"
            onClick={() => handleSendMessage(p)}
            className="px-2.5 py-1 rounded-full bg-white border border-slate-200 text-slate-600 hover:text-slate-900 hover:border-slate-300 text-[11px] whitespace-nowrap transition-colors"
          >
            {p}
          </button>
        ))}
      </div>

      {/* Input Box */}
      <form
        onSubmit={(e) => {
          e.preventDefault();
          handleSendMessage();
        }}
        className="p-3 border-t border-slate-200 flex items-center space-x-2 bg-white"
      >
        <input
          type="text"
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          placeholder="Ask a financial research question (e.g. 'Should I invest in Infosys for 5 years?')..."
          className="flex-1 px-3.5 py-2 text-xs bg-slate-50 border border-slate-200 rounded-lg text-slate-900 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:bg-white"
        />
        <button
          type="submit"
          disabled={!inputValue.trim() || isSubmitting}
          aria-label="Send query"
          className="px-4 py-2 rounded-lg bg-slate-900 hover:bg-slate-800 disabled:opacity-40 text-white text-xs font-semibold flex items-center space-x-1.5 transition-colors shadow-sm"
        >
          <span>Send</span>
          <Send className="w-3 h-3" />
        </button>
      </form>
    </div>
  );
};

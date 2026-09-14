import React from 'react';
import {
  MessageSquare,
  Cpu,
  TrendingUp,
  FileSpreadsheet,
  Newspaper,
  ShieldAlert,
  GitMerge,
  FileCheck,
} from 'lucide-react';

export const AgentPipelineVisualizer: React.FC<{ className?: string }> = ({
  className = '',
}) => {
  return (
    <div
      data-testid="agent-pipeline-visualizer"
      className={`bg-slate-900 text-white rounded-xl p-6 border border-slate-800 shadow-sm ${className}`}
    >
      <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-4 border-b border-slate-800 gap-2">
        <div>
          <span className="text-[10px] font-bold tracking-widest text-emerald-400 uppercase">
            Autonomous Multi-Agent Architecture
          </span>
          <h3 className="text-sm font-bold text-white mt-0.5">
            End-to-End Decision Support Pipeline
          </h3>
        </div>
        <span className="text-[11px] text-slate-400">
          Multi-Agent Analysis Pipeline
        </span>
      </div>

      {/* Visual Pipeline Flow */}
      <div className="mt-5 grid grid-cols-1 md:grid-cols-5 gap-3 text-center text-xs">
        {/* Stage 1: Conversation */}
        <div className="bg-slate-800/80 p-3.5 rounded-xl border border-slate-700/60 flex flex-col items-center justify-between space-y-2">
          <div className="w-8 h-8 rounded-lg bg-sky-500/20 text-sky-400 flex items-center justify-center">
            <MessageSquare className="w-4 h-4" />
          </div>
          <div>
            <div className="font-bold text-white text-xs">Conversation</div>
            <div className="text-[10px] text-slate-400 mt-0.5">
              Query intake & intent extraction
            </div>
          </div>
          <span className="text-[9px] font-semibold px-2 py-0.5 rounded bg-slate-900 text-slate-300">
            Intake
          </span>
        </div>

        {/* Stage 2: Clarification & CIO */}
        <div className="bg-slate-800/80 p-3.5 rounded-xl border border-slate-700/60 flex flex-col items-center justify-between space-y-2">
          <div className="w-8 h-8 rounded-lg bg-amber-500/20 text-amber-400 flex items-center justify-center">
            <Cpu className="w-4 h-4" />
          </div>
          <div>
            <div className="font-bold text-white text-xs">CIO Router</div>
            <div className="text-[10px] text-slate-400 mt-0.5">
              Clarification & task routing
            </div>
          </div>
          <span className="text-[9px] font-semibold px-2 py-0.5 rounded bg-slate-900 text-slate-300">
            Supervisor
          </span>
        </div>

        {/* Stage 3: Specialist Pod */}
        <div className="bg-slate-800/80 p-3.5 rounded-xl border border-slate-700/60 flex flex-col items-center justify-between space-y-2">
          <div className="flex space-x-1 text-emerald-400">
            <TrendingUp className="w-3.5 h-3.5" />
            <FileSpreadsheet className="w-3.5 h-3.5" />
            <Newspaper className="w-3.5 h-3.5" />
            <ShieldAlert className="w-3.5 h-3.5" />
          </div>
          <div>
            <div className="font-bold text-white text-xs">5 Specialists</div>
            <div className="text-[10px] text-slate-400 mt-0.5">
              Tech • Fund • News • RAG • Risk
            </div>
          </div>
          <span className="text-[9px] font-semibold px-2 py-0.5 rounded bg-slate-900 text-emerald-400 border border-emerald-900">
            Concurrent
          </span>
        </div>

        {/* Stage 4: Aggregator & Conflict Engine */}
        <div className="bg-slate-800/80 p-3.5 rounded-xl border border-slate-700/60 flex flex-col items-center justify-between space-y-2">
          <div className="w-8 h-8 rounded-lg bg-purple-500/20 text-purple-400 flex items-center justify-center">
            <GitMerge className="w-4 h-4" />
          </div>
          <div>
            <div className="font-bold text-white text-xs">Aggregator</div>
            <div className="text-[10px] text-slate-400 mt-0.5">
              Conflict & synthesis check
            </div>
          </div>
          <span className="text-[9px] font-semibold px-2 py-0.5 rounded bg-slate-900 text-slate-300">
            Consensus
          </span>
        </div>

        {/* Stage 5: Final Report */}
        <div className="bg-slate-800/80 p-3.5 rounded-xl border border-slate-700/60 flex flex-col items-center justify-between space-y-2">
          <div className="w-8 h-8 rounded-lg bg-emerald-500/20 text-emerald-400 flex items-center justify-center">
            <FileCheck className="w-4 h-4" />
          </div>
          <div>
            <div className="font-bold text-white text-xs">Report Generator</div>
            <div className="text-[10px] text-slate-400 mt-0.5">
              Evidence-linked decision dossier
            </div>
          </div>
          <span className="text-[9px] font-semibold px-2 py-0.5 rounded bg-slate-900 text-emerald-300">
            Output
          </span>
        </div>
      </div>
    </div>
  );
};

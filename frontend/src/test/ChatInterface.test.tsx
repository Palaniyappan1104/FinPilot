import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import { ChatWindow } from '../components/chat/ChatWindow';
import { ClarificationPromptCard } from '../components/chat/ClarificationPromptCard';
import { AppContextProvider } from '../context/AppContext';

describe('Chat/Query Interface & Clarification Flow (Phase 16.3)', () => {
  it('renders chat window with Conversation Agent welcome and prompt pills', () => {
    render(
      <BrowserRouter>
        <AppContextProvider>
          <ChatWindow />
        </AppContextProvider>
      </BrowserRouter>,
    );

    expect(screen.getByTestId('chat-window')).toBeInTheDocument();
    expect(screen.getByText('FinPilot Conversation Agent')).toBeInTheDocument();
    expect(
      screen.getByText(/Should I invest ₹1,00,000 in Infosys for 5 years\?/i),
    ).toBeInTheDocument();
  });

  it('renders clarification card and handles submission of missing constraints', () => {
    const handleSubmit = vi.fn();
    render(
      <ClarificationPromptCard
        questions={[
          'What is your intended investment time horizon?',
          'What is your risk tolerance profile?',
        ]}
        onSubmitAnswers={handleSubmit}
      />,
    );

    expect(screen.getByTestId('clarification-prompt-card')).toBeInTheDocument();
    expect(
      screen.getByText(/Clarification Required for Personalized Assessment/i),
    ).toBeInTheDocument();
    expect(
      screen.getByText('What is your intended investment time horizon?'),
    ).toBeInTheDocument();

    const submitBtn = screen.getByText(/Submit Clarifications & Resume/i);
    fireEvent.click(submitBtn);

    expect(handleSubmit).toHaveBeenCalledWith(
      expect.objectContaining({
        time_horizon: expect.any(String),
        risk_tolerance: expect.any(String),
      }),
    );
  });
});

import '@testing-library/jest-dom';
import { vi } from 'vitest';

// Polyfill scrollIntoView for jsdom
window.HTMLElement.prototype.scrollIntoView = vi.fn();

// Polyfill window.print
window.print = vi.fn();

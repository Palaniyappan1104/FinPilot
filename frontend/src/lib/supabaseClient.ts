/**
 * Supabase client initialization for FinPilot.
 * Uses environment variables set in frontend/.env
 */
import { createClient } from '@supabase/supabase-js';

const supabaseUrl = import.meta.env.VITE_SUPABASE_URL as string;
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY as string;

if (!supabaseUrl || supabaseUrl === 'your-supabase-project-url') {
  console.warn(
    '[FinPilot] Supabase URL not configured. Set VITE_SUPABASE_URL in frontend/.env',
  );
}

if (!supabaseAnonKey || supabaseAnonKey === 'your-supabase-anon-key') {
  console.warn(
    '[FinPilot] Supabase Anon Key not configured. Set VITE_SUPABASE_ANON_KEY in frontend/.env',
  );
}

export const supabase = createClient(supabaseUrl, supabaseAnonKey);

/**
 * Supabase client initialization for FinPilot (new_frontend).
 * Uses environment variables set in new_frontend/.env
 */
import { createClient } from '@supabase/supabase-js';

const supabaseUrl = (import.meta.env.VITE_SUPABASE_URL as string) || '';
const supabaseAnonKey = (import.meta.env.VITE_SUPABASE_ANON_KEY as string) || '';

if (!supabaseUrl) {
  console.warn(
    '[FinPilot] Supabase URL not configured. Set VITE_SUPABASE_URL in new_frontend/.env',
  );
}

if (!supabaseAnonKey) {
  console.warn(
    '[FinPilot] Supabase Anon Key not configured. Set VITE_SUPABASE_ANON_KEY in new_frontend/.env',
  );
}

export const supabase = createClient(supabaseUrl, supabaseAnonKey);

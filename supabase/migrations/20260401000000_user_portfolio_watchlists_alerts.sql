-- Per-user portfolio, watchlists, price alerts, and WhatsApp destination.
-- Run in Supabase SQL editor or via CLI. Service role from FastAPI bypasses RLS but policies protect direct anon access.

create table if not exists public.portfolio_holdings (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users (id) on delete cascade,
  ticker text not null,
  quantity double precision not null,
  avg_buy_price double precision not null,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create index if not exists idx_portfolio_holdings_user on public.portfolio_holdings (user_id);

alter table public.portfolio_holdings enable row level security;

create policy "portfolio_select_own" on public.portfolio_holdings
  for select using (auth.uid() = user_id);
create policy "portfolio_insert_own" on public.portfolio_holdings
  for insert with check (auth.uid() = user_id);
create policy "portfolio_update_own" on public.portfolio_holdings
  for update using (auth.uid() = user_id);
create policy "portfolio_delete_own" on public.portfolio_holdings
  for delete using (auth.uid() = user_id);

create table if not exists public.watchlists (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users (id) on delete cascade,
  name text not null,
  tickers text[] not null default '{}',
  updated_at timestamptz not null default now(),
  unique (user_id, name)
);

create index if not exists idx_watchlists_user on public.watchlists (user_id);

alter table public.watchlists enable row level security;

create policy "watchlists_select_own" on public.watchlists
  for select using (auth.uid() = user_id);
create policy "watchlists_insert_own" on public.watchlists
  for insert with check (auth.uid() = user_id);
create policy "watchlists_update_own" on public.watchlists
  for update using (auth.uid() = user_id);
create policy "watchlists_delete_own" on public.watchlists
  for delete using (auth.uid() = user_id);

create table if not exists public.alert_rules (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users (id) on delete cascade,
  ticker text not null,
  rule_type text not null check (rule_type in ('price_above', 'price_below')),
  threshold double precision not null,
  channel_email boolean not null default true,
  channel_whatsapp boolean not null default false,
  enabled boolean not null default true,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create index if not exists idx_alert_rules_user on public.alert_rules (user_id);

alter table public.alert_rules enable row level security;

create policy "alerts_select_own" on public.alert_rules
  for select using (auth.uid() = user_id);
create policy "alerts_insert_own" on public.alert_rules
  for insert with check (auth.uid() = user_id);
create policy "alerts_update_own" on public.alert_rules
  for update using (auth.uid() = user_id);
create policy "alerts_delete_own" on public.alert_rules
  for delete using (auth.uid() = user_id);

create table if not exists public.user_notification_settings (
  user_id uuid primary key references auth.users (id) on delete cascade,
  whatsapp_e164 text,
  updated_at timestamptz not null default now()
);

alter table public.user_notification_settings enable row level security;

create policy "notif_select_own" on public.user_notification_settings
  for select using (auth.uid() = user_id);
create policy "notif_insert_own" on public.user_notification_settings
  for insert with check (auth.uid() = user_id);
create policy "notif_update_own" on public.user_notification_settings
  for update using (auth.uid() = user_id);
create policy "notif_delete_own" on public.user_notification_settings
  for delete using (auth.uid() = user_id);

grant usage on schema public to anon, authenticated, service_role;
grant all on public.portfolio_holdings to service_role;
grant all on public.watchlists to service_role;
grant all on public.alert_rules to service_role;
grant all on public.user_notification_settings to service_role;
grant select, insert, update, delete on public.portfolio_holdings to authenticated;
grant select, insert, update, delete on public.watchlists to authenticated;
grant select, insert, update, delete on public.alert_rules to authenticated;
grant select, insert, update, delete on public.user_notification_settings to authenticated;

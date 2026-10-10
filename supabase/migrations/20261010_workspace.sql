-- Approved by the owner on 2026-10-10. Private server access only.
begin;
create table if not exists public.folio_workspaces (
  user_id text primary key,
  data jsonb not null check (jsonb_typeof(data) = 'object'),
  revision integer not null default 1 check (revision > 0),
  updated_at timestamptz not null default now()
);
alter table public.folio_workspaces enable row level security;
revoke all on public.folio_workspaces from public, anon, authenticated;
grant select, insert, update, delete on public.folio_workspaces to service_role;
commit;

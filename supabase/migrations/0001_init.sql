create extension if not exists pgcrypto;

create table if not exists public.profiles (
  id uuid primary key references auth.users(id) on delete cascade,
  display_name text,
  age_confirmed boolean not null default false,
  onboarding_complete boolean not null default false,
  tone_preference text,
  created_at timestamptz not null default now()
);

create table if not exists public.chat_sessions (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  title text not null default 'Untitled session',
  archived boolean not null default false,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists public.messages (
  id uuid primary key default gen_random_uuid(),
  session_id uuid not null references public.chat_sessions(id) on delete cascade,
  role text not null check (role in ('user', 'assistant')),
  content text not null,
  sentiment text,
  intent text,
  safety_level text,
  resource_payload jsonb not null default '[]'::jsonb,
  model_metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

create table if not exists public.journal_entries (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  title text not null,
  body text not null,
  mood text,
  created_at timestamptz not null default now()
);

create table if not exists public.daily_checkins (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  mood_score integer not null check (mood_score between 0 and 100),
  energy_score integer not null check (energy_score between 0 and 100),
  stress_score integer not null check (stress_score between 0 and 100),
  note text,
  created_at timestamptz not null default now()
);

create table if not exists public.memory_items (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  content text not null,
  category text,
  status text not null default 'pending' check (status in ('pending', 'approved', 'hidden')),
  created_at timestamptz not null default now()
);

create index if not exists chat_sessions_user_updated_idx
on public.chat_sessions (user_id, updated_at desc);

create index if not exists messages_session_created_idx
on public.messages (session_id, created_at asc);

create index if not exists journal_entries_user_created_idx
on public.journal_entries (user_id, created_at desc);

create index if not exists daily_checkins_user_created_idx
on public.daily_checkins (user_id, created_at desc);

create index if not exists memory_items_user_status_created_idx
on public.memory_items (user_id, status, created_at desc);

alter table public.profiles enable row level security;
alter table public.chat_sessions enable row level security;
alter table public.messages enable row level security;
alter table public.journal_entries enable row level security;
alter table public.daily_checkins enable row level security;
alter table public.memory_items enable row level security;

drop policy if exists "profiles_select_own" on public.profiles;
drop policy if exists "profiles_insert_own" on public.profiles;
drop policy if exists "profiles_update_own" on public.profiles;

create policy "profiles_select_own"
on public.profiles
for select
using ((select auth.uid()) = id);

create policy "profiles_insert_own"
on public.profiles
for insert
with check ((select auth.uid()) = id);

create policy "profiles_update_own"
on public.profiles
for update
using ((select auth.uid()) = id)
with check ((select auth.uid()) = id);

drop policy if exists "chat_sessions_select_own" on public.chat_sessions;
drop policy if exists "chat_sessions_insert_own" on public.chat_sessions;
drop policy if exists "chat_sessions_update_own" on public.chat_sessions;
drop policy if exists "chat_sessions_delete_own" on public.chat_sessions;

create policy "chat_sessions_select_own"
on public.chat_sessions
for select
using ((select auth.uid()) = user_id);

create policy "chat_sessions_insert_own"
on public.chat_sessions
for insert
with check ((select auth.uid()) = user_id);

create policy "chat_sessions_update_own"
on public.chat_sessions
for update
using ((select auth.uid()) = user_id)
with check ((select auth.uid()) = user_id);

create policy "chat_sessions_delete_own"
on public.chat_sessions
for delete
using ((select auth.uid()) = user_id);

drop policy if exists "messages_select_own" on public.messages;
drop policy if exists "messages_insert_own" on public.messages;
drop policy if exists "messages_update_own" on public.messages;
drop policy if exists "messages_delete_own" on public.messages;

create policy "messages_select_own"
on public.messages
for select
using (
  exists (
    select 1
    from public.chat_sessions
    where public.chat_sessions.id = messages.session_id
      and public.chat_sessions.user_id = (select auth.uid())
  )
);

create policy "messages_insert_own"
on public.messages
for insert
with check (
  exists (
    select 1
    from public.chat_sessions
    where public.chat_sessions.id = messages.session_id
      and public.chat_sessions.user_id = (select auth.uid())
  )
);

create policy "messages_update_own"
on public.messages
for update
using (
  exists (
    select 1
    from public.chat_sessions
    where public.chat_sessions.id = messages.session_id
      and public.chat_sessions.user_id = (select auth.uid())
  )
)
with check (
  exists (
    select 1
    from public.chat_sessions
    where public.chat_sessions.id = messages.session_id
      and public.chat_sessions.user_id = (select auth.uid())
  )
);

create policy "messages_delete_own"
on public.messages
for delete
using (
  exists (
    select 1
    from public.chat_sessions
    where public.chat_sessions.id = messages.session_id
      and public.chat_sessions.user_id = (select auth.uid())
  )
);

drop policy if exists "journal_entries_select_own" on public.journal_entries;
drop policy if exists "journal_entries_insert_own" on public.journal_entries;
drop policy if exists "journal_entries_update_own" on public.journal_entries;
drop policy if exists "journal_entries_delete_own" on public.journal_entries;

create policy "journal_entries_select_own"
on public.journal_entries
for select
using ((select auth.uid()) = user_id);

create policy "journal_entries_insert_own"
on public.journal_entries
for insert
with check ((select auth.uid()) = user_id);

create policy "journal_entries_update_own"
on public.journal_entries
for update
using ((select auth.uid()) = user_id)
with check ((select auth.uid()) = user_id);

create policy "journal_entries_delete_own"
on public.journal_entries
for delete
using ((select auth.uid()) = user_id);

drop policy if exists "daily_checkins_select_own" on public.daily_checkins;
drop policy if exists "daily_checkins_insert_own" on public.daily_checkins;
drop policy if exists "daily_checkins_update_own" on public.daily_checkins;
drop policy if exists "daily_checkins_delete_own" on public.daily_checkins;

create policy "daily_checkins_select_own"
on public.daily_checkins
for select
using ((select auth.uid()) = user_id);

create policy "daily_checkins_insert_own"
on public.daily_checkins
for insert
with check ((select auth.uid()) = user_id);

create policy "daily_checkins_update_own"
on public.daily_checkins
for update
using ((select auth.uid()) = user_id)
with check ((select auth.uid()) = user_id);

create policy "daily_checkins_delete_own"
on public.daily_checkins
for delete
using ((select auth.uid()) = user_id);

drop policy if exists "memory_items_select_own" on public.memory_items;
drop policy if exists "memory_items_insert_own" on public.memory_items;
drop policy if exists "memory_items_update_own" on public.memory_items;
drop policy if exists "memory_items_delete_own" on public.memory_items;

create policy "memory_items_select_own"
on public.memory_items
for select
using ((select auth.uid()) = user_id);

create policy "memory_items_insert_own"
on public.memory_items
for insert
with check ((select auth.uid()) = user_id);

create policy "memory_items_update_own"
on public.memory_items
for update
using ((select auth.uid()) = user_id)
with check ((select auth.uid()) = user_id);

create policy "memory_items_delete_own"
on public.memory_items
for delete
using ((select auth.uid()) = user_id);

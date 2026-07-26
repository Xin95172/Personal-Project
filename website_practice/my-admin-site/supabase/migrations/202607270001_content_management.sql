-- Replace the starter-shop data model with the trademark-site content model.
drop table if exists public.products cascade;

create table if not exists public.articles (
  id uuid primary key default gen_random_uuid(),
  title text not null check (char_length(title) <= 160),
  excerpt text not null default '' check (char_length(excerpt) <= 500),
  content text not null,
  author_name text not null check (char_length(author_name) <= 80),
  status text not null default 'draft' check (status in ('draft', 'published')),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists public.question_submissions (
  id uuid primary key default gen_random_uuid(),
  pseudonym text not null default '匿名' check (char_length(pseudonym) <= 80),
  question text not null check (char_length(question) <= 2000),
  status text not null default 'pending' check (status in ('pending', 'selected', 'answered', 'rejected')),
  answer text,
  created_at timestamptz not null default now(),
  answered_at timestamptz
);

alter table public.articles enable row level security;
alter table public.question_submissions enable row level security;

create policy "published articles are publicly readable" on public.articles for select using (status = 'published');
create policy "admins manage articles" on public.articles for all using (public.is_admin()) with check (public.is_admin());
create policy "anyone can submit a question" on public.question_submissions for insert with check (true);
create policy "admins manage questions" on public.question_submissions for all using (public.is_admin()) with check (public.is_admin());

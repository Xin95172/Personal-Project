create table if not exists public.site_content (
  id uuid primary key default gen_random_uuid(),
  page_slug text not null check (page_slug in ('home', 'services')),
  section_key text not null,
  title text not null default '' check (char_length(title) <= 200),
  body text not null default '' check (char_length(body) <= 5000),
  cta_label text not null default '' check (char_length(cta_label) <= 80),
  cta_href text not null default '' check (char_length(cta_href) <= 300),
  sort_order integer not null default 0,
  status text not null default 'draft' check (status in ('draft', 'published')),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  unique (page_slug, section_key)
);

alter table public.site_content enable row level security;

create policy "published site content is publicly readable"
  on public.site_content for select using (status = 'published');

create policy "admins manage site content"
  on public.site_content for all using (public.is_admin()) with check (public.is_admin());

insert into public.site_content (page_slug, section_key, title, body, cta_label, cta_href, sort_order, status)
values
  ('home', 'hero', '商標權益，從清楚的第一步開始', '提供商標檢索、申請策略與品牌保護建議，協助你在重要決策前掌握資訊。', '了解服務', '/services', 0, 'published'),
  ('services', 'search', '商標檢索與申請規劃', '協助初步檢索、商品與服務類別評估，以及申請前的方向建議。', '', '', 10, 'published'),
  ('services', 'consulting', '商標爭議與權利諮詢', '針對商標近似、使用風險與權利主張，提供可執行的分析方向。', '', '', 20, 'published'),
  ('services', 'protection', '品牌權利維護', '協助規劃商標註冊後的使用管理與權利維護事宜。', '', '', 30, 'published')
on conflict (page_slug, section_key) do nothing;

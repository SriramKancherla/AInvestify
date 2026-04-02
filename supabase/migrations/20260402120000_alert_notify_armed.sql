-- One notification per threshold crossing: re-arm when price leaves the alert zone.
alter table public.alert_rules
  add column if not exists notify_armed boolean not null default true;

comment on column public.alert_rules.notify_armed is 'When true, the next time the rule is hit we may notify; set false after sending until price leaves the alert zone.';

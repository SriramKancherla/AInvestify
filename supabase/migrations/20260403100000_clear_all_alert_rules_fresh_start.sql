-- One-time wipe of legacy price-alert rules so the app starts fresh with one-shot alerts only.
-- Safe to apply once; new rules are created from the app after this runs.
delete from public.alert_rules;

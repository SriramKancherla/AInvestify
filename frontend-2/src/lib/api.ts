import { supabase } from "@/lib/supabase";

async function withAuthHeaders(init?: RequestInit): Promise<RequestInit> {
  const token = (await supabase.auth.getSession()).data.session?.access_token;
  const headers = new Headers(init?.headers || {});
  if (token) headers.set("Authorization", `Bearer ${token}`);
  return { ...init, headers };
}

export async function apiJson<T>(url: string, init?: RequestInit): Promise<T> {
  const resp = await fetch(url, await withAuthHeaders(init));
  const payload = await resp.json().catch(() => ({}));
  if (!resp.ok) {
    const msg =
      (payload as { detail?: { error?: { message?: string } } }).detail?.error?.message ||
      (payload as { detail?: string }).detail ||
      "Request failed";
    throw new Error(msg);
  }
  return payload as T;
}

export async function apiBlob(url: string, init?: RequestInit): Promise<Blob> {
  const resp = await fetch(url, await withAuthHeaders(init));
  if (!resp.ok) throw new Error("Download failed");
  return await resp.blob();
}

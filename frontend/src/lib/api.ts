/** Public API origin for the Cloudflare Pages build.

Leave `VITE_API_BASE` empty during local `npm run dev`. Vite then proxies
`/api`, `/chart`, and the other API paths to the Python server. The Pages
build sets this to the Render URL, with no trailing slash.
*/
export function apiUrl(path: string): string {
  if (/^https?:\/\//i.test(path)) return path;
  const base = (import.meta.env.VITE_API_BASE || "").trim().replace(/\/$/, "");
  const rel = path.startsWith("/") ? path : `/${path}`;
  return base ? `${base}${rel}` : rel;
}

export async function apiJson<T>(url: string, init?: RequestInit): Promise<T> {
  const resp = await fetch(apiUrl(url), init);
  const payload = await resp.json().catch(() => ({}));
  if (!resp.ok) {
    const msg =
      (payload as { detail?: { error?: { message?: string } } }).detail?.error?.message ||
      (payload as { detail?: string }).detail ||
      "Request failed";
    throw new Error(typeof msg === "string" ? msg : "Request failed");
  }
  return payload as T;
}

export async function apiBlob(url: string, init?: RequestInit): Promise<Blob> {
  const resp = await fetch(apiUrl(url), init);
  if (!resp.ok) throw new Error("Download failed");
  return await resp.blob();
}

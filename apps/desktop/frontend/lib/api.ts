// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
const electronConfig = typeof window !== "undefined" ? window.electronAPI?.getApiConfig?.() : undefined;

export const DEFAULT_API_ENDPOINT = electronConfig?.endpoint || "http://127.0.0.1:8000";
const API_TOKEN = electronConfig?.token || "";
const API_ORIGIN = new URL(DEFAULT_API_ENDPOINT).origin;

const isLaunchApi = (url: URL): boolean =>
  Boolean(API_TOKEN) && url.origin === API_ORIGIN && !url.username && !url.password;

export const apiUrl = (path: string, baseUrl: string = DEFAULT_API_ENDPOINT): string => {
  const cleanBase = baseUrl.replace(/\/+$/, "");
  const cleanPath = path.startsWith("/") ? path : `/${path}`;
  const url = new URL(`${cleanBase}${cleanPath}`);
  if (isLaunchApi(url)) url.searchParams.set("token", API_TOKEN);
  return url.toString();
};

export async function apiFetch(path: string, init?: RequestInit, baseUrl: string = DEFAULT_API_ENDPOINT): Promise<Response> {
  const url = new URL(apiUrl(path, baseUrl));
  if (!isLaunchApi(url)) return fetch(url.toString(), init);
  url.searchParams.delete("token");
  const headers = new Headers(init?.headers);
  headers.set("x-recall-token", API_TOKEN);
  // Do not forward the launch credential if an endpoint redirects elsewhere.
  return fetch(url.toString(), { ...init, headers, redirect: "error" });
}

export const mediaUrl = (filename: string, baseUrl: string = DEFAULT_API_ENDPOINT): string =>
  apiUrl(`/media/${filename}`, baseUrl);

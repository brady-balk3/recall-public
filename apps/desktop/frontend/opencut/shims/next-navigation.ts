// SPDX-License-Identifier: AGPL-3.0-or-later
// Recall shim for next/navigation. OpenCut lives inside Recall's own view
// system, so routing calls are forwarded to whatever Recall registers here.
// Every hook returns a stable object: OpenCut lists them in effect deps.
export interface OpenCutRouterHost {
  projectId: string;
  onNavigate?: (to: string) => void;
}
let host: OpenCutRouterHost = { projectId: "" };
export function setOpenCutRouterHost(next: OpenCutRouterHost) {
  host = next;
  params = { project_id: next.projectId };
}
const router = {
  push: (to: string) => host.onNavigate?.(to),
  replace: (to: string) => host.onNavigate?.(to),
  back: () => host.onNavigate?.("back"),
  forward: () => {},
  refresh: () => {},
  prefetch: () => {},
};
let params = { project_id: "" };
export const useParams = () => params;
export const useRouter = () => router;
export const usePathname = () => "/editor";
const search = new URLSearchParams();
export const useSearchParams = () => search;
export const redirect = (to: string) => host.onNavigate?.(to);
export const notFound = () => { throw new Error("notFound"); };

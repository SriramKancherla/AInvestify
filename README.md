Three folders.

frontend   the website the browser shows
backend    the API, the diffusion model, and the score model
data       reports, slides, papers, old copies, and result files

For the review, open backend/open_these and frontend/open_these.
Those names are shortcuts to the real files. Editing a shortcut edits the real file.

Cloudflare Pages serves the website at a pages.dev address. It does not run the Python API, the diffusion model, or the score model.

Pages project settings:

- Root directory: frontend
- Build command: npm run build
- Build output directory: dist
- NODE_VERSION: 20

There is no 404.html, so Pages serves the app for /app, /models, /stock/AAPL, and /watchlists.
Local npm run dev still proxies API calls to the Python server on this machine.

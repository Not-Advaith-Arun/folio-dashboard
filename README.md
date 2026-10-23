# Folio

**An Indian investment dashboard for holdings, imports, portfolio analytics, and market research.**

A React/TypeScript interface with statement imports, editable holdings, portfolio snapshots, allocation and performance views, market/NAV lookups, and optional private workspace synchronization. A separate Python service implements experimental market/news research workflows.

## Run locally

Node.js 22.13+ and npm.

```sh
git clone https://github.com/Not-Advaith-Arun/folio-dashboard.git
cd folio-dashboard
npm ci
npm run dev
```

Open the loopback URL printed by the development server. The dashboard includes explicitly labeled sample data. Local-development authentication is a development convenience, not a production identity provider.

## What is implemented

- CSV/spreadsheet and supported PDF statement import, field mapping, validation, and backup export.
- Holdings, snapshots, allocation breakdowns, performance analytics, and market/NAV refreshes with explicit missing-data states.
- Optional Supabase workspace storage with authenticated access, version checks, and private user separation.
- A Python research service for historical data, ensemble experiments, news processing, and inspectable evaluations.

## Optional services

Cloud synchronization requires a separately configured Supabase project and server-side `SUPABASE_URL` / `SUPABASE_SECRET_KEY`. It is optional; credentials are not included. The existing schema is under `supabase/`; review it separately before applying anything.

Python research dependencies are listed in `research/requirements.txt`. Install them in a separate environment only if using that service; they include substantial ML dependencies. See the service source for its supported configuration. Broker integrations and hosted authentication are environment-specific and are not promised to work from a bare clone.

## Checks

```sh
npm test
npm run lint
npm run build
```

The JavaScript workflow tests pass in the publication checkout. The current lint command reports pre-existing findings; this documentation update does not claim a clean lint baseline. The research directory contains additional workflow tests. This repository is the expanded Folio implementation; `groww-portfolio` is the older related snapshot.

Research outputs are experimental and depend on data coverage. The project does not promise forecasting accuracy or execute trades automatically.

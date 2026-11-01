# nexaRAG web

A small Next.js front end for the nexaRAG API: sign in, upload PDFs, and chat with answers that
cite the file and page they come from.

```bash
cp .env.example .env.local      # NEXT_PUBLIC_API_URL=http://127.0.0.1:8000
npm install
npm run dev                     # http://localhost:3000
```

The browser calls the API directly, so the API must allow this site's origin:
`CORS_ALLOW_ORIGINS=["https://<your-site>.vercel.app"]` in the API's `.env`.

## Deploy to Vercel

```bash
npx vercel link                                   # once, from this folder
npx vercel env add NEXT_PUBLIC_API_URL production # the public HTTPS URL of the API
npx vercel --prod
```

The API itself (PyTorch models, ingestion worker, Postgres, Qdrant, Redis) does not fit
Vercel's serverless functions; run it with Docker as described in `../docs/deployment.md`.

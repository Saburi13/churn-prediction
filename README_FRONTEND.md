# Frontend (Vite + React)

Development:

1. cd frontend
2. npm install
3. npm run dev

The dev server runs at `http://localhost:5173` and expects the backend at `http://localhost:8000` by default. To change the backend URL set `VITE_API_URL` in an `.env` file in the `frontend/` folder.

Build for production:

1. npm run build
2. Serve `dist/` with any static server or copy into the backend static folder for production.

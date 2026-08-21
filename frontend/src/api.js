const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export async function predict(features, model = 'logreg') {
  const res = await fetch(`${API_BASE}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model, features })
  })
  if (!res.ok) {
    const txt = await res.text()
    throw new Error(txt)
  }
  return res.json()
}

export async function getModelInfo() {
  const res = await fetch(`${API_BASE}/model/info`)
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

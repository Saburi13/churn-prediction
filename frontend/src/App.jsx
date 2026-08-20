import React, { useState } from 'react'
import { predict } from './api'

const defaultFields = {
  gender: 'Male',
  SeniorCitizen: 0,
  Partner: 'No',
  Dependents: 'No',
  tenure: 1,
  PhoneService: 'No',
  MultipleLines: 'No phone service',
  InternetService: 'DSL',
  OnlineSecurity: 'No',
  OnlineBackup: 'No',
  DeviceProtection: 'No',
  TechSupport: 'No',
  StreamingTV: 'No',
  StreamingMovies: 'No',
  Contract: 'Month-to-month',
  PaperlessBilling: 'Yes',
  PaymentMethod: 'Electronic check',
  MonthlyCharges: 29.85,
  TotalCharges: 29.85
}

export default function App() {
  const [fields, setFields] = useState(defaultFields)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)

  function handleChange(e) {
    const { name, value } = e.target
    setFields(prev => ({ ...prev, [name]: value }))
  }

  async function handleSubmit(e) {
    e.preventDefault()
    setLoading(true)
    try {
      const res = await predict({
        ...fields,
        SeniorCitizen: Number(fields.SeniorCitizen),
        tenure: Number(fields.tenure),
        MonthlyCharges: Number(fields.MonthlyCharges),
        TotalCharges: Number(fields.TotalCharges)
      })
      setResult(res)
    } catch (err) {
      setResult({ error: err.message || String(err) })
    } finally {
      setLoading(false)
    }
  }

  const riskPercent = result?.probability ? Math.round(result.probability * 100) : 0
  const riskTone = riskPercent >= 60 ? 'high' : riskPercent >= 35 ? 'medium' : 'low'

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <div className="brand-mark">C</div>
        <div className="brand-copy"><strong>ChurnIQ</strong><span>Retention workspace</span></div>
        <nav className="side-nav" aria-label="Main navigation">
          <a className="nav-item active" href="#predict"><span className="nav-icon">+</span>Predict risk</a>
          <a className="nav-item" href="#insights"><span className="nav-icon">◌</span>Customer insights</a>
          <a className="nav-item" href="#model"><span className="nav-icon">↗</span>Model overview</a>
        </nav>
        <div className="sidebar-footer"><span className="status-dot" /> API connected<span className="api-label">v0.1</span></div>
      </aside>

      <section className="workspace" id="predict">
        <header className="topbar">
          <div><p className="eyebrow">RETENTION OPERATIONS / PREDICTION</p><h1>Customer risk check</h1></div>
          <div className="topbar-meta"><span className="live-dot" /> Live model <span className="avatar">AK</span></div>
        </header>

        <div className="content-grid">
          <section className="form-panel panel">
            <div className="panel-heading"><div><p className="section-kicker">Customer profile</p><h2>Build a risk snapshot</h2></div><span className="step-count">01 / 01</span></div>
            <p className="panel-intro">Tune the profile below to estimate the likelihood of a customer leaving.</p>

            <form onSubmit={handleSubmit}>
              <div className="form-section"><div className="section-title"><span>01</span><div><strong>Account basics</strong><small>Who is this customer?</small></div></div><div className="field-grid">
                <label>Gender<select name="gender" value={fields.gender} onChange={handleChange}><option>Male</option><option>Female</option></select></label>
                <label>Senior citizen<select name="SeniorCitizen" value={fields.SeniorCitizen} onChange={handleChange}><option value="0">No</option><option value="1">Yes</option></select></label>
                <label>Tenure <span className="field-unit">months</span><input name="tenure" type="number" min="0" max="100" value={fields.tenure} onChange={handleChange} /></label>
              </div></div>

              <div className="form-section"><div className="section-title"><span>02</span><div><strong>Service & plan</strong><small>What are they using?</small></div></div><div className="field-grid">
                <label>Internet service<select name="InternetService" value={fields.InternetService} onChange={handleChange}><option>DSL</option><option>Fiber optic</option><option>No</option></select></label>
                <label>Contract<select name="Contract" value={fields.Contract} onChange={handleChange}><option>Month-to-month</option><option>One year</option><option>Two year</option></select></label>
                <label>Payment method<select name="PaymentMethod" value={fields.PaymentMethod} onChange={handleChange}><option>Electronic check</option><option>Mailed check</option><option>Bank transfer (automatic)</option><option>Credit card (automatic)</option></select></label>
              </div></div>

              <div className="form-section"><div className="section-title"><span>03</span><div><strong>Commercial signals</strong><small>Current account value</small></div></div><div className="field-grid">
                <label>Monthly charges <span className="field-unit">USD</span><input name="MonthlyCharges" type="number" min="0" step="0.01" value={fields.MonthlyCharges} onChange={handleChange} /></label>
                <label>Total charges <span className="field-unit">USD</span><input name="TotalCharges" type="number" min="0" step="0.01" value={fields.TotalCharges} onChange={handleChange} /></label>
                <label>Paperless billing<select name="PaperlessBilling" value={fields.PaperlessBilling} onChange={handleChange}><option>Yes</option><option>No</option></select></label>
              </div></div>

              <button className="predict-button" type="submit" disabled={loading}><span>{loading ? 'Calculating risk' : 'Run prediction'}</span><b>↗</b></button>
            </form>
          </section>

          <aside className="insight-column" id="insights">
            <div className={`risk-card panel ${result && !result.error ? riskTone : ''}`}>
              <div className="risk-card-top"><span className="section-kicker">PREDICTION OUTPUT</span><span className="risk-badge">{result && !result.error ? (riskTone === 'high' ? 'Attention' : riskTone === 'medium' ? 'Monitor' : 'Healthy') : 'Awaiting input'}</span></div>
              {result?.error ? <pre className="error">{result.error}</pre> : <>
                <div className="risk-value">{result ? `${riskPercent}%` : '--'}</div>
                <p className="risk-label">Estimated churn probability</p>
                <div className="meter"><span style={{ width: `${riskPercent}%` }} /></div>
                <div className="risk-foot"><span>0%</span><span>50%</span><span>100%</span></div>
                <div className="recommendation"><span className="recommendation-icon">↗</span><div><strong>{result ? (result.prediction ? 'Proactive outreach recommended' : 'Customer looks stable') : 'Run a profile to see guidance'}</strong><p>{result ? (result.prediction ? 'Prioritize this account for a retention conversation.' : 'Continue regular engagement and monitor changes.') : 'The model will translate profile signals into an actionable risk score.'}</p></div></div>
              </>}
            </div>

            <div className="metric-row"><div className="metric-card panel"><span className="metric-icon green">↗</span><small>Model status</small><strong>Ready</strong><span className="metric-caption">Logistic regression</span></div><div className="metric-card panel"><span className="metric-icon blue">◎</span><small>Prediction target</small><strong>Churn</strong><span className="metric-caption">Binary classification</span></div></div>
            <div className="tip-card"><span className="tip-mark">i</span><div><strong>How to read this</strong><p>Scores above 50% indicate the customer is more likely to churn than stay.</p></div></div>
          </aside>
        </div>
        <footer className="page-footer"><span>ChurnIQ workspace</span><span>Secure prediction environment</span></footer>
      </section>
    </main>
  )
}

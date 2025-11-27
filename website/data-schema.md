# Scarlet Sails Dashboard Data Schema

This document defines the data binding contract for all dashboard pages and components.
Each block specifies:
- **Source**: Where data is read from (GitHub file path, API endpoint, or calculated)
- **Fallback**: What to display if data is unavailable
- **Type**: Data format (string, number, array, object, chart data)
- **Required**: Whether this block must be present for dashboard integrity

---

## Dashboard Home (dashboard.html)

### Block: System Status
**ID**: `status-indicator`
**Source**: `/README.md` (parse "Status:" field from main readme)
**Type**: String (status badge)
**Display**: 
  - If found: Display status value (e.g., "Active", "Maintenance", "Stable")
  - If missing: Display "Status: -- (No data)"
**Fallback**: "-- (No data available)"
**Required**: No
**Chart/Widget**: Single badge indicator

### Block: Sharpe Ratio (Latest)
**ID**: `sharpe-latest`
**Source**: `../results/` (look for latest analysis file with sharpe_ratio metric)
**Type**: Number (float, 2 decimals)
**Display**:
  - If found: Display numeric value + trend indicator (↑/↓) if historical data available
  - If missing: Display "Sharpe Ratio: -- (No data)"
**Fallback**: "-- (No data available)"
**Required**: No
**Chart/Widget**: Metric card with gauge

### Block: Win Rate (Latest)
**ID**: `win-rate-latest`
**Source**: `../results/` (parse win_rate from latest result file)
**Type**: Number (percentage 0-100)
**Display**:
  - If found: Display percentage with progress bar
  - If missing: Display "Win Rate: -- (No data)"
**Fallback**: "-- (No data available)"
**Required**: No
**Chart/Widget**: Metric card with progress bar

### Block: Total Trades
**ID**: `total-trades`
**Source**: `../results/` (sum of trades across all result files, or from latest single file)
**Type**: Number (integer)
**Display**:
  - If found: Display count
  - If missing: Display "Total Trades: -- (No data)"
**Fallback**: "-- (No data available)"
**Required**: No
**Chart/Widget**: Metric card

### Block: Performance Over Time
**ID**: `performance-chart`
**Source**: `../results/` (read all result files, extract cumulative_return by date/time)
**Type**: Array of {date, cumulative_return} objects
**Display**:
  - If found: Line chart with cumulative return trend
  - If missing: Display "Performance chart: -- (No historical data)"
**Fallback**: Empty chart with "No data available" message
**Required**: No
**Chart/Widget**: Line chart (Chart.js or similar)

### Block: Model Comparison
**ID**: `model-comparison`
**Source**: `../models.json` or individual model result summaries
**Type**: Array of {model_name, metric1, metric2, ...}
**Display**:
  - If found: Table or bar chart comparing P_rb, P_mi, P_hyb across metrics
  - If missing: Display "Model Comparison: -- (No model data)"
**Fallback**: Empty table/chart with "No model data available"
**Required**: No
**Chart/Widget**: Comparison table or grouped bar chart

---

## Models Page (models.html)

### Block: Model List
**ID**: `model-list`
**Source**: `../models/` (enumerate all .py files matching model pattern)
**Type**: Array of {model_name, description, parameters, formulas}
**Display**:
  - For each model found:
    - Model name
    - Description (from docstring or README)
    - Parameters (from README or code)
    - Formulas (from README or code)
  - If no models found: Display "Models: -- (No models found in repository)"
**Fallback**: "-- (No models available)"
**Required**: No (but critical for tool completeness)
**Chart/Widget**: Accordion/tabs for each model

### Block: P_rb (Range-Bound Strategy)
**ID**: `model-prb`
**Source**: `../models/P_rb.py` + `../README.md` (P_rb section)
**Type**: String (markdown/HTML)
**Display**:
  - If file exists: Show description, parameters, formula, and latest metric
  - If missing: Display "P_rb Model: -- (Model file not found)"
**Fallback**: "-- (Model not available)"
**Required**: No
**Chart/Widget**: Detailed model card

### Block: P_mi (Momentum Indicator Strategy)
**ID**: `model-pmi`
**Source**: `../models/P_mi.py` + `../README.md` (P_mi section)
**Type**: String (markdown/HTML)
**Display**:
  - If file exists: Show description, parameters, formula, and latest metric
  - If missing: Display "P_mi Model: -- (Model file not found)"
**Fallback**: "-- (Model not available)"
**Required**: No
**Chart/Widget**: Detailed model card

### Block: P_hyb (Hybrid Strategy)
**ID**: `model-phyb`
**Source**: `../models/P_hyb.py` + `../README.md` (P_hyb section)
**Type**: String (markdown/HTML)
**Display**:
  - If file exists: Show description, parameters, formula, and latest metric
  - If missing: Display "P_hyb Model: -- (Model file not found)"
**Fallback**: "-- (Model not available)"
**Required**: No
**Chart/Widget**: Detailed model card

---

## API Documentation (api.html)

### Block: API Endpoints
**ID**: `api-endpoints`
**Source**: `../README.md` (API section) or dedicated `api-reference.md`
**Type**: Array of {method, path, description, parameters, response}
**Display**:
  - If data exists: List all endpoints with method (GET/POST/etc), path, and description
  - If missing: Display "API Documentation: -- (No API documentation available)"
**Fallback**: "-- (API documentation not available)"
**Required**: No
**Chart/Widget**: Endpoint list/table

### Block: Example Requests
**ID**: `api-examples`
**Source**: `../README.md` (Examples section) or dedicated examples file
**Type**: Array of {method, code_block}
**Display**:
  - If found: Show code examples
  - If missing: Display "Examples: -- (No examples available)"
**Fallback**: "-- (No examples provided)"
**Required**: No
**Chart/Widget**: Code block display

---

## Analysis & Backtest Results (analysis.html - future)

### Block: Backtest Summary
**ID**: `backtest-summary`
**Source**: `../results/backtest_results.json` or latest backtest file
**Type**: Object {start_date, end_date, total_trades, winning_trades, losing_trades, sharpe_ratio, max_drawdown, cumulative_return}
**Display**:
  - If file exists: Display all summary metrics
  - If missing: Display "Backtest Results: -- (No backtest data)"
**Fallback**: "-- (Backtest data not available)"
**Required**: No
**Chart/Widget**: Summary card or table

### Block: Equity Curve
**ID**: `equity-curve`
**Source**: `../results/` (parse equity/balance over time)
**Type**: Array of {date, equity}
**Display**:
  - If data exists: Line chart showing equity growth
  - If missing: Display "Equity Curve: -- (No equity data)"
**Fallback**: Empty chart with "No equity data available"
**Required**: No
**Chart/Widget**: Line chart

### Block: Drawdown Analysis
**ID**: `drawdown-analysis`
**Source**: `../results/` (calculate max drawdown from equity data)
**Type**: Object {max_drawdown_percent, max_drawdown_duration, current_drawdown}
**Display**:
  - If calculated: Display metrics
  - If missing: Display "Drawdown Analysis: -- (No data)"
**Fallback**: "-- (No drawdown data available)"
**Required**: No
**Chart/Widget**: Metrics card or chart

---

## Data Service Requirements

The `data-service.js` module must:

1. **GitHub Raw Content Access**: Read files from `https://raw.githubusercontent.com/Antihrist-star/ScArlet-Sails/main/` and branch URLs
2. **Fallback Pattern**: All fetch calls include try/catch with explicit fallback values
3. **No Secrets**: Zero authentication tokens, API keys, or credentials in code
4. **Rate Limiting**: Respect GitHub API limits; use raw content URLs for simple file reads
5. **Caching**: Optional localStorage caching with 5-minute TTL to reduce API calls
6. **Error Handling**: All errors display as "-- (No data available)" to user, never expose raw errors
7. **Type Safety**: Validate JSON structure before use; default to missing-data fallback if malformed

---

## Implementation Checklist

- [ ] Create `data-service.js` with all data fetchers
- [ ] Implement `fetchFromGitHub(path, fallback)` function
- [ ] Implement `fetchJSON(path, fallback)` function
- [ ] Implement `parseMetrics(filePath)` for results files
- [ ] Update `dashboard.html` to call data service on page load
- [ ] Add data binding to all metric cards
- [ ] Add fallback/"no data" styling
- [ ] Test with missing data scenarios
- [ ] Document all data blocks in dashboard README

---

## Notes

- All data blocks are **optional** by design—the dashboard gracefully handles missing data
- No data is ever fabricated; if it doesn't exist in the repo, that absence is shown to the user
- URLs are all public GitHub endpoints; no authentication is needed
- The schema serves as the contract between dashboard UI and data-service.js

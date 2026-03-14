// API base URL — override at deploy time via REACT_APP_API_BASE env var.
// For local dev this defaults to the FastAPI dev server.
const API_BASE = process.env.REACT_APP_API_BASE || "http://127.0.0.1:8000";

export default API_BASE;

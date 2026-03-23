import { useEffect } from "react";
import { useNavigate } from "react-router-dom";
import axios from "axios";
import API_BASE from "../config";

export default function RoutineIndex() {
  const navigate = useNavigate();

  useEffect(() => {
    const stored = localStorage.getItem("auraiUser") || sessionStorage.getItem("auraiUser");
    if (!stored) { navigate("/login", { replace: true }); return; }
    let user;
    try { user = JSON.parse(stored); } catch { navigate("/login", { replace: true }); return; }

    axios.get(`${API_BASE}/scans`, { params: { user_id: user.id, limit: 1 } })
      .then(res => {
        const items = res.data?.items || res.data?.scans || [];
        if (items.length > 0) {
          navigate(`/routine/${items[0].id}`, { replace: true });
        } else {
          navigate("/scan", { replace: true }); // no scans yet — send to scan
        }
      })
      .catch(() => navigate("/scan", { replace: true }));
  }, [navigate]);

  return null; // just a redirect, no UI needed
}

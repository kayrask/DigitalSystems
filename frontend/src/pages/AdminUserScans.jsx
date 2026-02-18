import React from "react";
import { useParams } from "react-router-dom";
import History from "./History";

export default function AdminUserScans() {
  const { userId } = useParams();
  const parsed = Number(userId);
  return (
    <History
      adminMode
      userIdOverride={Number.isFinite(parsed) ? parsed : null}
      title="Admin - User Scans"
    />
  );
}


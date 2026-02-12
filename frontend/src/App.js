// src/App.js
import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import "./App.css";            // your main theme
import "./styles/theme.css";   // if you're using this too (optional)

import Account from "./pages/Account";  
import Orders from "./pages/Orders";
import History from "./pages/History";
import Home from "./pages/Home";
import Login from "./pages/Login";
import Register from "./pages/Register";
import Scan from "./pages/Scan";
import Profile from "./pages/Profile";
import AdminAnnotate from "./pages/AdminAnnotate";
import AdminUsers from "./pages/AdminUsers";
import AdminUserScans from "./pages/AdminUserScans";





function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Home />} />         {/* homepage */}
        <Route path="/login" element={<Login />} />   {/* login page */}
        <Route path="/register" element={<Register />} /> {/* register page */}
        <Route path="/scan" element={<Scan />} />     {/* scanner page */}
        <Route path="/account" element={<Account />} />
        <Route path="/orders" element={<Orders />} />
        <Route path="/history" element={<History />} /> 
        <Route path="/admin/history" element={<History adminMode />} />
        <Route path="/admin/users" element={<AdminUsers />} />
        <Route path="/admin/users/:userId/scans" element={<AdminUserScans />} />
        <Route path="/profile" element={<Profile />} />
        <Route path="/admin/annotate/:scanId" element={<AdminAnnotate />} />
      </Routes>
    </Router>
  );
}

export default App;

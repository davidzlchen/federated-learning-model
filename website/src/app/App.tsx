import React from "react";
import { BrowserRouter as Router } from "react-router-dom";

import "./App.css";

import ContentRouter from "../routes/ContentRouter";
import Sidebar from "../sidebar/Sidebar";
import Layout from "antd/es/layout";

function App() {
  return (
    <div className="App">
      <Router>
        <Layout id="parent-container">
          <Sidebar />
          <ContentRouter />
        </Layout>
      </Router>
    </div>
  );
}

export default App;

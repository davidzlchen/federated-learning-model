import React from "react";
import "./Run.css";

import { BrowserRouter as Router, Route, useParams } from "react-router-dom";

function Run() {
  let { runId } = useParams();
  return (
    <div>
      <h3>Requested run id: {runId}</h3>
    </div>
  );
}

export default Run;

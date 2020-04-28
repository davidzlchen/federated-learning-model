import React, { useState } from "react";
// eslint-disable-next-line
import { BrowserRouter as Router, Switch, Route } from "react-router-dom";

import Run from "../runs/Run";
import Runs from "../runs/Runs";
import Dashboard from "../dashboard/Dashboard";

function ContentRouter() {
  const [assignmentsByRunId, setAssignmentsByRunId] = useState({});

  return (
    <Switch>
      <Route path="/about">
        <div>Hello world</div>
      </Route>
      <Route path="/runs/:runId">
        <Run assignmentsByRunId={assignmentsByRunId} />
      </Route>
      <Route path="/runs">
        <Runs />
      </Route>
      <Route path="/">
        <Dashboard
          assignmentsByRunId={assignmentsByRunId}
          setAssignmentsByRunId={setAssignmentsByRunId}
        />
      </Route>
    </Switch>
  );
}

export default ContentRouter;

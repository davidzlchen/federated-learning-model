import React from "react";
// eslint-disable-next-line
import { BrowserRouter as Router, Switch, Route } from "react-router-dom";

import Run from "../runs/Run";
import Runs from "../runs/Runs";
import Dashboard from "../dashboard/Dashboard";

function ContentRouter() {
  return (
    <Switch>
      <Route path="/about">
        <div>Hello world</div>
      </Route>
      <Route path="/runs/:runId">
        <Run />
      </Route>
      <Route path="/runs">
        <Runs />
      </Route>
      <Route path="/">
        <Dashboard />
      </Route>
    </Switch>
  );
}

export default ContentRouter;

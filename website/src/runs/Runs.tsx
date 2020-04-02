import React from "react";
import { Link, useRouteMatch } from "react-router-dom";

function Runs() {
  const match = useRouteMatch();

  return (
    <div>
      <ul>
        <li>
          <Link to={`${match.path}/taskId`}>taskId</Link>
        </li>
      </ul>
    </div>
  );
}

export default Runs;

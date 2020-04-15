import React, { Component } from "react";
import "./Run.css";

import { BrowserRouter as Router, Route, useParams } from "react-router-dom";
import socketIOClient from "socket.io-client";

type RunProps = {};

type RunState = {
  response: Object;
  endpoint: string;
};

class Run extends Component<RunProps, RunState> {
  constructor(props: RunProps) {
    super(props);
    this.state = {
      response: false,
      endpoint: "localhost:5000",
    };
  }

  componentDidMount() {
    const { endpoint } = this.state;
    const socket = socketIOClient(endpoint);
    socket.on("FromAPI", (data: Object) => this.setState({ response: data }));
  }

  render() {
    const { response } = this.state;
    return (
      <div>
        <h3>Requested run id: {response}</h3>
      </div>
    );
  }
}

export default Run;
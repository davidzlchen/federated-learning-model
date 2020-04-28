import React, { Component } from "react";
import { RouteComponentProps, withRouter } from "react-router-dom";
import socketIOClient from "socket.io-client";
import "./Run.css";

import Layout from "antd/es/layout";
import Device from "../device/Device";

import Row from "antd/es/row";
import Col from "antd/es/col";

const { Footer, Content } = Layout;

export type Assignment = {
  device: string;
  topic: string;
  learning_type: string;
};

type PathParamsType = {
  runId: string;
};

type RunProps = RouteComponentProps<PathParamsType> & {
  assignmentsByRunId: { [key: string]: any };
};

type RunState = {
  assignments: Assignment;
  endpoint: string;
  responses: { [key: string]: any };
  runId: string;
};

class Run extends Component<RunProps, RunState> {
  assignment: Array<Assignment>;
  endpoint: string;
  runId: string | undefined;
  socket: SocketIOClient.Socket;

  constructor(props: RunProps) {
    super(props);
    const { runId } = this.props.match.params;

    this.assignment = this.props.assignmentsByRunId[runId];
    this.endpoint = "192.168.1.26:5000";
    this.runId = runId;
    this.socket = socketIOClient(this.endpoint);
  }

  render() {
    let reshapedAssignment = [];
    while (this.assignment.length)
      reshapedAssignment.push(this.assignment.splice(0, 2));

    let deviceRender = reshapedAssignment.map(
      (subarray: Array<Assignment>, outerIndex: number) => {
        let contents = subarray.map((item: Assignment, innerIndex: number) => {
          let index = outerIndex * 2 + innerIndex + 1;
          return (
            <Col sm={24} md={24} lg={24} xl={12}>
              <Device
                index={index}
                key={index}
                name={item.device}
                learningType={item.learning_type}
                socket={this.socket}
                topic={item.topic}
              />
            </Col>
          );
        });
        return <Row gutter={[24, 24]}>{contents}</Row>;
      }
    );

    return (
      <Layout>
        <Content style={{ margin: "16px" }}>
          <h3>Requested run id: {this.runId}</h3>
          {deviceRender}
        </Content>
        <Footer style={{ textAlign: "center" }}>
          Federated Machine Learning Demo ©2020 Created by H21 Senior Design
          Group
        </Footer>
      </Layout>
    );
  }
}

export default withRouter(Run);

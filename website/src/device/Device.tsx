/// <reference types="react-vis-types" />
import React, { Component } from "react";
import { XYPlot, XAxis, YAxis, LineSeriesPoint, LineSeries } from "react-vis";
import "../../node_modules/react-vis/dist/style.css";
import "./Device.css";

import Card from "antd/es/card";
import Spin from "antd/es/spin";
import Descriptions from "antd/es/descriptions";

import Row from "antd/es/row";
import Col from "antd/es/col";

type DeviceProps = {
  index: number;
  learningType: string;
  name: string;
  socket: SocketIOClient.Socket;
  topic: string;
};

type DeviceState = {
  bytes_sent: number;
  test_accuracies: Array<LineSeriesPoint>;
  test_losses: Array<LineSeriesPoint>;
  system_info: SystemInfo | undefined;
};

type SystemInfo = {
  version: string;
  release: string;
  node: string;
  system: string;
  machine: string;
  processor: string;
};

class Device extends Component<DeviceProps, DeviceState> {
  index: number;
  learningType: string;
  name: string;
  socket: SocketIOClient.Socket;
  topic: string;

  constructor(props: DeviceProps) {
    super(props);

    const { index, learningType, name, socket, topic } = this.props;
    this.index = index;
    this.learningType = learningType;
    this.name = name;
    this.socket = socket;
    this.topic = topic;

    this.state = {
      test_accuracies: [],
      test_losses: [],
      bytes_sent: 0,
      system_info: undefined,
    };
  }

  componentDidMount() {
    const { name, socket } = this;
    socket.on(name, (response: string) => {
      const data = JSON.parse(response);

      console.log(data);

      this.setState((prevState) => {
        const test_accuracy = {
          x: prevState.test_accuracies.length,
          y: parseFloat(
            data["model_accuracy"] !== "0., dt" ? data["model_accuracy"] : 0
          ),
        };
        const test_loss = {
          x: prevState.test_losses.length,
          y: parseFloat(data["test_loss"]),
        };
        const bytes_sent = parseInt(data["size"]);
        const { system, node, release, version, machine, processor } = data;
        const system_info = {
          system,
          node,
          release,
          version,
          machine,
          processor,
        };
        return {
          test_accuracies: [...prevState.test_accuracies, test_accuracy],
          test_losses: [...prevState.test_losses, test_loss],
          bytes_sent: prevState.bytes_sent + bytes_sent,
          system_info,
        };
      });
    });
  }

  render() {
    const {
      test_accuracies,
      test_losses,
      bytes_sent,
      system_info,
    } = this.state;
    return (
      <Card title={"Device #" + this.index}>
        {test_accuracies.length && test_losses.length ? (
          <>
            <Row>
              <Col span={12}>
                <XYPlot className="child" width={300} height={300}>
                  <XAxis title="Iteration" />
                  <YAxis title="Test Accuracy" />
                  <LineSeries data={test_accuracies} />
                </XYPlot>
              </Col>
              <Col span={12}>
                <XYPlot className="child" width={300} height={300}>
                  <XAxis title="Iteration" />
                  <YAxis title="Test Loss" />
                  <LineSeries data={test_losses} />
                </XYPlot>
              </Col>
            </Row>
            <Row>
              <Col span={24}>
                <Descriptions bordered>
                  <Descriptions.Item label="Device Name">
                    {this.name}
                  </Descriptions.Item>
                  <Descriptions.Item label="Learning Type">
                    {this.learningType}
                  </Descriptions.Item>
                  <Descriptions.Item label="Topic">
                    {this.topic}
                  </Descriptions.Item>
                  <Descriptions.Item label="Bytes Sent">
                    {bytes_sent}
                  </Descriptions.Item>
                  <Descriptions.Item label="System Info">
                    Node: {system_info?.node}
                    <br />
                    Version: {system_info?.version}
                    <br />
                    Release: {system_info?.release}
                    <br />
                    Machine: {system_info?.machine}
                    <br />
                    Processor: {system_info?.processor}
                  </Descriptions.Item>
                </Descriptions>
              </Col>
            </Row>
          </>
        ) : (
          <Spin />
        )}
      </Card>
    );
  }
}

export default Device;

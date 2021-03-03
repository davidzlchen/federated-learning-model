import React, { useState, SetStateAction, Dispatch } from "react";
import { useHistory } from "react-router-dom";
import "./Dashboard.css";

import Button from "antd/es/button";
import Card from "antd/es/card";
import Divider from "antd/es/divider";
import Form from "antd/es/form";
import InputNumber from "antd/es/input-number";
import Layout from "antd/es/layout";
import Radio from "antd/es/radio";
import { Store } from "rc-field-form/lib/interface";
import Switch from "antd/es/switch";
import Typography from "antd/es/typography";

const { Title } = Typography;
const { Footer, Content } = Layout;

interface Props {
  assignmentsByRunId: { [key: string]: any };
  setAssignmentsByRunId: Dispatch<SetStateAction<{}>>;
}

function Dashboard({ assignmentsByRunId, setAssignmentsByRunId }: Props) {
  let history = useHistory();

  const onFinish = (values: Store) => {
    fetch("http://localhost:5000/executeRun", {
      method: "POST",
      headers: {
        Accept: "application/json",
        "Content-Type": "application/json",
      },
      body: JSON.stringify(values),
    })
      .then((response) => response.json())
      .then((response) => {
        console.log(response);
        const { run_id: runId, assignments } = response;
        setAssignmentsByRunId({ ...assignmentsByRunId, [runId]: assignments });
        history.push("/runs/" + runId);
      });
  };

  const [clusterMode, setClusterMode] = useState(false);
  const numClusters = (
    <Form.Item label="Number of Clusters" name="numClusters">
      <InputNumber min={1} />
    </Form.Item>
  );
  return (
    <Layout className="site-layout">
      <Content style={{ margin: "16px 16px" }}>
        <Form
          name="form"
          initialValues={{ numDevices: 1, operationMode: 0, numClusters: 1 }}
          onFinish={onFinish}
        >
          <Card>
            <Title>Federated Machine Learning Demo</Title>
            <Divider />
            <Form.Item label="Number of devices" name="numDevices">
              <InputNumber min={1} max={6} />
            </Form.Item>
            <Form.Item label="Operation mode" name="operationMode">
              <Radio.Group>
                <Radio value={0}>Centralized</Radio>
                <Radio value={1}>Federated</Radio>
                <Radio value={2}>Personalized</Radio>
              </Radio.Group>
            </Form.Item>
            <Form.Item
              label="Cluster mode"
              name="clusterMode"
              valuePropName="checked"
            >
              <Switch onChange={(e) => setClusterMode(e)} />
            </Form.Item>
            {clusterMode && numClusters}
            {/* 
            <Divider />
            <Form.Item label="Maximum update size (in MB)" name="updateSize">
              <InputNumber min={1} />
            </Form.Item>
            <Form.Item label="Update mode" name="updateMode">
              <Radio.Group>
                <Radio value={0}>Random</Radio>
                <Radio value={1}>Smart Switch</Radio>
              </Radio.Group>
            </Form.Item>
            <Form.Item label="Compression method" name="compressionMode">
              <Radio.Group>
                <Radio value={0}>None</Radio>
                <Radio value={1}>Structured</Radio>
                <Radio value={2}>Sketched</Radio>
              </Radio.Group>
            </Form.Item>
             */}
            <Form.Item>
              <Button type="primary" htmlType="submit">
                Submit
              </Button>
            </Form.Item>
          </Card>
        </Form>
      </Content>
      <Footer style={{ textAlign: "center" }}>
        Federated Machine Learning Demo ©2020 Created by H21 Senior Design Group
      </Footer>
    </Layout>
  );
}

export default Dashboard;

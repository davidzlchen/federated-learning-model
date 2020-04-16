import React from "react";
import { useHistory } from "react-router-dom";
import "./Sidebar.css";

import Sider from "antd/es/layout/Sider";
import Menu from "antd/es/menu";

function Sidebar() {
  let history = useHistory();

  function handleClick(href: string) {
    history.push(href);
  }

  return (
    <Sider>
      <div className="logo" />
      <Menu theme="dark" defaultSelectedKeys={["1"]} mode="inline">
        <Menu.Item key="1" onClick={() => handleClick("/")}>
          <span>Home</span>
        </Menu.Item>
        <Menu.Item key="2" onClick={() => handleClick("/runs")}>
          <span>Runs</span>
        </Menu.Item>
        <Menu.Item key="3" onClick={() => handleClick("/about")}>
          <span>About</span>
        </Menu.Item>
      </Menu>
    </Sider>
  );
}

export default Sidebar;

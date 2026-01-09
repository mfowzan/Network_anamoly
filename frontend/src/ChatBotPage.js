import React from "react";

export default function Chatbot() {
  return (
    <div
      style={{
        display: "flex",
        justifyContent: "center",
        paddingTop: "40px",
        background: "#f5f6fa",
        height: "100vh",
        width: "100%",
      }}
    >
      <div
        style={{
          width: "420px",
          background: "white",
          borderRadius: "12px",
          boxShadow: "0px 4px 12px rgba(0,0,0,0.1)",
          padding: "20px",
        }}
      >
        <h2 style={{ textAlign: "center", marginBottom: "20px" }}>
          Cyber AI Assistant
        </h2>

        {/* Zapier Chatbot Container */}
        <div
          style={{
            width: "100%",
            height: "600px",
            borderRadius: "12px",
            overflow: "hidden",
          }}
        >
          
          <script
            async
            type="module"
            src="https://interfaces.zapier.com/assets/web-components/zapier-interfaces/zapier-interfaces.esm.js"
          ></script>

          <zapier-interfaces-chatbot-embed
            is-popup="false"
            chatbot-id="cmirt53fu007fdhl4oplvd7nq"
            height="600px"
            width="100%"
          ></zapier-interfaces-chatbot-embed>

        </div>
      </div>
    </div>
  );
}

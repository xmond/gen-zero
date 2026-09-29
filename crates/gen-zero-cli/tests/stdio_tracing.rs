//! Tracing must never share the MCP JSON-RPC stdout transport.
use serde_json::{json, Value};
use std::io::Write;
use std::process::{Command, Stdio};

#[test]
fn stdio_keeps_warning_traces_off_protocol_stream() {
    let mut child = Command::new(env!("CARGO_BIN_EXE_gen-zero"))
        .args(["serve", "--mode", "stdio"])
        .env("GENZERO_PYTHON_ENDPOINT", "off")
        .env("RUST_LOG", "warn")
        .env_remove("GENZERO_MOUNT_ASSETS")
        .env_remove("GENZERO_API_KEY")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let request = json!({"jsonrpc":"2.0", "id":1, "method":"tools/call", "params":{
        "name":"zero", "arguments":{"action":"ask", "context":"pick", "candidates":["a","b"]}
    }});
    writeln!(child.stdin.take().unwrap(), "{request}").unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    let response: Value =
        serde_json::from_str(&stdout).expect("stdout contains only one JSON-RPC response");
    assert_eq!(response["id"], 1);
    assert_eq!(response["jsonrpc"], "2.0");
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("request risk not assessed"), "{stderr}");
}

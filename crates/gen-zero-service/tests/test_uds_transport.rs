//! UDS transport over a real socket: MCP handshake, malformed frames sent as
//! raw bytes to the server, and socket file lifecycle.
use gen_zero_service::{
    uds::{bind_uds, UdsClient, MAX_UDS_FRAME_BYTES},
    McpServer,
};
use serde_json::{json, Value};
use std::{io::ErrorKind, path::Path, sync::Arc, time::Duration};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{UnixListener, UnixStream},
    task::JoinHandle,
};

const DEADLINE: Duration = Duration::from_secs(5);

/// Accept one connection and return what `serve_uds_stream` returned.
fn serve_one(listener: UnixListener) -> JoinHandle<std::io::Result<()>> {
    tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        McpServer::new().serve_uds_stream(stream).await
    })
}

async fn read_frame(stream: &mut UnixStream) -> std::io::Result<Value> {
    let length = tokio::time::timeout(DEADLINE, stream.read_u32())
        .await
        .expect("server hung before the reply header")? as usize;
    let mut body = vec![0; length];
    tokio::time::timeout(DEADLINE, stream.read_exact(&mut body))
        .await
        .expect("server hung before the reply body")?;
    Ok(serde_json::from_slice(&body).expect("reply frame is not JSON"))
}

async fn write_raw(stream: &mut UnixStream, header: u32, body: &[u8]) {
    stream.write_all(&header.to_be_bytes()).await.unwrap();
    stream.write_all(body).await.unwrap();
}

async fn ping(client: &mut UdsClient, id: i64) {
    let request = format!(r#"{{"jsonrpc":"2.0","method":"ping","id":{id}}}"#);
    let reply = tokio::time::timeout(DEADLINE, client.request(request.as_bytes()))
        .await
        .expect("ping hung")
        .unwrap();
    let value: Value = serde_json::from_slice(&reply).unwrap();
    assert_eq!(value["id"], id);
    assert_eq!(value["result"], json!({}));
}

async fn assert_eof(stream: &mut UnixStream) {
    let mut byte = [0_u8; 1];
    let read = tokio::time::timeout(DEADLINE, stream.read(&mut byte))
        .await
        .expect("server kept the connection open");
    assert_eq!(read.unwrap(), 0, "expected EOF after the error frame");
}

#[tokio::test]
async fn mcp_handshake_skips_notification_reply_and_keeps_framing() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("mcp.sock");
    let task = serve_one(UnixListener::bind(&path).unwrap());
    let mut client = UdsClient::connect(&path).await.unwrap();

    let reply = client
        .request(br#"{"jsonrpc":"2.0","method":"initialize","id":1,"params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"uds-test","version":"0"}}}"#)
        .await
        .unwrap();
    let value: Value = serde_json::from_slice(&reply).unwrap();
    assert_eq!(value["id"], 1);
    assert_eq!(value["result"]["serverInfo"]["name"], "gen-zero");

    client
        .send_notification(br#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#)
        .await
        .unwrap();
    // If the server had written a reply (empty or not) for the notification,
    // this request would read that frame instead of its own reply.
    ping(&mut client, 2).await;
    let tools = client
        .request(br#"{"jsonrpc":"2.0","method":"tools/list","id":3}"#)
        .await
        .unwrap();
    let tools: Value = serde_json::from_slice(&tools).unwrap();
    assert_eq!(tools["id"], 3);
    assert!(
        tools["result"]["tools"]
            .as_array()
            .is_some_and(|t| !t.is_empty()),
        "{tools}"
    );

    drop(client);
    tokio::time::timeout(DEADLINE, task)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
}

#[tokio::test]
async fn non_json_frame_gets_parse_error_and_connection_survives() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("mcp.sock");
    let task = serve_one(UnixListener::bind(&path).unwrap());
    let mut stream = UnixStream::connect(&path).await.unwrap();

    let garbage = b"\xff\xfe\x00not json at all {{{";
    write_raw(&mut stream, garbage.len() as u32, garbage).await;
    let error = read_frame(&mut stream).await.unwrap();
    assert_eq!(error["jsonrpc"], "2.0");
    assert_eq!(error["error"]["code"], -32700, "{error}");
    assert_eq!(error["id"], Value::Null);

    let truncated_json = br#"{"jsonrpc":"2.0","method":"ping","id":"#;
    write_raw(&mut stream, truncated_json.len() as u32, truncated_json).await;
    let error = read_frame(&mut stream).await.unwrap();
    assert_eq!(error["error"]["code"], -32700, "{error}");

    let valid = br#"{"jsonrpc":"2.0","method":"ping","id":9}"#;
    write_raw(&mut stream, valid.len() as u32, valid).await;
    let reply = read_frame(&mut stream).await.unwrap();
    assert_eq!(reply["id"], 9);
    assert_eq!(reply["result"], json!({}));

    drop(stream);
    tokio::time::timeout(DEADLINE, task)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
}

#[tokio::test]
async fn invalid_length_prefix_gets_error_frame_then_close() {
    for header in [0_u32, u32::MAX, MAX_UDS_FRAME_BYTES as u32 + 1] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mcp.sock");
        let task = serve_one(UnixListener::bind(&path).unwrap());
        let mut stream = UnixStream::connect(&path).await.unwrap();

        write_raw(&mut stream, header, b"").await;
        let error = read_frame(&mut stream).await.unwrap();
        assert_eq!(error["error"]["code"], -32600, "header {header}: {error}");
        assert_eq!(error["id"], Value::Null);
        assert_eof(&mut stream).await;

        let served = tokio::time::timeout(DEADLINE, task).await.unwrap().unwrap();
        assert_eq!(
            served.unwrap_err().kind(),
            ErrorKind::InvalidData,
            "header {header}"
        );
    }
}

#[tokio::test]
async fn truncated_frame_body_closes_without_hanging() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("mcp.sock");
    let task = serve_one(UnixListener::bind(&path).unwrap());
    let mut stream = UnixStream::connect(&path).await.unwrap();

    // Claim 100 bytes, send 10, then half-close the write side.
    write_raw(&mut stream, 100, b"{\"jsonrpc\"").await;
    stream.shutdown().await.unwrap();
    let served = tokio::time::timeout(DEADLINE, task).await.unwrap().unwrap();
    assert_eq!(served.unwrap_err().kind(), ErrorKind::UnexpectedEof);
    assert_eof(&mut stream).await;

    // A header cut short (2 of 4 bytes) is also an error, not a hang.
    let task = serve_one(UnixListener::bind(dir.path().join("b.sock")).unwrap());
    let mut stream = UnixStream::connect(dir.path().join("b.sock"))
        .await
        .unwrap();
    stream.write_all(&[0, 0]).await.unwrap();
    stream.shutdown().await.unwrap();
    let served = tokio::time::timeout(DEADLINE, task).await.unwrap().unwrap();
    assert_eq!(served.unwrap_err().kind(), ErrorKind::UnexpectedEof);
}

#[tokio::test]
async fn client_refuses_empty_and_oversized_frames_locally() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("mcp.sock");
    let _listener = UnixListener::bind(&path).unwrap();
    let mut client = UdsClient::connect(&path).await.unwrap();
    assert_eq!(
        client.request(&[]).await.unwrap_err().kind(),
        ErrorKind::InvalidData
    );
    let oversized = vec![b' '; MAX_UDS_FRAME_BYTES + 1];
    assert_eq!(
        client
            .send_notification(&oversized)
            .await
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidData
    );
}

async fn wait_for_socket(path: &Path) {
    use std::os::unix::fs::FileTypeExt;
    let deadline = tokio::time::Instant::now() + DEADLINE;
    loop {
        if std::fs::symlink_metadata(path).is_ok_and(|m| m.file_type().is_socket())
            && UnixStream::connect(path).await.is_ok()
        {
            return;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "listener never came up"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

#[tokio::test]
async fn run_uds_replaces_stale_socket_and_removes_it_on_shutdown() {
    use std::os::unix::fs::PermissionsExt;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("mcp.sock");
    // A socket file whose listener is gone, as a killed server leaves it.
    drop(std::os::unix::net::UnixListener::bind(&path).unwrap());
    assert!(path.exists(), "stale socket fixture missing");

    let (stop, stopped) = tokio::sync::oneshot::channel::<()>();
    let server = Arc::new(McpServer::new());
    let run = tokio::spawn({
        let path = path.clone();
        async move {
            server
                .run_uds_until(&path, async {
                    stopped.await.ok();
                })
                .await
        }
    });
    wait_for_socket(&path).await;
    let mode = std::fs::metadata(&path).unwrap().permissions().mode() & 0o777;
    assert_eq!(mode, 0o600);

    let mut client = UdsClient::connect(&path).await.unwrap();
    ping(&mut client, 1).await;

    // A second server on the same live path must be refused, not take over.
    let second = Arc::new(McpServer::new())
        .run_uds_until(&path, std::future::pending())
        .await;
    assert!(second.unwrap_err().to_string().contains("live listener"));
    assert!(
        path.exists(),
        "refused bind must not remove the live socket"
    );
    ping(&mut client, 2).await;

    stop.send(()).unwrap();
    tokio::time::timeout(DEADLINE, run)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert!(!path.exists(), "socket file left behind after shutdown");
}

#[tokio::test]
async fn bind_refuses_a_path_that_is_not_a_socket() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("mcp.sock");
    std::fs::write(&path, b"keep me").unwrap();
    let error = bind_uds(&path).unwrap_err();
    assert!(error.to_string().contains("not a socket"), "{error}");
    assert_eq!(std::fs::read(&path).unwrap(), b"keep me");
}

#[tokio::test]
async fn dropping_the_guard_removes_the_socket_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("mcp.sock");
    let (listener, guard) = bind_uds(&path).unwrap();
    assert!(path.exists());
    drop(listener);
    drop(guard);
    assert!(!path.exists());
}

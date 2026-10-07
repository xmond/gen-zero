//! Length-prefixed local MCP JSON-RPC transport. Each frame is a 4-byte
//! big-endian length followed by one JSON-RPC message. The server reuses one
//! scratch buffer per connection and parses each frame in that buffer; socket
//! I/O still copies between kernel and user space, and replies are allocated.
//!
//! JSON-RPC notifications get no reply frame, as on stdio. The server never
//! sends a zero-length frame.
use crate::{error::ServiceError, McpServer};
use std::{
    future::Future,
    path::{Path, PathBuf},
    sync::Arc,
};

pub const MAX_UDS_FRAME_BYTES: usize = 2 * 1024 * 1024;

#[cfg(unix)]
use crate::server::shutdown_signal;
#[cfg(unix)]
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{UnixListener, UnixStream},
};

#[cfg(unix)]
fn check_frame_length(length: usize) -> std::io::Result<()> {
    if length == 0 || length > MAX_UDS_FRAME_BYTES {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("invalid UDS frame length {length} (allowed 1..={MAX_UDS_FRAME_BYTES})"),
        ));
    }
    Ok(())
}

#[cfg(unix)]
async fn write_frame(stream: &mut UnixStream, body: &[u8]) -> std::io::Result<()> {
    check_frame_length(body.len())?;
    stream.write_all(&(body.len() as u32).to_be_bytes()).await?;
    stream.write_all(body).await
}

/// A connected client channel.
#[cfg(unix)]
pub struct UdsClient(UnixStream);

#[cfg(unix)]
impl UdsClient {
    pub async fn connect(path: impl AsRef<Path>) -> std::io::Result<Self> {
        Ok(Self(UnixStream::connect(path).await?))
    }

    /// Send a request and wait for its reply frame. Use
    /// [`Self::send_notification`] for messages without an `id`: the server
    /// sends no reply to them, so this call would wait forever.
    pub async fn request(&mut self, request: &[u8]) -> std::io::Result<Vec<u8>> {
        write_frame(&mut self.0, request).await?;
        let length = self.0.read_u32().await? as usize;
        check_frame_length(length)?;
        let mut reply = vec![0; length];
        self.0.read_exact(&mut reply).await?;
        Ok(reply)
    }

    /// Send a JSON-RPC notification. No reply frame follows.
    pub async fn send_notification(&mut self, notification: &[u8]) -> std::io::Result<()> {
        write_frame(&mut self.0, notification).await
    }
}

/// Removes the socket file when dropped, so a normal shutdown, an accept
/// error, or a cancelled serve future leaves no file behind. A killed process
/// leaves the file; the next [`bind_uds`] removes it.
#[cfg(unix)]
#[derive(Debug)]
pub struct UdsSocketGuard(PathBuf);

#[cfg(unix)]
impl Drop for UdsSocketGuard {
    fn drop(&mut self) {
        match std::fs::remove_file(&self.0) {
            Ok(()) => tracing::info!(path = %self.0.display(), "MCP UDS socket removed"),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => {
                tracing::error!(path = %self.0.display(), %error, "could not remove MCP UDS socket")
            }
        }
    }
}

/// Bind the listener at `path` with mode 0600. A stale socket file left by a
/// dead server is removed with a warning. A socket that still accepts
/// connections, or a path that is not a socket, is refused: replacing either
/// would take over another server or destroy an unrelated file.
#[cfg(unix)]
pub fn bind_uds(path: &Path) -> Result<(UnixListener, UdsSocketGuard), ServiceError> {
    use std::os::unix::fs::{FileTypeExt, PermissionsExt};
    match std::fs::symlink_metadata(path) {
        Ok(meta) if meta.file_type().is_socket() => {
            match std::os::unix::net::UnixStream::connect(path) {
                Ok(_) => {
                    return Err(ServiceError::SafetyRejected(format!(
                        "UDS path {} is served by a live listener",
                        path.display()
                    )));
                }
                Err(e) if e.kind() == std::io::ErrorKind::ConnectionRefused => {
                    tracing::warn!(path = %path.display(), "removing stale MCP UDS socket");
                    std::fs::remove_file(path)?;
                }
                Err(e) => {
                    return Err(ServiceError::SafetyRejected(format!(
                        "UDS path {} cannot be connected or safely replaced: {e}",
                        path.display()
                    )));
                }
            }
        }
        Ok(_) => {
            return Err(ServiceError::SafetyRejected(format!(
                "UDS path {} exists and is not a socket",
                path.display()
            )));
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error.into()),
    }
    let listener = UnixListener::bind(path)?;
    let guard = UdsSocketGuard(path.to_path_buf());
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))?;
    Ok((listener, guard))
}

#[cfg(unix)]
impl McpServer {
    /// Serve MCP on a Unix socket until Ctrl-C or SIGTERM. The socket file is
    /// removed on return.
    pub async fn run_uds(self: Arc<Self>, path: impl AsRef<Path>) -> Result<(), ServiceError> {
        self.run_uds_until(path, shutdown_signal()).await
    }

    /// Serve MCP on a Unix socket until `shutdown` resolves. Connections
    /// already accepted keep running on their own tasks.
    pub async fn run_uds_until(
        self: Arc<Self>,
        path: impl AsRef<Path>,
        shutdown: impl Future<Output = ()>,
    ) -> Result<(), ServiceError> {
        if self.auth_token.is_some() {
            return Err(ServiceError::SafetyRejected(
                "UDS transport does not support bearer tokens; use a private socket path".into(),
            ));
        }
        self.check_semantic(None).await?;
        let path = path.as_ref();
        let (listener, _guard) = bind_uds(path)?;
        tracing::info!(path = %path.display(), "MCP UDS listener ready");
        tokio::pin!(shutdown);
        loop {
            let stream = tokio::select! {
                () = &mut shutdown => {
                    tracing::info!(path = %path.display(), "MCP UDS listener shutting down");
                    return Ok(());
                }
                accepted = listener.accept() => accepted?.0,
            };
            let server = Arc::clone(&self);
            tokio::spawn(async move {
                if let Err(error) = server.serve_uds_stream(stream).await {
                    tracing::warn!(%error, "MCP UDS connection failed");
                }
            });
        }
    }

    /// Process an accepted stream until the peer closes it. A frame with an
    /// invalid length gets a JSON-RPC -32600 error frame, then the connection
    /// is closed with an error, because the stream can no longer be framed.
    /// A frame that is not valid JSON gets a -32700 error frame and the
    /// connection stays open.
    pub async fn serve_uds_stream(&self, mut stream: UnixStream) -> std::io::Result<()> {
        let mut frame = Vec::with_capacity(4096);
        loop {
            let mut header = [0_u8; 4];
            let first = stream.read(&mut header[..1]).await?;
            if first == 0 {
                return Ok(());
            }
            stream.read_exact(&mut header[1..]).await?;
            let length = u32::from_be_bytes(header) as usize;
            if let Err(error) = check_frame_length(length) {
                let reply = crate::server::jsonrpc_error(
                    serde_json::Value::Null,
                    -32600,
                    format!("Invalid Request: {error}"),
                )
                .to_string();
                write_frame(&mut stream, reply.as_bytes()).await?;
                return Err(error);
            }
            frame.resize(length, 0);
            stream.read_exact(&mut frame).await?;
            let response = self.handle_jsonrpc_frame(&mut frame).await;
            if response.is_empty() {
                continue;
            }
            if response.len() > MAX_UDS_FRAME_BYTES {
                tracing::error!(bytes = response.len(), "MCP UDS reply exceeds frame limit");
                let reply = crate::server::jsonrpc_error(
                    serde_json::Value::Null,
                    -32603,
                    format!(
                        "Internal error: reply of {} bytes exceeds the UDS frame limit",
                        response.len()
                    ),
                )
                .to_string();
                write_frame(&mut stream, reply.as_bytes()).await?;
                continue;
            }
            write_frame(&mut stream, response.as_bytes()).await?;
        }
    }
}

// ---------------------------------------------------------------------------
// Non-Unix fallback stubs (Windows, etc.)
// ---------------------------------------------------------------------------

#[cfg(not(unix))]
pub struct UdsClient;

#[cfg(not(unix))]
#[derive(Debug)]
pub struct UdsSocketGuard(PathBuf);

#[cfg(not(unix))]
pub fn bind_uds(_path: &Path) -> Result<((), UdsSocketGuard), ServiceError> {
    Err(ServiceError::SafetyRejected(
        "Unix domain socket (UDS) transport is only supported on Unix platforms".into(),
    ))
}

#[cfg(not(unix))]
impl McpServer {
    /// Serve MCP on a Unix socket (unsupported on non-Unix platforms).
    pub async fn run_uds(self: Arc<Self>, _path: impl AsRef<Path>) -> Result<(), ServiceError> {
        Err(ServiceError::SafetyRejected(
            "Unix domain socket (UDS) transport is only supported on Unix platforms".into(),
        ))
    }

    /// Serve MCP on a Unix socket until shutdown (unsupported on non-Unix platforms).
    pub async fn run_uds_until(
        self: Arc<Self>,
        _path: impl AsRef<Path>,
        _shutdown: impl Future<Output = ()>,
    ) -> Result<(), ServiceError> {
        Err(ServiceError::SafetyRejected(
            "Unix domain socket (UDS) transport is only supported on Unix platforms".into(),
        ))
    }
}
